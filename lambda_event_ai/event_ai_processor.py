import time
from datetime import timedelta, datetime, timezone
from decimal import Decimal
from collections import defaultdict

import boto3
from PIL import Image, ImageDraw, ImageFont

from amazon_kinesis_video_consumer_library.kinesis_video_streams_parser import KvsConsumerLibrary
from amazon_kinesis_video_consumer_library.kinesis_video_fragment_processor import KvsFragementProcessor
from sagemaker_controller import SageMakerController
from event_clip import EventClip, draw_objects_on_frame, draw_ppes_on_frame, draw_plates_on_frame
from alarm_controller import AlarmController
from lambda_logging import base_logger


class StreamNotReadyError(Exception):
    """ Exception raised when the KVS stream is not ready """
    def __init__(self, message="KVS stream is not ready yet. Please try again later."):
        self.message = message
        super().__init__(self.message)

class EventAIProcessor:

    def __init__(self, aws_region, stream_ai_config, s3_bucket = "motion-event-snapshots", logger=None):
        self.logger = logger or base_logger

        self.stream_ai_config = stream_ai_config
        self.kvs_fragment_processor = KvsFragementProcessor()
        
        self.session = boto3.Session(region_name=aws_region) 
        self.kvs_client = self.session.client("kinesisvideo")        
        
        self.sagemaker_inference = SageMakerController(aws_region, "sagemaker-inference-server-endpoint", logger=self.logger)
        self.alarm_controller = AlarmController(logger=self.logger)
        self.event_clip = EventClip(aws_region, s3_bucket, logger=self.logger)

        self.dynamodb = boto3.resource("dynamodb", region_name=aws_region)
        self.event_table = self.dynamodb.Table("events")
        self.event_predictions_table = self.dynamodb.Table("event_predictions")
        
        self.last_fragment_timestamp = None
        self.stream_exception = None

        self.triggered_alarm_rules = []


    def process_frames(self, stream_name : str, 
                       event_timestamp : datetime,
                       lambda_context : dict,
                       attempt_number : int = 0,
                       n_seconds : int = 10, 
                       one_in_frames_ratio : int = 10):
        
        self.stream_exception = None
        self.attempt_number = attempt_number
        self.stream_name = stream_name
        self.event_timestamp = event_timestamp
        self.n_seconds = n_seconds
        self.event_clip.clear_frames()
        self.event_clip_filepath = f"{stream_name}/{event_timestamp.isoformat()}.mp4"
        self.lambda_context = lambda_context
        
        self.one_in_frames_ratio = one_in_frames_ratio
        self.stream_end_timestamp = event_timestamp + timedelta(seconds=n_seconds)

        get_media_endpoint = self._get_data_endpoint(stream_name, 'GET_MEDIA')
        kvs_media_client = self.session.client('kinesis-video-media', endpoint_url=get_media_endpoint)

        # I'm assuming that the start of processing the event is before the first fragment arrives
        self._create_event_record(
            stream_name,
            self.event_timestamp,
            self.lambda_context,
            self.attempt_number
        )
        
        # TODO: if the KVS stream is still not there, it should be a good idea to wait a bit. Is my retry working?
        # TODO: If I read the fragments too fast, will I end up reaching on_stream_read_complete before I get all the frames?
        # time.sleep(5)  # Wait for the stream to be available TODO: improve that
        self.stream_start_timestamp = event_timestamp # TODO: might use the buffer before the event timestamp
        
        get_media_response = kvs_media_client.get_media(
            StreamName=stream_name,
            StartSelector={
                'StartSelectorType': 'PRODUCER_TIMESTAMP',
                'StartTimestamp': self.stream_start_timestamp
            }
        )

        self.stream = KvsConsumerLibrary(kvs_media_client, 
                                    get_media_response, 
                                    self.on_fragment_arrived, 
                                    self.on_stream_read_complete, 
                                    self.on_stream_read_exception)
        self.n_fragments = 0
        self.n_frames = 0
        
        self.predictions_summary = {
            "seen_classes": set(),
            "seen_plates": set(),
            "object_detection_stats": defaultdict(lambda: {
                "n_detections": 0,
                "total_confidence": 0.0,
                "max_confidence": 0.0,
            }),
            "plate_recognition_stats": defaultdict(lambda: {
                "n_detections": 0,
                "total_confidence": 0.0,
                "max_confidence": 0.0,
                "total_ocr_confidence": 0.0,
                "max_ocr_confidence": 0.0
            }),
            "n_frames": 0
        }
        self.stream.start()

    
    def on_fragment_arrived(self, stream_object, fragment_bytes, fragment_dom, fragment_receive_duration):
        """ Called when a new KVS fragment arrives """

        frag_tags = self.kvs_fragment_processor.get_fragment_tags(fragment_dom)
        frag_number = frag_tags["AWS_KINESISVIDEO_FRAGMENT_NUMBER"]
        frag_producer_timestamp = datetime.fromtimestamp(float(frag_tags["AWS_KINESISVIDEO_PRODUCER_TIMESTAMP"]), tz=timezone.utc)
        frag_server_timestamp = datetime.fromtimestamp(float(frag_tags["AWS_KINESISVIDEO_SERVER_TIMESTAMP"]), tz=timezone.utc)

        self.logger.info(f'Fragment arrived: #{frag_number}, with timestamp {frag_producer_timestamp}')
        if frag_producer_timestamp < self.stream_start_timestamp:
            self.logger.info(f'Fragment timestamp {frag_producer_timestamp} is before start timestamp {self.stream_start_timestamp}. Skipping.')
            return
        
        if frag_producer_timestamp > self.stream_end_timestamp:
            self.logger.info(f'Fragment timestamp {frag_producer_timestamp} is after end timestamp {self.stream_end_timestamp}. Stopping stream.')
            self.stream.stop_thread()
            return
        
        ndarray_frames, n_frames_in_fragment = self.kvs_fragment_processor.get_frames_as_ndarray(fragment_bytes, self.one_in_frames_ratio)
        if (self.n_frames == 0): self.logger.info(f"Will get one in {self.one_in_frames_ratio} frames, total of {len(ndarray_frames)} from fragment with {n_frames_in_fragment} frames")
        for i, frame in enumerate(ndarray_frames):
            image_pil = Image.fromarray(frame) # since the frame is read using pyav, it is in RGB format 
            
            # image_pil.save("frame_to_send.png", format="PNG")  # Save the frame for debugging
            frame_timestamp = self._compute_frame_timestamp(frag_producer_timestamp, n_frames_in_fragment, i, self.one_in_frames_ratio)

            # run predictions, whichever is configured for this stream
            model_predictions = self.sagemaker_inference.predict(image_pil, 
                                                                self.stream_ai_config["models"], 
                                                                self.stream_ai_config.get("per_model_params", None), 
                                                                verbose=(self.n_frames == 0))
            # draw predictions on the frame for video/alarms
            drawn_image_pil = image_pil.copy()
            drawn_image_pil = self._draw_predictions_on_frame(drawn_image_pil, self.stream_ai_config["models"], model_predictions["results"])
            
            # add the frame to the event clip
            self.event_clip.add_frame(drawn_image_pil)
            self.n_frames += 1          

            # update the seen classes and stats (average confidence, max confidence, etc.)
            self._update_predictions_summary(self.stream_ai_config["models"], model_predictions["results"])

            # check if an alarm must be send
            self.triggered_alarm_rules.extend(self.alarm_controller.check_alarms(self.stream_name, self.predictions_summary, model_predictions["results"], self.stream_start_timestamp,
                                                                    frame_timestamp, image_pil, drawn_image_pil, exclude_rules=self.triggered_alarm_rules, verbose=False))


            # store the predictions in DynamoDB
            self._store_predictions(
                device_id=self.stream_name,
                frame_timestamp=frame_timestamp,
                event_timestamp=self.event_timestamp,
                fragment_number=frag_number,
                fragment_producer_timestamp=frag_producer_timestamp,
                fragment_server_timestamp=frag_server_timestamp,
                predictions=model_predictions
            )
            

        self.n_fragments += 1
        self.last_fragment_timestamp = frag_producer_timestamp

    def _draw_predictions_on_frame(self, frame_pil, models_to_draw, model_predictions):
        for model in models_to_draw:
            detections = model_predictions[model].get("detections", [])
            if model.startswith("object_detection"):
                frame_pil = draw_objects_on_frame(frame_pil, detections)
                if "ppe" in model:
                    ppe_detections = [det for det in detections if det['label'] == 'person' and 'ppe' in det]
                    if ppe_detections:
                        frame_pil = draw_ppes_on_frame(frame_pil, ppe_detections)
                if "lpr" in model:
                    plates = [
                        plate
                        for det in detections
                        if "license_plate" in det and len(det["license_plate"]) > 0
                        for plate in det["license_plate"]
                    ]
                    frame_pil = draw_plates_on_frame(frame_pil, plates)

            elif model == "ppe_classification":
                raise Exception("No way to draw PPE classification on a frame and you probably should not be calling this model directly.")
                # frame_pil = draw_ppes_on_frame(frame_pil, detections)
            elif model == "license_plate_recognition":
                frame_pil = draw_plates_on_frame(frame_pil, detections)

        return frame_pil
    
    def _store_predictions(self, device_id, frame_timestamp, event_timestamp, fragment_number, fragment_producer_timestamp, fragment_server_timestamp, predictions):
        item = {
            "device_id": device_id,
            "frame_timestamp": frame_timestamp.isoformat(),
            "event_timestamp": event_timestamp.isoformat(),
            "fragment_number": fragment_number,
            "fragment_producer_timestamp": fragment_producer_timestamp.isoformat(),
            "fragment_server_timestamp": fragment_server_timestamp.isoformat(),
            "predictions": self._convert_floats_to_decimals(predictions),
        }
        try:
            self.event_predictions_table.put_item(Item=item)
        except Exception as e:
            self.logger.info(f"Error storing detections in DynamoDB: {e}")

    def _create_object_detection_from_ppe(self, person_detection, ppe_level, ppe_confidence):
        """ Create an object detection from a person with PPE detection """
        return {
            "label": f"person_ppe_{ppe_level}",
            "confidence": person_detection["confidence"] * ppe_confidence,
            "bbox": person_detection["bbox"],
        }


    def _update_predictions_summary(self, models, predictions):
        self.predictions_summary["n_frames"] += 1
        for model in models:
            detections = predictions[model].get("detections", [])
            self.predictions_summary["seen_classes"].update([det['label'] for det in detections])
            for det in detections:
                self._update_detection_stats(det)

            if model.startswith("object_detection_then"):
                for det in detections:
                    if "ppe" in det: # also conside the person with PPE a different object
                        ppe_det = self._create_object_detection_from_ppe(det, det["ppe"]["ppe_level"], det["ppe"]["confidence"])
                        self.predictions_summary["seen_classes"].add(ppe_det['label'])
                        self._update_detection_stats(ppe_det)
                    elif "license_plate" in det and len(det["license_plate"]) > 0:
                        self.predictions_summary["seen_classes"].add("car_plate")
                        # if there are multiple plates, we will add them all
                        for plate in det["license_plate"]:
                            if plate["ocr_text"] not in self.predictions_summary["seen_plates"]:
                                self.predictions_summary["seen_plates"].add(plate["ocr_text"])
                            self._update_plate_stats(plate)
        

    def _update_detection_stats(self, det):
        cls, confidence = det['label'], det['confidence']
        stats = self.predictions_summary["object_detection_stats"][cls]
        stats["n_detections"] += 1
        stats["total_confidence"] += confidence
        stats["max_confidence"] = max(stats["max_confidence"], confidence)
        stats["avg_confidence"] = stats["total_confidence"] / stats["n_detections"]

    def _update_plate_stats(self, plate):
        text = plate['ocr_text']
        stats = self.predictions_summary["plate_recognition_stats"][text]
        stats["n_detections"] += 1
        stats["total_confidence"] += plate['confidence']
        stats["max_confidence"] = max(stats["max_confidence"], plate['confidence'])
        stats["avg_confidence"] = stats["total_confidence"] / stats["n_detections"]
        stats["total_ocr_confidence"] += plate['ocr_confidence']
        stats["max_ocr_confidence"] = max(stats["max_ocr_confidence"], plate['ocr_confidence'])
        stats["avg_ocr_confidence"] = stats["total_ocr_confidence"] / stats["n_detections"]

    def _compute_frame_timestamp(self, fragment_timestamp, n_frames_in_fragment, frame_index, one_in_frames_ratio):
        if self.last_fragment_timestamp is None:
            # since we don´t have the last timestamp, we need to assume that the fragment duration base on a gop = 26
            fragment_duration = n_frames_in_fragment / 26
        else:
            fragment_duration = (fragment_timestamp - self.last_fragment_timestamp).total_seconds()
        
        frame_duration = fragment_duration / n_frames_in_fragment
        original_frame_index = frame_index * one_in_frames_ratio
        frame_timestamp = fragment_timestamp + timedelta(seconds=frame_duration * original_frame_index)
        return frame_timestamp

    def on_stream_read_complete(self, stream):
        self.logger.info(f'Read Media on stream: {stream} Completed')
        if self.event_clip.frames == []:
            self.stream_exception = StreamNotReadyError("No frames were processed. The stream might not be ready yet.")
            self.logger.info(self.stream_exception.message)
            self.stream.stop_thread()
            return

        self.logger.info('Writing event clip to S3')
        start_time = time.time()
        self.event_clip.send_clip_to_s3(self.event_clip_filepath)
        elapsed_ms = (time.time() - start_time) * 1000
        self.logger.info(f"send_clip_to_s3 took {elapsed_ms:.2f} ms")

        # ---- Build finals from self.predictions_summary ----
        ps = self.predictions_summary

        # object stats -> drop running totals, keep avg/max/n_detections
        final_object_detection_stats = {}
        for cls, s in ps["object_detection_stats"].items():
            final_object_detection_stats[cls] = {
                "n_detections": s["n_detections"],
                "avg_confidence": s.get("avg_confidence", 0.0),
                "max_confidence": s.get("max_confidence", 0.0),
            }

        # plate stats -> drop running totals, keep avg/max + OCR avg/max + n_detections
        final_plate_stats = {}
        for plate_text, s in ps["plate_recognition_stats"].items():
            final_plate_stats[plate_text] = {
                "n_detections": s["n_detections"],
                "avg_confidence": s.get("avg_confidence", 0.0),
                "max_confidence": s.get("max_confidence", 0.0),
                "avg_ocr_confidence": s.get("avg_ocr_confidence", 0.0),
                "max_ocr_confidence": s.get("max_ocr_confidence", 0.0),
            }
        
        # ---- Logging summary (unchanged style, but sourced from ps) ----
        self.logger.info("Event summary:")
        self.logger.info(f"  Stream name: {self.stream_name}")
        self.logger.info(f"  Event timestamp: {self.event_timestamp.isoformat()}")
        self.logger.info(f"  Attempt number: {self.attempt_number}")
        self.logger.info(f"  Stream start timestamp: {self.stream_start_timestamp.isoformat()}")
        self.logger.info(f"  Stream end timestamp: {self.stream_end_timestamp.isoformat()}")
        self.logger.info(f"  Number of processed fragments: {self.n_fragments}")
        self.logger.info(f"  Number of processed frames: {self.n_frames}")
        self.logger.info(f"  Video S3 key: {self.event_clip_filepath}")
        self.logger.info(f"  Seen classes: {list(ps['seen_classes'])}")
        self.logger.info(f"  Seen plates: {list(ps['seen_plates'])}")
        self.logger.info(f"  Object detection stats:")
        for cls, s in final_object_detection_stats.items():
            self.logger.info(f"    - {cls}: avg_conf={s['avg_confidence']}, max_conf={s['max_confidence']}, n_detections={s['n_detections']}")
        self.logger.info(f"  Plate detection stats:")
        for plate, s in final_plate_stats.items():
            self.logger.info(f"    - {plate}: avg_conf={s['avg_confidence']}, max_conf={s['max_confidence']}, "
                f"ocr_avg_conf={s['ocr_avg_confidence']}, ocr_max_conf={s['ocr_max_confidence']}, n_detections={s['n_detections']}")

        # ---- Keep DynamoDB payload shape the same; just source values from ps ----
        self._update_event_record(
            stream_name=self.stream_name,
            event_timestamp=self.event_timestamp,
            stream_start_timestamp=self.stream_start_timestamp,
            stream_end_timestamp=self.stream_end_timestamp,
            n_processed_fragments=self.n_fragments,
            n_processed_frames=self.n_frames,
            video_key=self.event_clip_filepath,
            seen_classes=list(ps["seen_classes"]),
            seen_plates=list(ps["seen_plates"]),
            detection_stats=final_object_detection_stats,
            plate_recognition_stats=final_plate_stats
        )
    def on_stream_read_exception(self, stream_name, error, exc_info=None):
        self.logger.error(f'Error on stream: {stream_name}', exc_info=exc_info)

        # if error is `pyav` can not handle the given uri assumes stream is not ready yet and tries again later
        if isinstance(error, OSError) and str(error) == '`pyav` can not handle the given uri.':
            self.stream_exception = StreamNotReadyError("KVS stream is not ready: PyAV cannot handle URI")
        else:
            self.stream_exception = error
        self.stream.stop_thread()

    def _get_data_endpoint(self, stream_name, api_name):
        """ Fetch KVS data endpoint """
        response = self.kvs_client.get_data_endpoint(StreamName=stream_name, APIName=api_name)
        return response['DataEndpoint']
        
    def _create_event_record(self, stream_name, event_timestamp, lambda_context, attempt_number):
        """ Create a record in the events table """
        start = time.time()
        self.event_table.put_item(
            Item={
                "device_id": stream_name,
                "event_timestamp": event_timestamp.isoformat(),
                "processing_start_timestamp": datetime.now(timezone.utc).isoformat(),
                **lambda_context,
                "process_attempt": attempt_number,
            }
        )
        elapsed_ms = (time.time() - start) * 1000
        # this record should be updated at the end of the processing
        self.logger.info(f"Created event record for {stream_name} at {event_timestamp}. Took {elapsed_ms:.2f} ms")

    def _update_event_record(self, 
                             stream_name, 
                             event_timestamp, 
                             stream_start_timestamp, 
                             stream_end_timestamp, 
                             n_processed_fragments, 
                             n_processed_frames, 
                             video_key, 
                             seen_classes,
                             seen_plates,
                             detection_stats,
                             plate_recognition_stats):
        """ Update the event record in the events table """
        start_time = time.time()
        self.event_table.update_item(
            Key={
            "device_id": stream_name,
            "event_timestamp": event_timestamp.isoformat(),
            },
            UpdateExpression="SET processing_end_timestamp = :processing_end_timestamp, "
                    "stream_start_timestamp = :stream_start_timestamp, "
                    "stream_end_timestamp = :stream_end_timestamp, "
                    "n_processed_fragments = :n_processed_fragments, "
                    "n_processed_frames = :n_processed_frames, "
                    "video_key = :video_key, "
                    "seen_classes = :seen_classes, "
                    "seen_plates = :seen_plates, "
                    "detection_stats = :detection_stats, "
                    "plate_recognition_stats = :plate_recognition_stats",
            ExpressionAttributeValues={
            ":processing_end_timestamp": datetime.now(timezone.utc).isoformat(),
            ":stream_start_timestamp": stream_start_timestamp.isoformat(),
            ":stream_end_timestamp": stream_end_timestamp.isoformat(),
            ":n_processed_fragments": n_processed_fragments,
            ":n_processed_frames": n_processed_frames,
            ":video_key": video_key,
            ":seen_classes": seen_classes,
            ":seen_plates": list(seen_plates),
            ":detection_stats": self._convert_floats_to_decimals(detection_stats),
            ":plate_recognition_stats": self._convert_floats_to_decimals(plate_recognition_stats)
            }
        )
        elapsed_ms = (time.time() - start_time) * 1000
        self.logger.info(f"event_table.update_item took {elapsed_ms:.2f} ms")

    def _convert_floats_to_decimals(self, obj):
            if isinstance(obj, float):
                return Decimal(str(obj))  # str() preserves precision
            elif isinstance(obj, list):
                return [self._convert_floats_to_decimals(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: self._convert_floats_to_decimals(v) for k, v in obj.items()}
            else:
                return obj

