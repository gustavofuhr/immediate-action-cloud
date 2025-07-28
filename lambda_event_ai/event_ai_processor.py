import time
from datetime import timedelta, datetime, timezone
from decimal import Decimal
from collections import defaultdict


import boto3
from PIL import Image, ImageDraw, ImageFont

from amazon_kinesis_video_consumer_library.kinesis_video_streams_parser import KvsConsumerLibrary
from amazon_kinesis_video_consumer_library.kinesis_video_fragment_processor import KvsFragementProcessor
# from dfine_controller_ort import DFINEControllerORT
from sagemaker_controller import SageMakerController
from event_clip import EventClip, draw_boxes_on_frame




CLASSES_TO_STORE = ['person', 'car_plate','bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
    


class StreamNotReadyError(Exception):
    """ Exception raised when the KVS stream is not ready """
    def __init__(self, message="KVS stream is not ready yet. Please try again later."):
        self.message = message
        super().__init__(self.message)

class EventAIProcessor:

    def __init__(self, aws_region, s3_bucket = "motion-event-snapshots"):
        self.kvs_fragment_processor = KvsFragementProcessor()
        
        self.session = boto3.Session(region_name=aws_region) # TODO: do I need aws_region?
        self.kvs_client = self.session.client("kinesisvideo")        
        
        # DEBUG: TODO: WARNING: This is a temporary solution for testing
        self.sagemaker_inference = SageMakerController(aws_region, "sagemaker-inference-server-endpoint-loadtest")

        self.event_clip = EventClip(aws_region, s3_bucket)

        self.dynamodb = boto3.resource("dynamodb", region_name=aws_region)
        self.event_table = self.dynamodb.Table("events")
        self.event_detections_table = self.dynamodb.Table("event_detections")
        self.event_plate_recognitions_table = self.dynamodb.Table("event_plate_recognitions")
        self.last_fragment_timestamp = None
        self.stream_exception = None

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
        self.seen_classes = set()
        self.seen_plates = set()
        self.detection_stats = defaultdict(lambda: {
            "total_confidence": 0.0,
            "max_confidence": 0.0,
            "n_frames": 0
        })
        self.plate_recognition_stats = defaultdict(lambda: {
            "total_confidence": 0.0,
            "max_confidence": 0.0,
            "n_frames": 0,
            "ocr_max_confidence": 0.0,
            "ocr_total_confidence": 0.0
        })
        self.stream.start()

    
    def on_fragment_arrived(self, stream_name, fragment_bytes, fragment_dom, fragment_receive_duration):
        """ Called when a new KVS fragment arrives """

        frag_tags = self.kvs_fragment_processor.get_fragment_tags(fragment_dom)
        frag_number = frag_tags["AWS_KINESISVIDEO_FRAGMENT_NUMBER"]
        frag_producer_timestamp = datetime.fromtimestamp(float(frag_tags["AWS_KINESISVIDEO_PRODUCER_TIMESTAMP"]), tz=timezone.utc)
        frag_server_timestamp = datetime.fromtimestamp(float(frag_tags["AWS_KINESISVIDEO_SERVER_TIMESTAMP"]), tz=timezone.utc)

        print(f'Fragment arrived: #{frag_number}, with timestamp {frag_producer_timestamp}')
        if frag_producer_timestamp < self.stream_start_timestamp:
            print(f'Fragment timestamp {frag_producer_timestamp} is before start timestamp {self.stream_start_timestamp}. Skipping.')
            return
        
        if frag_producer_timestamp > self.stream_end_timestamp:
            print(f'Fragment timestamp {frag_producer_timestamp} is after end timestamp {self.stream_end_timestamp}. Stopping stream.')
            self.stream.stop_thread()
            return
        
        ndarray_frames, n_frames_in_fragment = self.kvs_fragment_processor.get_frames_as_ndarray(fragment_bytes, self.one_in_frames_ratio)
        if (self.n_frames == 0): print(f"Will get one in {self.one_in_frames_ratio} frames, total of {len(ndarray_frames)} from fragment with {n_frames_in_fragment} frames")
        for i, frame in enumerate(ndarray_frames):
            image_pil = Image.fromarray(frame)  
            
            frame_timestamp = self._compute_frame_timestamp(frag_producer_timestamp, n_frames_in_fragment, i, self.one_in_frames_ratio)
            # print("Frame timestamp: ", frame_timestamp)


            self.sagemaker_inference.predict(image_pil, models, params)

            # first general object detection
            filtered_detections, raw_detections = self.sagemaker_inference.detect_objects(image_pil, classes_to_detect=DETECTION_CLASS_COLORS.keys(), threshold=0.5, include_ppe_classification=True, verbose=(self.n_frames == 0))
            self._store_detections(self.stream_name, frame_timestamp, self.event_timestamp, frag_number, frag_producer_timestamp, frag_server_timestamp, raw_detections)            
            if filtered_detections:
                image_pil = self._draw_detections_on_frame(image_pil, filtered_detections)
            
            ppe_detections = [det for det in filtered_detections if det['label'] == 'person' and 'ppe' in det]
            if ppe_detections:
                image_pil = self._draw_ppes_on_frame(image_pil, ppe_detections)

            for det in filtered_detections:
                self._update_detection_stats(det)
                self.seen_classes.add(det['label'])
                if det['label'] == 'person' and "ppe" in det:
                    ppe_label = "person_ppe_" + det["ppe"]["ppe_level"]
                    self.seen_classes.add(ppe_label)
                    self._update_detection_stats({
                        'label': ppe_label,
                        'score': det["ppe"]["confidence"]
                    })

            # license plate recognition
            filtered_plates, raw_plates = self.detector.detect_plates(image_pil, threshold=0.7, ocr_theshold=0.7, verbose=(self.n_frames == 0))
            for plate in filtered_plates:
                self._update_plate_stats(plate)
                self.seen_plates.add(plate['ocr_text'])

                # a plate is also an object
                self.seen_classes.add("plate")
                self._update_detection_stats({
                    'label': 'car_plate',
                    'score': plate['score']
                })

            if filtered_plates:
                self._store_plates(self.stream_name, frame_timestamp, self.event_timestamp, frag_number, frag_producer_timestamp, frag_server_timestamp, raw_plates)
                image_pil = self._draw_plates_on_frame(image_pil, filtered_plates)
            
            self.event_clip.add_frame(image_pil)
            self.n_frames += 1

        self.n_fragments += 1
        self.last_fragment_timestamp = frag_producer_timestamp


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

    def _update_plate_stats(self, plate):
        text = plate['ocr_text']
        if text not in self.seen_plates:
            self.seen_plates.add(text)
        stats = self.plate_recognition_stats[text]
        stats["total_confidence"] += plate['score']
        stats["max_confidence"] = max(stats["max_confidence"], plate['score'])
        stats["n_frames"] += 1
        stats["ocr_total_confidence"] += plate['ocr_confidence']
        stats["ocr_max_confidence"] = max(stats["ocr_max_confidence"], plate['ocr_confidence'])

    def _update_detection_stats(self, det):
        cls, score = det['label'], det['score']
        stats = self.detection_stats[cls]
        stats["total_confidence"] += score
        stats["max_confidence"] = max(stats["max_confidence"], score)
        stats["n_frames"] += 1


    def on_stream_read_complete(self, stream):
        print(f'Read Media on stream: {stream} Completed')
        if self.event_clip.frames == []:
            self.stream_exception = StreamNotReadyError("No frames were processed. The stream might not be ready yet.")
            print(self.stream_exception.message)
            self.stream.stop_thread()
        else:
            
            print('Writing event clip to S3')
            start_time = time.time()
            self.event_clip.send_clip_to_s3(self.event_clip_filepath)
            elapsed_ms = (time.time() - start_time) * 1000
            print(f"send_clip_to_s3 took {elapsed_ms:.2f} ms")

            # Finalize detection_stats (compute average)
            final_detection_stats = {}
            for cls, stats in self.detection_stats.items():
                final_detection_stats[cls] = {
                    "avg_confidence": round(stats["total_confidence"] / stats["n_frames"], 4),
                    "max_confidence": round(stats["max_confidence"], 4),
                    "n_frames": stats["n_frames"]
                }
            final_plate_stats = {}
            for plate, stats in self.plate_recognition_stats.items():
                final_plate_stats[plate] = {
                    "avg_confidence": round(stats["total_confidence"] / stats["n_frames"], 4),
                    "max_confidence": round(stats["max_confidence"], 4),
                    "ocr_avg_confidence": round(stats["ocr_total_confidence"] / stats["n_frames"], 4),
                    "ocr_max_confidence": round(stats["ocr_max_confidence"], 4),
                    "n_frames": stats["n_frames"]
                }

            print("Event summary:")
            print(f"  Stream name: {self.stream_name}")
            print(f"  Event timestamp: {self.event_timestamp.isoformat()}")
            print(f"  Attempt number: {self.attempt_number}")
            print(f"  Stream start timestamp: {self.stream_start_timestamp.isoformat()}")
            print(f"  Stream end timestamp: {self.stream_end_timestamp.isoformat()}")
            print(f"  Number of processed fragments: {self.n_fragments}")
            print(f"  Number of processed frames: {self.n_frames}")
            print(f"  Video S3 key: {self.event_clip_filepath}")
            print(f"  Seen classes: {list(self.seen_classes)}")
            print(f"  Seen plates: {list(self.seen_plates)}")
            print(f"  Detection stats:")
            for cls, stats in final_detection_stats.items():
                print(f"    - {cls}: avg_conf={stats['avg_confidence']}, max_conf={stats['max_confidence']}, n_frames={stats['n_frames']}")
            print(f"  Plate detection stats:")
            for plate, stats in final_plate_stats.items():
                print(f"    - {plate}: avg_conf={stats['avg_confidence']}, max_conf={stats['max_confidence']}, "
                      f"ocr_avg_conf={stats['ocr_avg_confidence']}, ocr_max_conf={stats['ocr_max_confidence']}, n_frames={stats['n_frames']}")

            self._update_event_record(
                stream_name=self.stream_name,
                event_timestamp=self.event_timestamp,
                stream_start_timestamp=self.stream_start_timestamp,
                stream_end_timestamp=self.stream_end_timestamp,
                n_processed_fragments=self.n_fragments,
                n_processed_frames=self.n_frames,
                video_key=self.event_clip_filepath,
                seen_classes=list(self.seen_classes),
                seen_plates=list(self.seen_plates),
                detection_stats=final_detection_stats,
                plate_recognition_stats=final_plate_stats
            )

    def on_stream_read_exception(self, stream_name, error):
        print(f'Error on stream: {stream_name}')
        print(repr(error))
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
    
    def _draw_ppes_on_frame(self, frame_pil, detections):
        def label_fn(det):
            label_map = {
                "full": "PPE: full",
                "upper": "PPE: upper",
                "bottom": "PPE: bottom",
                "noppe": "no PPE",
                "na": "PPE: n/a"
            }
            if det["label"] == "person" and "ppe" in det:
                level = det["ppe"]["ppe_level"]
                return label_map.get(level, "PPE: unknown") + f" ({det['ppe']['confidence']:.2f})"
        
        def color_fn(det):
            color_map = {
                "full": (0, 255, 0),  # green
                "upper":  (225, 165, 0),  # orange
                "bottom": (225, 165, 0),  # orange
                "noppe": (210, 0, 0),  # dark red
                "na": (128, 128, 128)  # gray
            }
            if det["label"] == "person" and "ppe" in det:
                level = det["ppe"]["ppe_level"]
                return color_map.get(level, (255, 0, 0))
            
        return draw_boxes_on_frame(
            frame_pil,
            detections,
            label_fn=label_fn,
            color_fn=color_fn,
            font_size=14,
            text_color=(255, 255, 255),
            padding=2,
            label_position="bottom",
            invisible_box=True  # do not draw the bounding box for PPE
        )
        
    
    def _draw_detections_on_frame(self, frame_pil, detections):
        return draw_boxes_on_frame(
            frame_pil,
            detections,
            label_fn=lambda det: f"{det['label']}: {det['score']:.2f}",
            color_fn=lambda d: DETECTION_CLASS_COLORS[d['label']],
            font_size=18
        )

    def _draw_plates_on_frame(self, frame_pil, plates):
        return draw_boxes_on_frame(
            frame_pil,
            plates,
            label_fn=lambda d: f"{d['ocr_text']} | {d['score']:.1f} | OCR: {d['ocr_confidence']:.1f}",
            color_fn=lambda d: DETECTION_CLASS_COLORS["plate"],
            font_size=18,
            label_position="bottom"
        )
    
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
        print(f"Created event record for {stream_name} at {event_timestamp}. Took {elapsed_ms:.2f} ms")

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
        print(f"event_table.update_item took {elapsed_ms:.2f} ms")

    def _convert_floats_to_decimals(self, obj):
            if isinstance(obj, float):
                return Decimal(str(obj))  # str() preserves precision
            elif isinstance(obj, list):
                return [self._convert_floats_to_decimals(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: self._convert_floats_to_decimals(v) for k, v in obj.items()}
            else:
                return obj

    
    def _store_detections(self, 
                          device_id: str, 
                          frame_timestamp: datetime, 
                          event_timestamp: datetime,
                          fragment_number: int,
                          fragment_producer_timestamp: datetime,
                          fragment_server_timestamp: datetime, 
                          detections: list):


        detections_to_store = [det for det in detections if det['label'] in CLASSES_TO_STORE and det['score'] >= 0.5]
        item = {
            "device_id": device_id,
            "frame_timestamp": frame_timestamp.isoformat(),
            "event_timestamp": event_timestamp.isoformat(),
            "fragment_number": fragment_number,
            "fragment_producer_timestamp": fragment_producer_timestamp.isoformat(),
            "fragment_server_timestamp": fragment_server_timestamp.isoformat(),
            "detections": self._convert_floats_to_decimals(detections_to_store),
        }
        self.seen_classes.update([det['label'] for det in detections_to_store])
        try:
            self.event_detections_table.put_item(Item=item)
            # print(f"Stored detection results for {device_id} at {frame_timestamp}")
        except Exception as e:
            print(f"Error storing detections in DynamoDB: {e}")


        
    def _store_plates(self, 
                          device_id: str, 
                          frame_timestamp: datetime, 
                          event_timestamp: datetime,
                          fragment_number: int,
                          fragment_producer_timestamp: datetime,
                          fragment_server_timestamp: datetime, 
                          plates: list):


        item = {
            "device_id": device_id,
            "frame_timestamp": frame_timestamp.isoformat(),
            "event_timestamp": event_timestamp.isoformat(),
            "fragment_number": fragment_number,
            "fragment_producer_timestamp": fragment_producer_timestamp.isoformat(),
            "fragment_server_timestamp": fragment_server_timestamp.isoformat(),
            "plates": self._convert_floats_to_decimals(plates),
        }
        try:
            self.event_plate_recognitions_table.put_item(Item=item)
        except Exception as e:
            print(f"Error storing plates in DynamoDB: {e}")