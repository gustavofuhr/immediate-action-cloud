import time
from datetime import timedelta, datetime, timezone
from decimal import Decimal

import boto3
from PIL import Image, ImageDraw, ImageFont

from amazon_kinesis_video_consumer_library.kinesis_video_streams_parser import KvsConsumerLibrary
from amazon_kinesis_video_consumer_library.kinesis_video_fragment_processor import KvsFragementProcessor
# from dfine_controller_ort import DFINEControllerORT
from sagemaker_controller import SageMakerController
from event_clip import EventClip


DETECTION_CLASS_COLORS = {
    'person': (255, 0, 0),  
    'car': (0, 255, 0),  
    'motorcycle': (0, 255, 0),  
    'bus': (0, 255, 0),  
    'train': (0, 255, 0),  
    'truck': (0, 255, 0),  
    'bird': (0, 0, 255),  
    'dog': (0, 0, 255)  
}


class EventAIProcessor:

    def __init__(self, aws_region, s3_bucket = "motion-event-snapshots"):
        self.kvs_fragment_processor = KvsFragementProcessor()
        
        self.session = boto3.Session(region_name=aws_region) # TODO: do I need aws_region?
        self.kvs_client = self.session.client("kinesisvideo")        
        
        self.detector = SageMakerController(aws_region, "dfine-x-obj2coco-endpoint")

        self.event_clip = EventClip(aws_region, s3_bucket, resize_clip_height=720)

        self.dynamodb = boto3.resource("dynamodb", region_name=aws_region)
        self.event_table = self.dynamodb.Table("events")
        self.event_detections_table = self.dynamodb.Table("event_detections")
        self.last_fragment_timestamp = None

    def process_frames(self, stream_name : str, 
                       event_timestamp : datetime,
                       lambda_context : dict, 
                       n_seconds : int = 10, 
                       one_in_frames_ratio : int = 10,
                       try_again = True):
        self.stream_name = stream_name
        self.event_timestamp = event_timestamp
        self.n_seconds = n_seconds
        self.try_again = try_again
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
            self.lambda_context
        )
        

        # TODO: if the KVS stream is still not there, it should be a good idea to wait a bit. Is my retry working?
        # TODO: If I read the fragments too fast, will I end up reaching on_stream_read_complete before I get all the frames?
        time.sleep(5)  # Wait for the stream to be available TODO: improve that
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
        self.detection_stats = {}
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
            
            filtered_detections, raw_detections = self.detector.run(image_pil, classes_to_detect=DETECTION_CLASS_COLORS.keys(), threshold=0.5, verbose=(self.n_frames == 0))
            self._store_detections(self.stream_name, frame_timestamp, self.event_timestamp, frag_number, frag_producer_timestamp, frag_server_timestamp, raw_detections)            

            for det in filtered_detections:
                cls = det['label']
                score = det['score']
                if cls not in self.detection_stats:
                    self.detection_stats[cls] = {
                        "total_confidence": 0.0,
                        "max_confidence": 0.0,
                        "n_frames": 0
                    }
                self.detection_stats[cls]["total_confidence"] += score
                self.detection_stats[cls]["max_confidence"] = max(self.detection_stats[cls]["max_confidence"], score)
                self.detection_stats[cls]["n_frames"] += 1


            if filtered_detections:
                image_pil = self._draw_detections_on_frame(image_pil, filtered_detections)
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



    def on_stream_read_complete(self, stream_name):
        print(f'Read Media on stream: {stream_name} Completed')
        if self.event_clip.frames == [] and self.try_again:
            print("Unable to read frames, will try again")
            time.sleep(30) # Wait for the stream to be available TODO: improve that
            self.process_frames(self.stream_name, self.stream_start_timestamp, self.lambda_context, self.n_seconds, self.one_in_frames_ratio, try_again=False)
        
        else:
            print('Writing event clip to S3')
            self.event_clip.send_clip_to_s3(self.event_clip_filepath)

            # Finalize detection_stats (compute average)
            final_detection_stats = {}
            for cls, stats in self.detection_stats.items():
                final_detection_stats[cls] = {
                    "avg_confidence": round(stats["total_confidence"] / stats["n_frames"], 4),
                    "max_confidence": round(stats["max_confidence"], 4),
                    "n_frames": stats["n_frames"]
                }

            print("Event summary:")
            print(f"  Stream name: {self.stream_name}")
            print(f"  Event timestamp: {self.event_timestamp.isoformat()}")
            print(f"  Stream start timestamp: {self.stream_start_timestamp.isoformat()}")
            print(f"  Stream end timestamp: {self.stream_end_timestamp.isoformat()}")
            print(f"  Number of processed fragments: {self.n_fragments}")
            print(f"  Number of processed frames: {self.n_frames}")
            print(f"  Video S3 key: {self.event_clip_filepath}")
            print(f"  Seen classes: {list(self.seen_classes)}")
            print(f"  Detection stats:")
            for cls, stats in final_detection_stats.items():
                print(f"    - {cls}: avg_conf={stats['avg_confidence']}, max_conf={stats['max_confidence']}, n_frames={stats['n_frames']}")

            self._update_event_record(
                stream_name=self.stream_name,
                event_timestamp=self.event_timestamp,
                stream_start_timestamp=self.stream_start_timestamp,
                stream_end_timestamp=self.stream_end_timestamp,
                n_processed_fragments=self.n_fragments,
                n_processed_frames=self.n_frames,
                video_key=self.event_clip_filepath,
                seen_classes=list(self.seen_classes),
                detection_stats=final_detection_stats
            )

    def on_stream_read_exception(self, stream_name, error):
        print(f'Error on stream: {stream_name} - {error}')
        raise error

    def _get_data_endpoint(self, stream_name, api_name):
        """ Fetch KVS data endpoint """
        response = self.kvs_client.get_data_endpoint(StreamName=stream_name, APIName=api_name)
        return response['DataEndpoint']
    
    
    def _draw_detections_on_frame(self, frame_pil, detections):
        draw = ImageDraw.Draw(frame_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()

        for det in detections:
            x1 = int(det['bbox']['top_left']['x'])
            y1 = int(det['bbox']['top_left']['y'])
            x2 = int(det['bbox']['bottom_right']['x'])
            y2 = int(det['bbox']['bottom_right']['y'])
            score = det['score']
            obj_class = det['label']

            draw.rectangle([x1, y1, x2, y2], outline=DETECTION_CLASS_COLORS[obj_class], width=2)
            draw.text((x1, y1 - 10), f"{obj_class}: {score:.2f}", fill=DETECTION_CLASS_COLORS[obj_class], font=font)

        return frame_pil
    
    def _create_event_record(self, stream_name, event_timestamp, lambda_context):
        """ Create a record in the events table """
        self.event_table.put_item(
            Item={
                "device_id": stream_name,
                "event_timestamp": event_timestamp.isoformat(),
                "processing_start_timestamp": datetime.now(timezone.utc).isoformat(),
                **lambda_context,
            }
        )

        # this record should be updated at the end of the processing
        print(f"Created event record for {stream_name} at {event_timestamp}")

    def _update_event_record(self, 
                             stream_name, 
                             event_timestamp, 
                             stream_start_timestamp, 
                             stream_end_timestamp, 
                             n_processed_fragments, 
                             n_processed_frames, 
                             video_key, 
                             seen_classes,
                             detection_stats):
        """ Update the event record in the events table """
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
                            "detection_stats = :detection_stats",
            ExpressionAttributeValues={
                ":processing_end_timestamp": datetime.now(timezone.utc).isoformat(),
                ":stream_start_timestamp": stream_start_timestamp.isoformat(),
                ":stream_end_timestamp": stream_end_timestamp.isoformat(),
                ":n_processed_fragments": n_processed_fragments,
                ":n_processed_frames": n_processed_frames,
                ":video_key": video_key,
                ":seen_classes": seen_classes,
                ":detection_stats": self._convert_floats_to_decimals(detection_stats),
            }
        )

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


        CLASSES_TO_STORE = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
    
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