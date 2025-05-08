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

CLASSES_TO_DETECT = {
    0: 'pedestrian', 
    2: 'car', 
    3: 'motorcycle', 
    5: 'bus', 
    6: 'train', 
    7: 'truck', 
    14: 'bird', 
    16: 'dog'
}


DET_COLORS = {
    0: (255, 0, 0), #'pedestrian', 
    2: (0, 255, 0), #'car', 
    3: (0, 255, 0), #'motorcycle', 
    5: (0, 255, 0), #'bus', 
    6: (0, 255, 0), #'train', 
    7: (0, 255, 0), #'truck', 
    14: (0, 0, 255), #'bird', 
    16: (0, 0, 255)
}



class EventAIProcessor:

    def __init__(self, aws_region, s3_bucket = "motion-event-snapshots"):
        self.kvs_fragment_processor = KvsFragementProcessor()
        
        self.session = boto3.Session(region_name=aws_region) # TODO: do I need aws_region?
        self.kvs_client = self.session.client("kinesisvideo")        
        
        self.detector = SageMakerController(aws_region, "dfine-s-obj2coco-endpoint")

        self.event_clip = EventClip(aws_region, s3_bucket, resize_clip_height=720)

        self.dynamodb = boto3.resource("dynamodb", region_name=aws_region)
        self.event_table = self.dynamodb.Table("event_ai")

    def process_frames(self, stream_name : str, 
                       start_timestamp : datetime, 
                       n_seconds : int = 10, 
                       one_in_frames_ratio : int = 10,
                       try_again = True):
        self.stream_name = stream_name
        self.start_timestamp = start_timestamp
        self.n_seconds = n_seconds
        self.try_again = try_again
        self.event_clip_filepath = f"{stream_name}/{start_timestamp.strftime('%Y-%m-%dT%H:%M:%S')}.mp4"
        
        self.one_in_frames_ratio = one_in_frames_ratio
        self.end_timestamp = start_timestamp + timedelta(seconds=n_seconds)

        get_media_endpoint = self._get_data_endpoint(stream_name, 'GET_MEDIA')
        kvs_media_client = self.session.client('kinesis-video-media', endpoint_url=get_media_endpoint)

        # TODO: if the KVS stream is still not there, it should be a good idea to wait a bit. Is my retry working?
        # TODO: If I read the fragments too fast, will I end up reaching on_stream_read_complete before I get all the frames?
        time.sleep(5)  # Wait for the stream to be available TODO: improve that
        get_media_response = kvs_media_client.get_media(
            StreamName=stream_name,
            StartSelector={
                'StartSelectorType': 'PRODUCER_TIMESTAMP',
                'StartTimestamp': start_timestamp
            }
        )

        self.stream = KvsConsumerLibrary(kvs_media_client, 
                                    get_media_response, 
                                    self.on_fragment_arrived, 
                                    self.on_stream_read_complete, 
                                    self.on_stream_read_exception)
        self.stream.start()

    def on_fragment_arrived(self, stream_name, fragment_bytes, fragment_dom, fragment_receive_duration):
        """ Called when a new KVS fragment arrives """
        frag_tags = self.kvs_fragment_processor.get_fragment_tags(fragment_dom)
        frag_number = frag_tags["AWS_KINESISVIDEO_FRAGMENT_NUMBER"]
        print(f'Fragment arrived: {frag_number}')
        frag_producer_timestamp = datetime.fromtimestamp(float(frag_tags["AWS_KINESISVIDEO_PRODUCER_TIMESTAMP"]), tz=timezone.utc)
        if frag_producer_timestamp < self.start_timestamp:
            print(f'Fragment timestamp {frag_producer_timestamp} is before start timestamp {self.start_timestamp}. Skipping.')
            return
        if frag_producer_timestamp > self.end_timestamp:
            print(f'Fragment timestamp {frag_producer_timestamp} is after end timestamp {self.end_timestamp}. Stopping stream.')
            self.stream.stop_thread()
            return
        
        ndarray_frames = self.kvs_fragment_processor.get_frames_as_ndarray(fragment_bytes, self.one_in_frames_ratio)
        for i, frame in enumerate(ndarray_frames):
            image_pil = Image.fromarray(frame)  
            
            filtered_detections, raw_detections = self.detector.run(image_pil, classes_to_detect=CLASSES_TO_DETECT.keys(), threshold=0.7, verbose=True)
            self._store_detections(self.stream_name, f"{frag_producer_timestamp}_{str(i).zfill(3)}", raw_detections)
            if filtered_detections:
                image_pil = self._draw_detections_on_frame(image_pil, filtered_detections)

            self.event_clip.add_frame(image_pil)
        

    def on_stream_read_complete(self, stream_name):
        print(f'Read Media on stream: {stream_name} Completed')
        if self.event_clip.frames == [] and self.try_again:
            print("Unable to read frames, will try again")
            time.sleep(5)
            self.process_frames(self.stream_name, self.start_timestamp, self.n_seconds, self.one_in_frames_ratio, try_again=False)
            return
        
        print('Writing event clip to S3')
        self.event_clip.send_clip_to_s3(self.event_clip_filepath)

    def on_stream_read_exception(self, stream_name, error):
        print(f'Error on stream: {stream_name} - {error}')

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

            draw.rectangle([x1, y1, x2, y2], outline=DET_COLORS[obj_class], width=2)
            draw.text((x1, y1 - 10), f"{CLASSES_TO_DETECT[obj_class]}: {score:.2f}", fill=DET_COLORS[obj_class], font=font)

        return frame_pil
    
    def _store_detections(self, device_id: str, timestamp: str, detections: list):
        def convert_floats_to_decimals(obj):
            if isinstance(obj, float):
                return Decimal(str(obj))  # str() preserves precision
            elif isinstance(obj, list):
                return [convert_floats_to_decimals(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: convert_floats_to_decimals(v) for k, v in obj.items()}
            else:
                return obj

        item = {
            "device": device_id,
            "timestamp": timestamp,
            "all_fragment_detections": convert_floats_to_decimals(detections),
        }
        try:
            self.event_table.put_item(Item=item)
            print(f"Stored detection results for {device_id} at {timestamp}")
        except Exception as e:
            print(f"Error storing detections in DynamoDB: {e}")