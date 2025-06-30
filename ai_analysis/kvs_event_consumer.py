import sys
import boto3
from datetime import datetime, timedelta, timezone

sys.path.append('../lambda_event_ai')
from amazon_kinesis_video_consumer_library.kinesis_video_streams_parser import KvsConsumerLibrary
from amazon_kinesis_video_consumer_library.kinesis_video_fragment_processor import KvsFragementProcessor


class KVSEventConsumer:
    def __init__(self):
        self.kvs_fragment_processor = KvsFragementProcessor()
        self.aws_region = "eu-west-1"
        self.process_n_seconds = 10
        self.one_in_frames_ratio = 25
        
        self.session = boto3.Session(region_name=self.aws_region) 
        self.kvs_client = self.session.client("kinesisvideo")        
        
    def _get_data_endpoint(self, stream_name, api_name):
        """ Fetch KVS data endpoint """
        response = self.kvs_client.get_data_endpoint(StreamName=stream_name, APIName=api_name)
        return response['DataEndpoint']

    def get_event_frames(self, event):
        self.stream_start_timestamp = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
        get_media_endpoint = self._get_data_endpoint(event['device_id'], 'GET_MEDIA')
        kvs_media_client = self.session.client('kinesis-video-media', endpoint_url=get_media_endpoint)

        get_media_response = kvs_media_client.get_media(
            StreamName=event['device_id'],
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
        self.stream_end_timestamp = self.stream_start_timestamp + timedelta(seconds=self.process_n_seconds)

        self.frames = []
        self.stream.start()
        self.stream.join()  # Wait for the stream to finish processing
        return self.frames

        
    def on_fragment_arrived(self, stream_name, fragment_bytes, fragment_dom, fragment_receive_duration):
        frag_tags = self.kvs_fragment_processor.get_fragment_tags(fragment_dom)
        frag_number = frag_tags["AWS_KINESISVIDEO_FRAGMENT_NUMBER"]
        frag_producer_timestamp = datetime.fromtimestamp(float(frag_tags["AWS_KINESISVIDEO_PRODUCER_TIMESTAMP"]), tz=timezone.utc)        

        if frag_producer_timestamp < self.stream_start_timestamp:
            return
        
        if frag_producer_timestamp > self.stream_end_timestamp:
            self.stream.stop_thread()
            return
        
        ndarray_frames, _ = self.kvs_fragment_processor.get_frames_as_ndarray(fragment_bytes, self.one_in_frames_ratio)
        for i, frame in enumerate(ndarray_frames):
            self.frames.append(frame)
            

    def on_stream_read_complete(self, stream_name):
        pass
        # print(f"Stream {stream_name} read complete.")

    def on_stream_read_exception(self, stream_name, exception):
        print(f"Stream {stream_name} encountered an error: {exception}")


