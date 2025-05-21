import json
from datetime import datetime
import sys
sys.path.append("package/")

from event_ai_processor import EventAIProcessor

        
def lambda_handler(event, context):
    """
    The event is expected to be a MQTT message from the topic:
        cameras/<stream_name>/events/motion/start

    A JSON like this:
    {
        "topic": "cameras/axis-local/events/motion/start",
        "timestamp": "2025-04-15T19:29:24.012364Z",
        "profile": "Camera1Profile1",
        "active": True
    }
    """        
    print(json.dumps(event, indent=4))
    event_ai_processor = EventAIProcessor(aws_region="eu-west-1")
    topic = event["topic"]
    stream_name = topic.split("/")[1]
    start_timestamp = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))

    lambda_context = {
        "lambda_function_invoked_arn": context.invoked_function_arn,
        "lambda_function_version": context.function_version,
        "lambda_aws_request_id": context.aws_request_id,
        "lambda_log_stream_name": context.log_stream_name,
    }
    
    event_ai_processor.process_frames(stream_name, start_timestamp, lambda_context, one_in_frames_ratio = 7, n_seconds = 10)
    event_ai_processor.stream.join(timeout=300)  # Wait for up to 5 minutes

    return {
        'statusCode': 200,
        'body': json.dumps(f'Finished AI processing')
    }

if __name__ == "__main__":
    class MockLambdaContext:
        function_version = "$LATEST"
        invoked_function_arn = "arn:aws:lambda:eu-west-1:123456789012:function:mock_lambda"
        aws_request_id = "2afc22dd-4c08-4e5b-8cb5-b002d9be13d5"
        log_stream_name = "2025/05/15/[$LATEST]7baa3150e9d642eb9a8d7d97e920c2be"

    # event = {
    #     "topic": "cameras/axis-local/events/streaming/start",
    #     "timestamp": "2025-05-12T20:31:55.115170Z"
    # }

    event = {
        "topic": "cameras/B8A44F976508__panoramic_tree/events/streaming/start",
        "streaming_options": {
            "gop_length": 26,
            "frame": "1280x720",
            "fps": 25,
            "encoder": "h264"
        },
        "post_buffer_seconds": 10,
        "pre_buffer_seconds": 10,
        "timestamp": "2025-05-20T04:53:52.899667Z"
    }
    
    lambda_handler(event, MockLambdaContext())