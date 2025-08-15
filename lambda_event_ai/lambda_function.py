import json
from datetime import datetime
import time

from lambda_config import get_ai_config
from event_ai_processor import EventAIProcessor, StreamNotReadyError



def return_ok_response():
    d = {
        'statusCode': 200,
        'body': "Event processed successfully"
    }
    print(f"Response: {d}")
    return d
        
def return_error_response(error_message):
    d = {
        'statusCode': 500,
        'body': json.dumps(f'Error processing stream: {error_message}')
    }
    print(f"Response: {d}")
    return d

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
    print("-------------- Lambda Event AI Processor --------------")
    print(json.dumps(event, indent=4))
    topic = event["topic"]
    stream_name = topic.split("/")[1]
    start_timestamp = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
    event_ai_processor = EventAIProcessor(aws_region="eu-west-1", stream_ai_config=get_ai_config(stream_name))

    lambda_context = {
        "lambda_function_invoked_arn": context.invoked_function_arn,
        "lambda_function_version": context.function_version,
        "lambda_aws_request_id": context.aws_request_id,
        "lambda_log_stream_name": context.log_stream_name,
    }

    max_retries = 6
    base_delay = 1

    for attempt in range(max_retries):
        print(f"\nAttempt #{attempt + 1} to process event '{stream_name}'")
        start_time = time.time()
        event_ai_processor.process_frames(stream_name, start_timestamp, lambda_context, attempt+1, one_in_frames_ratio=7, n_seconds=10)
        event_ai_processor.stream.join(timeout=300)
        elapsed_time = time.time() - start_time
        print(f"Elapsed time for processing event '{stream_name}': {elapsed_time:.2f} seconds")

        if event_ai_processor.stream_exception:
            if isinstance(event_ai_processor.stream_exception, StreamNotReadyError):
                if attempt < max_retries - 1:
                    backoff_time = base_delay * (2 ** attempt)
                    print(f"Stream not ready. Retrying in {backoff_time:.1f}s...")
                    time.sleep(backoff_time)
                else:
                    print(f"Processing failed after {max_retries} attempts.")
                    return return_error_response(str(event_ai_processor.stream_exception))
            else:
                print(f"Unexpected stream error: {event_ai_processor.stream_exception}")
                return return_error_response(f"Unexpected error: {event_ai_processor.stream_exception}")
        else:
            break
    
    return return_ok_response()

if __name__ == "__main__":
    class MockLambdaContext:
        function_version = "$LATEST"
        invoked_function_arn = "arn:aws:lambda:eu-west-1:123456789012:function:mock_lambda"
        aws_request_id = "2afc22dd-4c08-4e5b-8cb5-b002d9be13d5"
        log_stream_name = "2025/05/15/[$LATEST]7baa3150e9d642eb9a8d7d97e920c2be"


    # event = {
    #     "topic": "cameras/B8A44FB3A1F9__front/events/streaming/start",
    #     "streaming_options": {
    #         "gop_length": 26,
    #         "frame": "1280x720",
    #         "fps": 25,
    #         "encoder": "h264"
    #     },
    #     "post_buffer_seconds": 10,
    #     "pre_buffer_seconds": 10,
    #     "timestamp": "2025-05-26T07:51:56.138485Z"
    # }

    # I think we have plats on the bottom event
    # event = {
    #     "topic": "cameras/B8A44FE6D078__booth_entrance/events/streaming/start",
    #     "streaming_options": {
    #         "gop_length": 26,
    #         "frame": "1820x1080",
    #         "fps": 25,
    #         "encoder": "h264"
    #     },
    #     "post_buffer_seconds": 10,
    #     "pre_buffer_seconds": 10,
    #     "timestamp": "2025-07-04T21:22:00.771253Z"
    # }

    # EVENT FROM FRONT, PEOPLE WITH AND WITHOUT PPE
    # event = ("B8A44FB3A1F9__front, "2025-06-18T15:38:14.946325+00:00")
    
    # CAR PLATE EVENT, plate 162D11338
    event = ("B8A44FE6D078__booth_entrance", "2025-07-31T17:52:01.943080+00:00")

    dummy_event = {
        "topic": "cameras/{}/events/streaming/start",
        "streaming_options": {
            "gop_length": 26,
            "frame": "1920x1080",
            "fps": 25,
            "encoder": "h264"
        },
        "post_buffer_seconds": 10,
        "pre_buffer_seconds": 10,
        "timestamp": "{}"
    }
    dummy_event["topic"] = f"cameras/{event[0]}/events/streaming/start"
    dummy_event["timestamp"] = event[1].replace("+00:00", "Z")

    lambda_handler(dummy_event, MockLambdaContext())