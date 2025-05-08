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
    
    event_ai_processor.process_frames(stream_name, start_timestamp, one_in_frames_ratio = 5, n_seconds = 10)
    event_ai_processor.stream.join(timeout=300)  # Wait for up to 5 minutes

    return {
        'statusCode': 200,
        'body': json.dumps(f'Finished AI processing')
    }

if __name__ == "__main__":
    event = {
        "topic": "cameras/axis-local/events/motion/start",
        "timestamp": "2025-04-19T00:20:21.303729Z",
        "profile": "Camera1ProfileANY",
        "active": True
    }
    
    lambda_handler(event, None)