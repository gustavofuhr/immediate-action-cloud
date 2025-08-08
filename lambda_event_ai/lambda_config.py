import time
from decimal import Decimal
from functools import wraps
import json

import boto3

dynamodb = boto3.resource("dynamodb", region_name="eu-west-1")
configs_table = dynamodb.Table("stream_configs")


DEFAULT_AI_CONFIG = {
    "debug_mode": "none",
    "models": ["object_detection_then_ppe"],
    "per_model_params": {
        "object_detection_then_ppe": {
            "threshold": 0.5,
            "classes_to_detect": ['person', 'car_plate','bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
        }
    }
}

def ttl_cache(seconds=30):
    def decorator(func):
        cache = {}
        @wraps(func)
        def wrapper(*args):
            now = time.time()
            if args in cache:
                result, ts = cache[args]
                if now - ts < seconds:
                    return result
            result = func(*args)
            cache[args] = (result, now)
            return result
        return wrapper
    return decorator

def convert_decimals(obj):
    """
    Recursively converts Decimal objects to float in dicts/lists.
    """
    if isinstance(obj, list):
        return [convert_decimals(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, Decimal):
        return float(obj)
    else:
        return obj


def get_ai_config(device_id):
    try:
        response = configs_table.get_item(Key={"device_id": device_id, "config_type": "ai_config"})
        config = response.get("Item", {}).get("config", DEFAULT_AI_CONFIG)
    except Exception as e:
        config = DEFAULT_AI_CONFIG

    config = convert_decimals(config)  
    print(json.dumps(config, indent=4))
    return config

@ttl_cache(seconds=30)
def get_alarm_config(device_id):
    try:
        response = configs_table.get_item(Key={"device_id": device_id, "config_type": "alarm_config"})
        config = response.get("Item", {}).get("config", {})
    except Exception as e:
        return None

    config = convert_decimals(config)  
    return config