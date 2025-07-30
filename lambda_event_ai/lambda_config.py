import time
from decimal import Decimal

import boto3

dynamodb = boto3.resource("dynamodb")
configs_table = dynamodb.Table("stream_configs")


DEFAULT_AI_CONFIG = {
    "debug": False,
    "models": []
}

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
    start = time.time()
    try:
        response = configs_table.get_item(Key={"device_id": device_id, "config_type": "ai_config"})
        config = response.get("Item", {}).get("config", DEFAULT_AI_CONFIG)
    except Exception as e:
        print(f"[WARNING] Failed to get ai_config for {device_id}: {e}")
        config = DEFAULT_AI_CONFIG

    config = convert_decimals(config)  
    elapsed = (time.time() - start) * 1000
    print(f"[INFO] Fetched ai_config for '{device_id}' in {elapsed:.2f} ms")
    return config