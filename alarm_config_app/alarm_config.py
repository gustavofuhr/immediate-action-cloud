import json
from decimal import Decimal
from datetime import datetime

import boto3


def get_all_stream_ids() -> list[str]:
    # read from json
    with open("streams.json", "r") as f:
        data = json.load(f)
        return sorted(data["stream_ids"])


class AlarmConfigController:

    def __init__(self):
        self.dynamodb = boto3.resource("dynamodb", region_name="eu-west-1")
        self.table = self.dynamodb.Table("stream_configs")

    def get_stream_alarm_config(self, device_id: str) -> dict[str, object]:
        response = self.table.get_item(Key={"device_id": device_id, "config_type": "alarm_config"})
        return self._to_native(response.get("Item", {}))

    def put_stream_alarm_config(self, stream_id: str, config: dict[str, object]) -> None:
        item = {
            "device_id": stream_id,
            "config_type": "alarm_config",
            "config": self._to_dynamo(config),
            "updated_at": datetime.utcnow().isoformat()
        }
        self.table.put_item(Item=item)
        print(f"✅ Inserted alarm config for stream_id='{stream_id}'")

    def _to_native(self, obj: object) -> object:
        if isinstance(obj, list):
            return [self._to_native(v) for v in obj]
        if isinstance(obj, dict):
            return {k: self._to_native(v) for k, v in obj.items()}
        if isinstance(obj, Decimal):
            return int(obj) if obj % 1 == 0 else float(obj)
        return obj
    
    def _to_dynamo(self, obj: object) -> object:
        if isinstance(obj, dict):
            return {k: self._to_dynamo(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_dynamo(v) for v in obj]
        elif isinstance(obj, float):
            return Decimal(str(obj))  # convert float to Decimal safely
        return obj
