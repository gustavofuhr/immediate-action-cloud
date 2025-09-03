# lambda_event_ai/cooldown.py
import os, time, boto3

_TABLE = os.getenv("ALARM_RULE_STATE_TABLE", "alarm_rule_state")
_dynamodb = boto3.resource("dynamodb", region_name="eu-west-1")
_table = _dynamodb.Table(_TABLE)

class AlarmCooldown:
    @staticmethod
    def try_acquire(stream_name: str, rule_id: str, cooldown_seconds: int, now: int | None = None) -> bool:
        if cooldown_seconds <= 0:
            return True
        now = int(now or time.time())
        next_allowed = now + int(cooldown_seconds)
        # Keep rows from living forever; at least one cooldown period, min 1h
        ttl = now + max(int(cooldown_seconds), 3600)
        try:
            _table.update_item(
                Key={"stream_name": stream_name, "rule_id": rule_id},
                UpdateExpression="SET last_triggered_at=:now, next_allowed_at=:next, #ttl=:ttl",
                ConditionExpression="attribute_not_exists(next_allowed_at) OR :now >= next_allowed_at",
                ExpressionAttributeValues={":now": now, ":next": next_allowed, ":ttl": ttl},
                ExpressionAttributeNames={"#ttl": "ttl"},
            )
            return True
        except _table.meta.client.exceptions.ConditionalCheckFailedException:
            return False
