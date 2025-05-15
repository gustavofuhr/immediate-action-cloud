import boto3
from boto3.dynamodb.conditions import Key
from datetime import datetime, timedelta



class EventQuery:
    def __init__(self, region_name="eu-west-1", table_name="events"):
        self.dynamodb = boto3.resource("dynamodb", region_name=region_name)
        self.table = self.dynamodb.Table(table_name)
        self.bucket_name = "motion-event-snapshots"
        self.s3 = boto3.client("s3", region_name=region_name)

    def _fetch_dynamo_items(self, device_id : str, start_date: datetime, end_date : datetime):
        start_key = start_date.isoformat()
        end_key = end_date.isoformat()
        
        all_items = []
        last_key = None
        while True:
            if last_key:
                response = self.table.query(
                    KeyConditionExpression=Key("device_id").eq(device_id) &
                                            Key("event_timestamp").between(start_key, end_key),
                    ExclusiveStartKey=last_key
                )
            else:
                response = self.table.query(
                    KeyConditionExpression=Key("device_id").eq(device_id) &
                                            Key("event_timestamp").between(start_key, end_key)
                )

            all_items.extend(response.get("Items", []))

            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break

        return all_items
    
    def _filter_items_by_classes(self, items : list, target_classes : list[str], condition : str = "OR"):
        filtered_items = []
        for item in items:
            if "seen_classes" in item and item["seen_classes"]:
                seen_classes = set(item["seen_classes"])
                if condition == "OR" and seen_classes.intersection(target_classes):
                    filtered_items.append(item)
                elif condition == "AND" and seen_classes.issubset(target_classes):
                    filtered_items.append(item)

        return filtered_items
    

    def _get_video_presigned_urls(self, items : list, expires_in = 3600):
        urls = []
        for item in items:
            if "video_key" in item:
                url = self.s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self.bucket_name, "Key": item["video_key"]},
                    ExpiresIn=expires_in,
                )
                urls.append(url)
        return urls
    

    def query_events(self, device_id : str, start_date : datetime, end_date : datetime, target_classes : list[str], condition : str = "OR"):
        items = self._fetch_dynamo_items(device_id, start_date, end_date)
        filtered_items = self._filter_items_by_classes(items, target_classes, condition)
        video_urls = self._get_video_presigned_urls(filtered_items)
        return video_urls

            