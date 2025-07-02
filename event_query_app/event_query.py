import boto3
from boto3.dynamodb.conditions import Key, Attr
from datetime import datetime
from cachetools import TTLCache, cached
import json

# Global cache: max 1 item (we're only caching this one call), TTL = 3600 seconds (1 hour)
device_id_cache = TTLCache(maxsize=1, ttl=3600)

class EventQuery:
    def __init__(self, region_name="eu-west-1", table_name="events"):
        self.dynamodb = boto3.resource("dynamodb", region_name=region_name)
        self.table = self.dynamodb.Table(table_name)
        self.bucket_name = "motion-event-snapshots"
        self.s3 = boto3.client("s3", region_name=region_name)

    def _fetch_dynamo_items(self, device_ids: list[str], start_date: datetime, end_date: datetime):
        if not device_ids:
            raise ValueError("device_ids list cannot be empty.")

        start_key = start_date.isoformat()
        end_key = end_date.isoformat()
        all_items = []

        for device_id in device_ids:
            last_key = None
            while True:
                kwargs = {
                    "KeyConditionExpression": Key("device_id").eq(device_id) &
                                              Key("event_timestamp").between(start_key, end_key)
                }
                if last_key:
                    kwargs["ExclusiveStartKey"] = last_key

                response = self.table.query(**kwargs)
                all_items.extend(response.get("Items", []))
                last_key = response.get("LastEvaluatedKey")
                if not last_key:
                    break

        return all_items

    def _filter_items_by_detection_stats(
        self,
        items: list,
        target_classes: list[str],
        threshold: float = 0.5,
        condition: str = "OR"
    ):
        filtered_items = []

        for item in items:
            detection_stats = item.get("detection_stats", {})
            passed_classes = [
                cls for cls in target_classes
                if cls in detection_stats and detection_stats[cls].get("max_confidence", 0) >= threshold
            ]

            if condition == "OR" and passed_classes:
                filtered_items.append(item)
            elif condition == "AND" and all(cls in passed_classes for cls in target_classes):
                filtered_items.append(item)

        return filtered_items
    
    def _filter_items_by_plate_text(self, items: list, search_plate: str, plate_threshold: float = 0.0, ocr_threshold: float = 0.0):
        filtered_items = []
        for item in items:
            plate_stats = item.get("plate_recognition_stats", {})
            passed_plates = [
                plate for plate in plate_stats
                if plate_stats[plate].get("max_confidence", 0) >= plate_threshold
                and search_plate.lower() in plate.lower()
            ]
            if passed_plates:
                filtered_items.append(item)

        return filtered_items

    def _get_video_presigned_urls(self, items: list, expires_in=3600):
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
    
    def _item_to_dict(self, item: dict) -> dict:
        """
        Convert a DynamoDB item to a dictionary, handling nested structures.
        """
        d = {
            "device_id": item.get("device_id"),
            "event_timestamp": item.get("event_timestamp"),
            "seen_classes": item.get("seen_classes", []),
            "detection_stats": item.get("detection_stats", {}),
            "seen_plates": item.get("seen_plates", []),
            "plate_recognition_stats": item.get("plate_recognition_stats", {}),
            "video_key": item.get("video_key"),
        }
        if "video_key" in item:
            d["video_url"] = self.s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": item["video_key"]},
                ExpiresIn=3600,
            )
        return d

    def query_events(
        self,
        start_date: datetime,
        end_date: datetime,
        target_classes: list[str],
        threshold: float = 0.5,
        condition: str = "OR",
        device_ids: list[str] = [],
    ) -> list[dict]:
        if not device_ids:
            raise ValueError("device_ids must be a non-empty list.")
        
        items = self._fetch_dynamo_items(device_ids, start_date, end_date)
        
        if "animals" in target_classes:
            target_classes.extend(['dog', 'sheep', 'cow', 'bird', 'horse', 'elephant', 'bear', 'zebra', 'giraffe'])
            target_classes.remove('animals')
        if "vehicles" in target_classes:
            target_classes.extend(['car', 'motorcycle', 'bus', 'bicycle', 'train', 'truck', 'boat', 'airplane'])
            target_classes.remove('vehicles')

        if not target_classes:
            filtered_items = items
        else:
            filtered_items = self._filter_items_by_detection_stats(items, target_classes, threshold, condition)

        results = [self._item_to_dict(item) for item in filtered_items]
        return results
    
    def query_events_by_plate(
        self,
        start_date: datetime,
        end_date: datetime,
        search_plate: str,
        plate_threshold: float = 0.0,
        ocr_threshold: float = 0.0,
        device_ids: list[str] = [],
    ) -> list[dict]:
        if not device_ids:
            raise ValueError("device_ids must be a non-empty list.")

        items = self._fetch_dynamo_items(device_ids, start_date, end_date)
        filtered_items = self._filter_items_by_plate_text(items, search_plate, plate_threshold, ocr_threshold)

        results = [self._item_to_dict(item) for item in filtered_items]
        return results

    def get_event_by_id_and_timestamp(self, device_id: str, event_timestamp: str) -> dict | None:
        response = self.table.get_item(
            Key={
                "device_id": device_id,
                "event_timestamp": event_timestamp
            }
        )
        item = response.get("Item")
        return self._item_to_dict(item) if item else None


    @cached(cache=device_id_cache)
    def get_all_device_ids(self) -> list[str]:
        # read from json
        with open("cameras.json", "r") as f:
            data = json.load(f)
            return sorted(data["cameras"])
        
    
