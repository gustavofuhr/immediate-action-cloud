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


    def _filter_items_by_classes(self, items: list, target_classes: list[str], condition: str = "OR"):
        filtered_items = []
        for item in items:
            if "seen_classes" in item and item["seen_classes"]:
                seen_classes = set(item["seen_classes"])
                if condition == "OR" and seen_classes.intersection(target_classes):
                    filtered_items.append(item)
                elif condition == "AND" and seen_classes.issuperset(target_classes):
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
        
        if "animals" in target_classes:
            target_classes.extend(['dog', 'sheep', 'cow', 'bird', 'horse', 'elephant', 'bear', 'zebra', 'giraffe'])
            target_classes.remove('animals')
        if "vehicles" in target_classes:
            target_classes.extend(['car', 'motorcycle', 'bus', 'bicycle', 'train', 'truck', 'boat', 'airplane'])
            target_classes.remove('vehicles')

        items = self._fetch_dynamo_items(device_ids, start_date, end_date)
        filtered_items = self._filter_items_by_detection_stats(items, target_classes, threshold, condition)

        results = []
        for item in filtered_items:
            entry = {
                "timestamp": item.get("event_timestamp"),
                "device_id": item.get("device_id"),
                "seen_classes": item.get("seen_classes", []),
                "detection_stats": item.get("detection_stats", {}),
                "video_url": None
            }
            if "video_key" in item:
                entry["video_url"] = self.s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self.bucket_name, "Key": item["video_key"]},
                    ExpiresIn=3600,
                )
            results.append(entry)

        return results


    @cached(cache=device_id_cache)
    def get_all_device_ids(self) -> list[str]:
        # read from json
        with open("cameras.json", "r") as f:
            data = json.load(f)
            return sorted(data["cameras"])
        # device_ids = set()
        # last_key = None

        # while True:
        #     kwargs = {
        #         "ProjectionExpression": "device_id"
        #     }
        #     if last_key:
        #         kwargs["ExclusiveStartKey"] = last_key

        #     response = self.table.scan(**kwargs)
        #     for item in response.get("Items", []):
        #         if "device_id" in item:
        #             device_ids.add(item["device_id"])

        #     last_key = response.get("LastEvaluatedKey")
        #     if not last_key:
        #         break
        # print( sorted(device_ids))
        # return sorted(device_ids)
