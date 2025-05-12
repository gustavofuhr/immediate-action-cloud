import boto3
from boto3.dynamodb.conditions import Key
from datetime import datetime, timedelta
from pprint import pprint




COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat',
    9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe',
    24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
    31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
    37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
    52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

class EventQuery:

    def __init__(self, region_name="eu-west-1", table_name="event_ai"):
        dynamodb = boto3.resource("dynamodb", region_name=region_name)

        self.table = dynamodb.Table(table_name)
        self.bucket_name = "motion-event-snapshots"
        self.s3 = boto3.client("s3", region_name=region_name)

    def fetch_dynamo_items(self, device_name, start_date, end_date):
        start_key = start_date.isoformat() + " 00:00:00+00:00"
        end_key = end_date.isoformat() + " 23:59:59+00:00"


        all_items = []
        last_key = None
        print("Fetching items from DynamoDB...")
        print(f"Device: {device_name}", f"Start: {start_key}", f"End: {end_key}")
        while True:
            if last_key:
                print(f"Last evaluated key: {last_key}")
                response = self.table.query(
                    KeyConditionExpression=Key("device").eq(device_name) &
                                            Key("timestamp").between(start_key, end_key),
                    ExclusiveStartKey=last_key
                )
            else:
                response = self.table.query(
                    KeyConditionExpression=Key("device").eq(device_name) &
                                            Key("timestamp").between(start_key, end_key)
                )
            all_items.extend(response.get("Items", []))

            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break

        print(f"Total items: {len(all_items)}")
    
    def filter_items_by_classes(self, items, obj_classes):
        print(f"Filtering items matching classes {obj_classes}...")
        match_items = []
        for item in items:
            classes = set([COCO_CLASSES[int(d["label"])] for d in item["all_fragment_detections"] if d["score"] >= 0.7])
            if classes.intersection(obj_classes):
                timestamp = item["timestamp"][:-4] # remove the frame number
                if timestamp not in match_items:
                    match_items.append(timestamp)
                
        print(f"Found {len(match_items)} matches")

    def get_video_urls(self, device_name, matched_timestamps):
        

        video_files = []
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=device_name):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".mp4"):
                    ts_str = key.removeprefix(f"{device_name}/").removesuffix(".mp4")
                    ts = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
                    video_files.append((ts, key))

        video_files.sort()

        # 2. Map match_items to video file
        match_to_video = {}
        matched_videos = set()

        for ts_str in matched_timestamps:
            match_ts = datetime.fromisoformat(ts_str.split("+")[0])  # strip offset if any
            lower_bound = match_ts - timedelta(seconds=11)

            # Find best match: latest video before or at match_ts
            best_match = None
            for vid_ts, key in reversed(video_files):
                if lower_bound <= vid_ts <= match_ts:
                    best_match = key
                    break

            match_to_video[ts_str] = best_match
            print(f"Match: {ts_str} -> Video: {best_match}")
            if best_match:
                matched_videos.add(best_match)
        
        video_s3_urls = []
        for m in matched_videos:
            url = self.s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": m},
                ExpiresIn=3600,
            )
            video_s3_urls.append(url)

        print(f"Found {len(video_s3_urls)} video URLs")
        return video_s3_urls


    def get_filtered_videos(self, start_time, end_time, obj_classes, device_name):
        # first get all items from DynamoDB
        all_items = self.fetch_dynamo_items(device_name, start_time, end_time)
        filtered_items_timestamps = self.filter_items_by_classes(all_items, obj_classes)
        # then get the video URLs from S3
        return self.get_video_urls(device_name, filtered_items_timestamps)        

    

    