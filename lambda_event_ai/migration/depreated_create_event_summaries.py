

import boto3
from datetime import datetime, timedelta

from decimal import Decimal
from boto3.dynamodb.conditions import Key
from tqdm import tqdm

from event_query import COCO_CLASSES

devices = ["axis-p3827-front-far", "axis-p3827-panoramic-tree"]

for DEVICE_NAME in devices:

    bucket_name = "motion-event-snapshots"
    s3 = boto3.client("s3", region_name="eu-west-1")

    def get_video_urls():
        video_files = []
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=DEVICE_NAME):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".mp4"):
                    ts_str = key.removeprefix(f"{DEVICE_NAME}/").removesuffix(".mp4")
                    ts = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
                    video_files.append((ts, key))

        video_files.sort()
        return video_files

    print("Fetching video URLs...")
    video_files = get_video_urls()
    print(f"Found {len(video_files)} video files.")
    print(f"First video file: {video_files[0][0]} - {video_files[0][1]}")




    dynamodb = boto3.resource("dynamodb", region_name="eu-west-1")
    table = dynamodb.Table("event_ai_clean")
    summary_table = dynamodb.Table("event_summaries")

    for ts, key in tqdm(video_files):
        start_ts = ts.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
        end_ts = (ts + timedelta(seconds=10)).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")

        # print(f"Searching for events from {start_ts} to {end_ts}...")

        all_items = []
        last_key = None

        while True:
            query_kwargs = {
                "KeyConditionExpression": Key("device").eq(DEVICE_NAME) &
                                        Key("timestamp").between(start_ts, end_ts)
            }

            if last_key:
                query_kwargs["ExclusiveStartKey"] = last_key

            response = table.query(**query_kwargs)
            items = response.get("Items", [])
            all_items.extend(items)
            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break

        # Extract unique class labels >= 0.7
        classes_above_thresh = set()

        for item in all_items:
            detections = item.get("all_fragment_detections", [])
            for det in detections:
                if float(det["score"]) >= 0.7:
                    classes_above_thresh.add(int(det["label"]))

        # Build summary record
        summary_item = {
            "timestamp": ts.isoformat(),
            "video_key": f"{key}",
            "duration": 10,
            "device": DEVICE_NAME,
            "classes": sorted({COCO_CLASSES[c] for c in classes_above_thresh if c in COCO_CLASSES}),
            "n_processed_frames": len(all_items),
        }

        # print(f"[SUMMARY] {summary_item}")
        summary_table.put_item(Item=summary_item)
        
