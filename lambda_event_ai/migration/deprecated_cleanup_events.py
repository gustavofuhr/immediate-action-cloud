import boto3
from boto3.dynamodb.conditions import Key
from datetime import datetime, timedelta
from tqdm import tqdm

dynamodb = boto3.resource("dynamodb", region_name="eu-west-1")  # change region

# table = dynamodb.create_table(
#     TableName="event_ai_clean",
#     KeySchema=[
#         {"AttributeName": "device", "KeyType": "HASH"},
#         {"AttributeName": "timestamp", "KeyType": "RANGE"},
#     ],
#     AttributeDefinitions=[
#         {"AttributeName": "device", "AttributeType": "S"},
#         {"AttributeName": "timestamp", "AttributeType": "S"},
#     ],
#     BillingMode='PAY_PER_REQUEST'
# )

# table.wait_until_exists()
# print("Table created.")


source_table = dynamodb.Table("event_ai")
print("Approximate count in source table:", source_table.item_count)

target_table = dynamodb.Table("event_ai_clean")

devices = ["axis-p3827-panoramic-tree", "axis-p3827-front-far"]

def adjust_timestamp(ts_str, frame_index):
    base = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    adjusted = base + timedelta(milliseconds=frame_index * 100)
    return adjusted.isoformat()


def query_all_items_for_device(device_name):
    last_key = None
    while True:
        kwargs = {
            "KeyConditionExpression": Key("device").eq(device_name),
        }
        if last_key:
            kwargs["ExclusiveStartKey"] = last_key

        response = source_table.query(**kwargs)
        items = response.get("Items", [])
        for item in items:
            yield item

        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            break

for device_name in devices:
    print(f"Processing device: {device_name}")
    total_migrated = 0

    for item in tqdm(query_all_items_for_device(device_name), desc=f"Migrating {device_name}"):
        original_ts = item["timestamp"]

        if "_" in original_ts:
            base_ts, suffix = original_ts.rsplit("_", 1)
            try:
                frame_index = int(suffix)
            except ValueError:
                frame_index = 0
        else:
            base_ts = original_ts
            frame_index = 0

        clean_ts = adjust_timestamp(base_ts, frame_index)

        item["timestamp"] = clean_ts  # new sort key
        item["ith_frame_in_fragment"] = frame_index

        target_table.put_item(Item=item)
        total_migrated += 1

    print(f"Migrated {total_migrated} items from {device_name}")
