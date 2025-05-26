import os
import sys
sys.path.append("../lambda_event_ai") 
import boto3
from boto3.dynamodb.conditions import Key, Attr
from datetime import datetime, timedelta
from colorama import Fore, Style, init
from concurrent.futures import ThreadPoolExecutor, as_completed

from lambda_function import lambda_handler
import json

init(autoreset=True)  # So colors auto-reset after each print

class IncompleteEventHandler:
    def __init__(self, table_name="events", log_group_name="/aws/lambda/event_ai_debug_sagemaker", region="eu-west-1"):
        self.dynamodb = boto3.resource("dynamodb", region_name=region)
        self.logs_client = boto3.client("logs", region_name=region)
        self.table = self.dynamodb.Table(table_name)
        self.log_group_name = log_group_name

    def fetch_incomplete_events(self, device_id: str, hours_back=24):
        start_time = (datetime.utcnow() - timedelta(hours=hours_back)).isoformat()
        end_time = datetime.utcnow().isoformat()

        all_items = []
        last_key = None

        while True:
            query_kwargs = {
                "KeyConditionExpression": Key("device_id").eq(device_id) &
                                        Key("event_timestamp").between(start_time, end_time),
                "FilterExpression": Attr("processing_end_timestamp").not_exists()
            }
            if last_key:
                query_kwargs["ExclusiveStartKey"] = last_key

            response = self.table.query(**query_kwargs)
            all_items.extend(response.get("Items", []))
            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break

        print(Fore.LIGHTRED_EX + f"Fetched {len(all_items)} incomplete events for {device_id}.")
        return all_items
    
    def get_event_logs(self, item: dict) -> str:
        log_stream_name = item.get("lambda_log_stream_name")
        request_id = item.get("lambda_aws_request_id")

        if not log_stream_name:
            return "No log stream name"

        if not request_id:
            return "No request ID"

        try:
            response = self.logs_client.get_log_events(
                logGroupName=self.log_group_name,
                logStreamName=log_stream_name,
                limit=10000,
                startFromHead=True
            )
            messages = [event["message"] for event in response["events"]]

            start_prefix = f"START RequestId: {request_id}"
            end_prefix = f"END RequestId: {request_id}"

            start_index = next((i for i, msg in enumerate(messages) if start_prefix in msg), None)
            end_index = next((i for i, msg in enumerate(messages) if end_prefix in msg), None)

            if start_index is not None and end_index is not None and end_index >= start_index:
                trimmed_logs = "".join(messages[start_index:end_index + 1])
            elif start_index is not None:
                trimmed_logs = "".join(messages[start_index:])  # END not found, return till end
            else:
                trimmed_logs = "START line not found in log stream.\n" + "".join(messages)

            return trimmed_logs

        except self.logs_client.exceptions.ResourceNotFoundException:
            return "Log stream not found"
        except Exception as e:
            return f"Error: {e}"

    def retry_events(self, items):
        class MockLambdaContext:
            function_version = "$LATEST"
            invoked_function_arn = "failed_event_lambda"
            aws_request_id = "LOCAL_MOCK_REQUEST_ID"
            log_stream_name = ""

        print(Fore.LIGHTRED_EX + f"Preparing to retry {len(items)} events...")
        for item in items:
            print(Fore.YELLOW + f"Retrying event for device {item['device_id']} with timestamp {item['event_timestamp']}")
            event = {
                "timestamp": item["event_timestamp"],
                "topic": f"cameras/{item['device_id']}/events/streaming/start",
            }
            ret_lambda = lambda_handler(event, MockLambdaContext())

            if ret_lambda.get("statusCode") != 200:
                print(Fore.RED + f"Error processing event: {ret_lambda}")
                os.makedirs("failed_retries", exist_ok=True)
                file_name = f"failed_retries/{item['device_id']}_{item['event_timestamp'].replace(':', '-')}.json"
                with open(file_name, "w") as f:
                    json.dump(item, f, indent=4)
            else:
                print(Fore.GREEN + "Event processed successfully.")

if __name__ == "__main__":

    device_ids = [
        "B8A44F976508__panoramic_tree",
        "B8A44F9C9902__train_rails",
        "B8A44FB97C3C__panoramic_garages",
        "B8A44FCDF536__parking_lot",
        "B8A44FD014E5__distorted_tree",
        "B8A44FE820CC__construction_ptz",
        "B8A44F9CA507__front_far",
        "B8A44FB0AC74__trains_close",
        "B8A44FB3A1F9__front",
        "B8A44FB981BB__panoramic_trains"
    ]

    def process_device(device_id):
        try:
            handler = IncompleteEventHandler()
            incomplete_events = handler.fetch_incomplete_events(device_id)
            handler.retry_events(incomplete_events)
            return f"{device_id}: done"
        except Exception as e:
            return f"{device_id}: failed with {e}"

    if __name__ == "__main__":
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_device, device_id) for device_id in device_ids]

            for future in as_completed(futures):
                print(Fore.CYAN + future.result())
