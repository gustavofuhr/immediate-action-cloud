import time
from datetime import datetime, time as dtime
from threading import Thread
from decimal import Decimal, ROUND_HALF_UP
import math
import json
import traceback
from zoneinfo import ZoneInfo

import boto3
from PIL import Image
from json_logic import jsonLogic

from lambda_config import get_alarm_config
from alarm_notification_controller import AlarmNotificationController
from lambda_logging import base_logger
from alarm_cooldown import AlarmCooldown



def _run_on_background(fn, *args, **kwargs):
    def _wrapped():
        try:
            fn(*args, **kwargs)
        except Exception as e:
            print("[ERROR] Background task failed:", e)
            traceback.print_exc()
    t = Thread(target=_wrapped, daemon=True)
    t.start()
    return t

def convert_floats(obj):
    if isinstance(obj, list):
        return [convert_floats(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_floats(v) for k, v in obj.items()}
    elif isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        d = Decimal(str(obj))
        return d.quantize(Decimal("1.000000"), rounding=ROUND_HALF_UP)
    else:
        return obj


class AlarmController:

    def __init__(self, logger=None):
        self.logger = logger or base_logger
        
        self.dynamodb = boto3.resource("dynamodb", region_name="eu-west-1")
        self.alarms_table = self.dynamodb.Table("event_alarms")  # PK=device_id, SK=frame_timestamp
        self.notifier = AlarmNotificationController(logger=self.logger)

    def _normalize_plate_text(self, plate_text: str) -> str:
        return plate_text.upper().replace("-", "")

    def _rule_in_schedule(self, rule: dict, frame_timestamp: datetime) -> bool:
        sched = rule.get("schedule")
        if not sched:
            return True  # no schedule → always active

        DOW = ["mon","tue","wed","thu","fri","sat","sun"]
        tz = ZoneInfo(sched.get("timezone", "UTC"))
        local_dt = frame_timestamp.astimezone(tz)
        local_day = DOW[local_dt.weekday()]
        local_t = local_dt.time().replace(second=0, microsecond=0)

        for win in sched.get("include", []):
            if local_day not in win.get("days", DOW):
                continue

            h1, m1 = map(int, win["start"].split(":"))
            h2, m2 = map(int, win["end"].split(":"))
            start_t = dtime(h1, m1)
            end_t = dtime(h2, m2)

            if start_t <= end_t:
                # Same-day window
                if start_t <= local_t < end_t:
                    return True
            else:
                # Overnight window
                if local_t >= start_t or local_t < end_t:
                    return True

        return False
    

    def _build_facts_from_summary(self, predictions_summary: dict) -> dict:
        return {
            "object": dict(predictions_summary["object_detection_stats"]), 
            "plate": dict(predictions_summary["plate_recognition_stats"]),
            "motion": True,
            "n_frames": predictions_summary.get("n_frames", 0)
        }

    def _evaluate_alarm_rules(self, predictions_summary: dict, rules: dict, frame_timestamp: datetime):
        triggered_alarms = []
        for rule in rules:
            if not self._rule_in_schedule(rule, frame_timestamp):
                continue

            facts = self._build_facts_from_summary(predictions_summary)
            
            rule_matched = jsonLogic(rule["expr"], facts)
            if rule_matched:
                triggered_alarms.append(rule)

        return triggered_alarms

    
    def _upload_alarm_images(
        self,
        stream_name: str,
        frame_timestamp: datetime,
        original_image_pil: Image.Image,
        drawn_image_pil: Image.Image,
        verbose: bool = False,
    ):
        start = time.time()
        image_filename = f"{stream_name}/{frame_timestamp.isoformat()}_original.jpg"
        original_image_s3_url = self.notifier.upload_image_to_s3(original_image_pil, image_filename)
        if verbose:
            self.logger.info(f"Uploaded original image to S3 in {(time.time() - start) * 1000:.2f} ms")

        start = time.time()
        image_filename = f"{stream_name}/{frame_timestamp.isoformat()}_detections.jpg"
        drawn_image_s3_url = self.notifier.upload_image_to_s3(drawn_image_pil, image_filename)
        if verbose:
            self.logger.info(f"Uploaded drawn image to S3 in {(time.time() - start) * 1000:.2f} ms")

        return original_image_s3_url, drawn_image_s3_url

    def check_alarms(
        self,
        stream_name: str,
        predictions_summary: dict,
        frame_predictions: dict,
        event_timestamp: datetime,
        frame_timestamp: datetime,
        original_image_pil: Image.Image,
        drawn_image_pil: Image.Image,
        exclude_rules: dict = None,
        verbose: bool = False,
    ):
        """
        Check if predictions_summary match the alarm configuration for the device.
        If they do, trigger outbound notifications (email/WhatsApp) and store the alarm.
        """
        start = time.time()
        alarm_config = get_alarm_config(stream_name)
        if not alarm_config:
            if verbose:
                self.logger.info(f"No alarm configuration found for device {stream_name}.")
            return False
        if verbose:
            self.logger.info(f"Retrieved alarm config in {(time.time() - start) * 1000:.2f} ms")

        start = time.time()

        # TODO: remove exclude_rules from checking alarms

        
        # Evaluate rules
        matched_rules = self._evaluate_alarm_rules(
            predictions_summary=predictions_summary,
            rules=alarm_config.get("rules", []),
            frame_timestamp=frame_timestamp,
        )

        # Check cooldown and send/store for each triggered rule
        triggered_rules = []
        for trule in matched_rules:
            if trule["id"] in exclude_rules:
                print(f"Skipping already triggered rule {trule['id']}")
                continue

            cooldown_sec = int(trule.get("cooldown_seconds", trule.get("cooldown", 0)) or 0)
            if AlarmCooldown.try_acquire(stream_name, trule.get("id"), cooldown_sec):
                self.logger.info(f"Alarm cooldown passed for rule {trule.get('id')}.")
                triggered_rules.append(trule.get("id"))

                # Store alarm in background
                _run_on_background(
                    self._store_event_alarm,
                    stream_name=stream_name,
                    frame_timestamp=frame_timestamp,
                    alarm_rule=trule,
                    original_image_pil=original_image_pil.copy(),
                    drawn_image_pil=drawn_image_pil.copy(),
                    predictions_summary=predictions_summary,
                    frame_predictions=frame_predictions,
                    verbose=verbose,
                )

                # Send notifications in background
                _run_on_background(
                    self.send_nofitications,
                    stream_name=stream_name,
                    alarm_rule=trule,
                    alarm_config=alarm_config,
                    event_timestamp=event_timestamp,
                    frame_timestamp=frame_timestamp,
                    image_pil=drawn_image_pil.copy(),
                )
            else:
                self.logger.info(f"Alarm cooldown not passed for rule {trule.get('id')}.")

            
        return triggered_rules

    def _merge_channels(self, default_channels: dict | None, rule_channels: dict | None) -> dict:
        """Merge default and per-rule channels; rule overrides defaults per channel key."""
        merged = {}
        if default_channels:
            for k, v in default_channels.items():
                merged[k] = dict(v) if isinstance(v, dict) else v
        if rule_channels:
            for k, v in rule_channels.items():
                merged[k] = dict(v) if isinstance(v, dict) else v
        return merged

    def send_nofitications(
        self,
        stream_name: str,
        alarm_rule: dict,
        alarm_config: dict,
        event_timestamp: datetime,
        frame_timestamp: datetime,
        image_pil: Image.Image,
    ):
        channels = self._merge_channels(
            alarm_config.get("default_channels"), alarm_rule.get("channels")
        )

        # Fallback values; rule evaluation can later supply richer context
        object_class = alarm_rule.get("id", "unknown")
        confidence = float(alarm_rule.get("confidence", 0.0))
        alarm_time = frame_timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
        event_time = event_timestamp.isoformat()

        if "email" in channels and channels["email"].get("recipients"):
            self.logger.info("Sending email alarm notification...")
            _run_on_background(
                self.notifier.send_email_alarm,
                device_id=stream_name,
                object_class=object_class,
                confidence=confidence,
                event_formatted_time=event_time,
                alarm_formatted_time=alarm_time,
                image_pil=image_pil.copy(),
                to_addrs=channels["email"]["recipients"],
                from_addr="notifications@immediateaction.io",
            )

        if "whatsapp" in channels and channels["whatsapp"].get("numbers"):
            self.logger.info("Sending WhatsApp alarm notification...")
            _run_on_background(
                self.notifier.send_whatsapp_alarm,
                device_id=stream_name,
                object_class=object_class,
                confidence=confidence,
                alarm_formatted_time=alarm_time,
                event_formatted_time=event_time,
                image_pil=image_pil.copy(),
                to_numbers=channels["whatsapp"]["numbers"],
            )
    
    def _store_event_alarm(
        self,
        stream_name: str,
        frame_timestamp: datetime,
        alarm_rule: dict,
        original_image_pil: Image.Image,
        drawn_image_pil: Image.Image,
        predictions_summary: dict = None,
        frame_predictions: dict = None,
        verbose: bool = False,
    ):
        """
        Uploads images via _upload_alarm_images and stores the activation in DynamoDB.
        Assumes self.alarms_table points to 'event_alarms' (PK=device_id, SK=frame_timestamp).
        """
        s3_url_original, s3_url_drawn = self._upload_alarm_images(
            stream_name=stream_name,
            frame_timestamp=frame_timestamp,
            original_image_pil=original_image_pil,
            drawn_image_pil=drawn_image_pil,
            verbose=verbose,
        )

        frame_ts_iso = frame_timestamp.isoformat().replace("+00:00", "Z")

        # dynamodb does not work with empty sets, so we need to convert to lists
        dynamodb_pred_summary = predictions_summary.copy() if predictions_summary else {}
        dynamodb_pred_summary["seen_classes"] = list(dynamodb_pred_summary.get("seen_classes", []))
        dynamodb_pred_summary["seen_plates"] = list(dynamodb_pred_summary.get("seen_plates", []))
        item = {
            "device_id": stream_name,            # PK
            "frame_timestamp": frame_ts_iso,     # SK (lexicographically sortable)
            "created_at": datetime.now().isoformat().replace("+00:00", "Z"),
            # Store rule context instead of previous alarm_type/rules/channels
            "alarm_rule_id": alarm_rule.get("id"),
            "alarm_rule": convert_floats(alarm_rule),
            "s3_url_original": s3_url_original,
            "s3_url_detections": s3_url_drawn,
            "predictions_summary": convert_floats(dynamodb_pred_summary) if dynamodb_pred_summary else None,
            "frame_predictions": convert_floats(frame_predictions) if frame_predictions else None,
        }

        self.alarms_table.put_item(Item=item)

        if verbose:
            self.logger.info(
                f"Stored alarm in DynamoDB: device={stream_name} ts={frame_ts_iso} rule_id={item.get('alarm_rule_id')}"
            )

        return {
            "frame_timestamp": frame_ts_iso,
            "s3_url_original": s3_url_original,
            "s3_url_drawn": s3_url_drawn,
        }
        
