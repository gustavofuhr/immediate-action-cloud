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

    def dep_evaluate_alarm_rules(self, predictions_summary: dict, rules: dict):
        # 1) Any motion: fire immediately
        if "any_motion" in rules:
            return True, "any_motion", None

        # 2) Object-based rules
        if "object" in rules:
            targets = rules["object"].get("targets", {})  # {label: threshold}
            target_classes = targets.keys()
            # Choose the highest-confidence valid detection (stable and deterministic)
            best_det = None
            for detected_class in predictions_summary.get("seen_classes", []):
                max_conf = predictions_summary["object_detection_stats"].get(detected_class, {}).get("max_confidence", 0.0)
                
                if detected_class in target_classes:
                    thr = targets[detected_class].get("min_confidence", None)
                    if thr is not None and max_conf >= float(thr):
                        if best_det is None or max_conf > best_det.get("confidence", 0.0):
                            best_det = {
                                "label": detected_class,
                                "confidence": max_conf
                            }
            if best_det is not None:
                return True, "object_detection", best_det

        # 3) Plate rules 
        if "plate" in rules:
            seen_plates = predictions_summary.get("seen_plates", [])
            if seen_plates:
                target_plates = {self._normalize_plate_text(p): v for p, v in rules["plate"].get("targets", {}).items()}
                for detected_plate in seen_plates:
                    det_plate_text = self._normalize_plate_text(detected_plate)
                    plate_confidence = predictions_summary["plate_recognition_stats"].get(det_plate_text, {}).get("max_confidence", 0.0)
                    ocr_confidence = predictions_summary["plate_recognition_stats"].get(det_plate_text, {}).get("max_ocr_confidence", 0.0)

                    if det_plate_text in target_plates and \
                        float(plate_confidence) >= float(target_plates[det_plate_text]["min_plate_confidence"]) and \
                        float(ocr_confidence) >= float(target_plates[det_plate_text].get("min_ocr_confidence", 0.0)):
                        return True, "plate_detection", {
                            "label": det_plate_text,
                            "confidence": float(plate_confidence),
                            "ocr_confidence": float(ocr_confidence)
                        }
            

        return False, "", None
    
    def _rule_in_schedule(self, rule: dict, frame_timestamp: datetime) -> bool:
        sched = rule.get("schedule")
        if not sched:
            return True  # no schedule → always active

        print("Frame time: ", frame_timestamp.isoformat())
        DOW = ["mon","tue","wed","thu","fri","sat","sun"]
        tz = ZoneInfo(sched.get("timezone", "UTC"))
        local_dt = frame_timestamp.astimezone(tz)
        local_day = DOW[local_dt.weekday()]
        local_t = local_dt.time().replace(second=0, microsecond=0)

        print(f"[DEBUG] Schedule: {sched}, Local day: {local_day}, Local time: {local_t}")

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
        for rule in rules:
            if not self._rule_in_schedule(rule, frame_timestamp):
                self.logger.info(f"Rule {rule['id']} is not in schedule.")
                continue

            print("Evaluating rule:")
            print(json.dumps(rule["expr"], indent=2))

            facts = self._build_facts_from_summary(predictions_summary)
            print("With facts:")
            print(json.dumps(facts, indent=2))

            triggered_rule = jsonLogic(rule["expr"], facts)
            print("Result:", triggered_rule)
            print()
            # alarm_detected, alarm_type, alarm_detection = self._check_rule_conditions(rule, predictions_summary)
            # if alarm_detected:
            #     return True, alarm_type, alarm_detection

        # print(rules)
        

    
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

    def check_alarm(
        self,
        stream_name: str,
        predictions_summary: dict,
        frame_predictions: dict,
        event_timestamp: datetime,
        frame_timestamp: datetime,
        original_image_pil: Image.Image,
        drawn_image_pil: Image.Image,
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

        # Evaluate rules (your real implementation should replace the stub below)
        alarm_detected, alarm_type, alarm_detection = self._evaluate_alarm_rules(
            predictions_summary=predictions_summary,
            rules=alarm_config.get("rules", {}),
            frame_timestamp=frame_timestamp,
        )

        if not alarm_detected:
            return False

        alarm_object_class = alarm_detection["label"] if alarm_detection else "unknown"
        alarm_confidence = float(alarm_detection.get("confidence", 0.0)) if alarm_detection else 0.0
        alarm_time = frame_timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")

        self.logger.info(f"Alarm triggered for device {stream_name} at {frame_timestamp}")
        self.logger.info(f"Alarm type: {alarm_type}")
        self.logger.info(f"Alarm time: {alarm_time}")
        self.logger.info(f"Alarm confidence: {alarm_confidence:.2f}")
        self.logger.info(f"Alarm object class: {alarm_object_class}")

        # ---- SEND NOTIFICATIONS DIRECTLY VIA Notifier (background) ----
        channels = alarm_config.get("channels", {})

        if "email" in channels:
            self.logger.info("Sending email alarm notification...")
            _run_on_background(
                self.notifier.send_email_alarm,
                device_id=stream_name,
                object_class=alarm_object_class,
                confidence=alarm_confidence,
                event_formatted_time=event_timestamp.isoformat(),
                alarm_formatted_time=alarm_time,
                image_pil=drawn_image_pil.copy(),
                to_addrs=channels["email"]["recipients"],
                from_addr="notifications@immediateaction.io",
            )

        if "whatsapp" in channels:
            self.logger.info("Sending WhatsApp alarm notification...")
            _run_on_background(
                self.notifier.send_whatsapp_alarm,
                device_id=stream_name,
                object_class=alarm_object_class,
                confidence=alarm_confidence,
                alarm_formatted_time=alarm_time,
                event_formatted_time=event_timestamp.isoformat(),
                image_pil=drawn_image_pil.copy(),
                to_numbers=channels["whatsapp"]["numbers"],
            )

        # ---- STORE EVENT (background) ----
        _run_on_background(
            self._store_event_alarm,
            stream_name=stream_name,
            frame_timestamp=frame_timestamp,
            alarm_type=alarm_type,
            original_image_pil=original_image_pil.copy(),
            drawn_image_pil=drawn_image_pil.copy(),
            rules=alarm_config.get("rules", {}),
            channels=channels,
            object_class=alarm_object_class,
            confidence=alarm_confidence,
            predictions_summary=predictions_summary,
            frame_predictions=frame_predictions,
            verbose=verbose,
        )

        return True
    
    def _store_event_alarm(
        self, 
        stream_name: str, 
        frame_timestamp: datetime, 
        alarm_type: str, 
        original_image_pil: Image.Image, 
        drawn_image_pil: Image.Image,
        rules : dict,
        channels: dict,
        object_class: str = None,
        confidence: float = None,
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
            "alarm_type": alarm_type,
            "s3_url_original": s3_url_original,
            "s3_url_detections": s3_url_drawn,
            "rules": convert_floats(rules),
            "channels": convert_floats(channels),
            "predictions_summary": convert_floats(dynamodb_pred_summary) if dynamodb_pred_summary else None,
            "frame_predictions": convert_floats(frame_predictions) if frame_predictions else None,
        }
        if object_class:
            item["detection_label"] = object_class
        if confidence is not None:
            item["detection_confidence"] = convert_floats(confidence)

        self.alarms_table.put_item(Item=item)

        if verbose:
            self.logger.info(f"Stored alarm in DynamoDB: device={stream_name} ts={frame_ts_iso} type={alarm_type}")

        return {
            "frame_timestamp": frame_ts_iso,
            "s3_url_original": s3_url_original,
            "s3_url_drawn": s3_url_drawn,
        }
        