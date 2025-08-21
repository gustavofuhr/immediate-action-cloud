import time
from datetime import datetime
import json
from io import BytesIO
from threading import Thread
from decimal import Decimal, ROUND_HALF_UP
import math
import traceback

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

import boto3
from PIL import Image

from lambda_config import get_alarm_config
from lambda_logging import base_logger



EMAIL_PLAIN_TEXT = """
    An important event was detected in stream {0}: {1} with confidence {2} at {3}. 

    An image snapshot can be found at the following link: {4}
    """
SNS_TOPIC_ARN = "arn:aws:sns:eu-west-1:354918369325:event-alarms"


EMAIL_HTML = """
<html>
  <body>
    <h4>Event Alert: {device_id}</h4>
    <p>
      <b>Class:</b> {object_class}<br/>
      <b>Confidence:</b> {confidence:.2f}<br/>
      <b>Time:</b> {alarm_time}<br/><br/>
      <i>Event timestamp: {event_timestamp}</i>
    </p>
    <img src="cid:snapshot" alt="snapshot"/>
  </body>
</html>
"""

WHATSAPP_ORIGINATION_PHONE_NUMBER_ID = "phone-number-id-cc7253fceb0947f0be779ea4f0f4fdde"
# WHATSAPP_TO_E164 = "+5551996039983"  
WHATSAPP_API_VERSION = "v20.0"
WHATSAPP_REGION = "eu-west-1"

WHATSAPP_TEMPLATE_NAME = "event_alert" 
WHATSAPP_TEMPLATE_LANG = "en" 


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
        self.s3 = boto3.client('s3')
        self.bucket_name = "immediate-action-alarm-snapshots"
        # self.sns = boto3.client('sns')
        self.ses = boto3.client('ses', region_name='eu-west-1')
        self.dynamodb = boto3.resource("dynamodb", region_name="eu-west-1")
        self.alarms_table = self.dynamodb.Table("event_alarms")  # PK=device_id, SK=frame_timestamp

        self.social = boto3.client("socialmessaging", region_name=WHATSAPP_REGION)


    def send_text_email_alarm(self, device_id : str, object_class : str, confidence : float, alarm_formatted_time : str, image_url : str):
        raise NotImplementedError("This method is deprecated. Use send_email_alarm instead.")
        self.sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Message=EMAIL_PLAIN_TEXT.format(
                device_id, object_class, confidence, alarm_formatted_time, image_url
            ),
            Subject="Event Alert! {0}".format(device_id),
        )
    
    def _resize_image_to_send(self, image_pil: Image.Image, max_w: int = 1280, max_h: int = 720) -> Image.Image:
        w, h = image_pil.size
        if w > max_w or h > max_h:
            image_pil = image_pil.copy()
            image_pil.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

        return image_pil

    def send_email_alarm(self, device_id : str, object_class : str, confidence : float, event_formatted_time : str, alarm_formatted_time : str, 
                                image_pil : Image.Image, to_addrs : list[str], from_addr : str):
        image_pil = self._resize_image_to_send(image_pil)
        
        buf = BytesIO()
        image_pil.save(buf, format="JPEG", quality=90)
        img_bytes = buf.getvalue()

        msg_root = MIMEMultipart("related")
        msg_root["Subject"] = f"Event Alert! {device_id}"
        msg_root["From"] = from_addr
        msg_root["To"] = ", ".join(to_addrs)

        alt = MIMEMultipart("alternative")
        msg_root.attach(alt)

        text_part = f"Event Alert: {device_id}\nClass: {object_class}\nConfidence: {confidence:.2f}\nTime: {alarm_formatted_time}"
        alt.attach(MIMEText(text_part, "plain", "utf-8"))

        html_part = EMAIL_HTML.format(device_id=device_id, object_class=object_class,
            confidence=confidence, event_timestamp=event_formatted_time, alarm_time=alarm_formatted_time)
        alt.attach(MIMEText(html_part, "html", "utf-8"))

        img = MIMEImage(img_bytes, _subtype="jpeg", name="snapshot.jpg")
        img.add_header("Content-ID", "<snapshot>")
        img.add_header("Content-Disposition", "inline", filename="snapshot.jpg")
        msg_root.attach(img)

        self.ses.send_raw_email(
            Source=from_addr,
            Destinations=to_addrs,
            RawMessage={"Data": msg_root.as_string()},
        )

        
    def send_whatsapp_alarm(self, device_id: str, object_class: str, confidence: float,
                            alarm_formatted_time: str, event_formatted_time: str, image_pil: Image.Image, to_numbers: list[str]):
        # 1) Resize and upload image
        img = self._resize_image_to_send(image_pil)
        s3_key = f"{device_id}/{alarm_formatted_time.replace(' ', 'T')}_whatsapp.jpg"
        self.upload_image_to_s3(img, s3_key, generate_presigned_url=False)

        # 2) Register media with WhatsApp
        upload = self.social.post_whatsapp_message_media(
            originationPhoneNumberId=WHATSAPP_ORIGINATION_PHONE_NUMBER_ID,
            sourceS3File={"bucketName": self.bucket_name, "key": s3_key},
        )
        media_id = upload["mediaId"]
        template = {
            "name": WHATSAPP_TEMPLATE_NAME,
            "language": {"code": WHATSAPP_TEMPLATE_LANG},
            "components": [
                {   # header image
                    "type": "header",
                    "parameters": [
                        {"type": "image", "image": {"id": media_id}}
                    ],
                },
                {   # body variables in numeric order
                    "type": "body",
                    "parameters": [
                        {"type": "text", "text": device_id},                 # {{1}}
                        {"type": "text", "text": object_class},              # {{2}}
                        {"type": "text", "text": f"{confidence:.2f}"},       # {{3}}
                        {"type": "text", "text": alarm_formatted_time},      # {{4}}
                        {"type": "text", "text": event_formatted_time},      # {{5}}
                    ],
                },
            ],
        }
        results = {}
        for number in to_numbers:
            payload = {
                "messaging_product": "whatsapp",
                "to": number,
                "type": "template",
                "template": template,
            }
            self.logger.info(f"Sending WhatsApp to {number}")
            try:
                resp = self.social.send_whatsapp_message(
                    originationPhoneNumberId=WHATSAPP_ORIGINATION_PHONE_NUMBER_ID,
                    message=json.dumps(payload).encode("utf-8"),
                    metaApiVersion=WHATSAPP_API_VERSION,
                )
                results[number] = resp.get("messageId")
            except Exception as e:
                results[number] = str(e)

        return results


    def _normalize_plate_text(self, plate_text: str) -> str:
        return plate_text.upper().replace("-", "")

    def _evaluate_alarm_rules(self, predictions_summary: dict, rules: dict):
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
    
    def upload_image_to_s3(self, image_pil : Image.Image, filename : str, generate_presigned_url : bool = False):
        buffer = BytesIO()
        image_pil.save(buffer, format='JPEG')
        buffer.seek(0)  

        self.s3.upload_fileobj(buffer, self.bucket_name, filename, ExtraArgs={'ContentType': 'image/jpeg'})

        if not generate_presigned_url:
            return f"s3://{self.bucket_name}/{filename}"
        

        return self.s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.bucket_name, 'Key': filename},
            ExpiresIn=3600  # 1 hour
        )
    
    def _upload_alarm_images(self, 
                            stream_name : str, 
                            frame_timestamp : datetime, 
                            original_image_pil : Image.Image, 
                            drawn_image_pil : Image.Image, 
                            verbose : bool = False):
        """
        Upload the image snapshots to S3 and return image s3 URLs.
        """
        start = time.time()
        image_filename = f"{stream_name}/{frame_timestamp.isoformat()}_original.jpg"
        original_image_s3_url = self.upload_image_to_s3(original_image_pil, image_filename)
        elapsed = (time.time() - start) * 1000
        if verbose: self.logger.info(f"Uploaded original image to S3 in {elapsed:.2f} ms")

        start = time.time()
        image_filename = f"{stream_name}/{frame_timestamp.isoformat()}_detections.jpg"
        drawn_image_s3_url = self.upload_image_to_s3(drawn_image_pil, image_filename)
        elapsed = (time.time() - start) * 1000
        if verbose: self.logger.info(f"Uploaded drawn image to S3 in {elapsed:.2f} ms")

        return original_image_s3_url, drawn_image_s3_url

    def check_alarm(self, stream_name : str, predictions_summary : dict, frame_predictions : dict, event_timestamp : datetime,
                        frame_timestamp : datetime, original_image_pil : Image.Image, drawn_image_pil : Image.Image, verbose : bool = False):
        """
        Check if the predictions_summary match the alarm configuration for the device.
        If they do, send an alarm notification.
        """

        start = time.time()
        alarm_config = get_alarm_config(stream_name)
        if not alarm_config:
            if verbose: self.logger.info(f"No alarm configuration found for device {stream_name}.")
            return False
        elapsed = (time.time() - start) * 1000
        if verbose: self.logger.info(f"Retrieved alarm config in {elapsed:.2f} ms")

        # Check if any of the predicted classes match the alarm configuration
        start = time.time() 
        alarm_detected, alarm_type, alarm_detection = self._evaluate_alarm_rules(predictions_summary=predictions_summary, rules=alarm_config.get("rules", {}))
        elapsed = (time.time() - start) * 1000
        # if verbose: self.logger.info(f"Checked predictions_summary for in {elapsed:.2f} ms")

        # TODO: sending an alarm can take a while, so we should not block the main thread
        if alarm_detected:
            alarm_object_class = alarm_detection["label"] if alarm_detection else "unknown"
            alarm_confidence = alarm_detection["confidence"] if alarm_detection else 0.0
            alarm_time = frame_timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")

            self.logger.info(f"Alarm triggered for device {stream_name} at {frame_timestamp}")
            self.logger.info(f"Alarm type: {alarm_type}")
            self.logger.info(f"Alarm time: {alarm_time}")
            self.logger.info(f"Alarm confidence: {alarm_confidence:.2f}")
            self.logger.info(f"Alarm object class: {alarm_object_class}")

            if "email" in alarm_config["channels"]:
                self.logger.info("Sending email alarm notification...")
                _run_on_background(
                    self.send_email_alarm,
                    device_id=stream_name,
                    object_class=alarm_object_class,
                    confidence=alarm_confidence,
                    event_formatted_time=event_timestamp.isoformat(),
                    alarm_formatted_time=alarm_time,
                    image_pil=drawn_image_pil.copy(),
                    to_addrs=alarm_config["channels"]["email"]["recipients"],
                    from_addr="notifications@immediateaction.io",
                )
                
            if "whatsapp" in alarm_config["channels"]:
                self.logger.info("Sending whatsapp alarm notification...")
                _run_on_background(self.send_whatsapp_alarm,
                    device_id=stream_name,
                    object_class=alarm_object_class,
                    confidence=alarm_confidence,
                    alarm_formatted_time=alarm_time,
                    event_formatted_time=event_timestamp.isoformat(),
                    image_pil=drawn_image_pil.copy(),
                    to_numbers=alarm_config["channels"]["whatsapp"]["numbers"]
                )

            _run_on_background(
                self._store_event_alarm,
                stream_name=stream_name,
                frame_timestamp=frame_timestamp,
                alarm_type=alarm_type,
                original_image_pil=original_image_pil.copy(),
                drawn_image_pil=drawn_image_pil.copy(),
                rules=alarm_config.get("rules", {}),
                channels=alarm_config.get("channels", {}),
                object_class=alarm_object_class,
                confidence= alarm_confidence,
                predictions_summary=predictions_summary,
                frame_predictions=frame_predictions,
                verbose=verbose,
            )
        
        return alarm_detected
    
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
        