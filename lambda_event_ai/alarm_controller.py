import time
import io
from datetime import datetime
import json
from io import BytesIO

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

import boto3
from PIL import Image

from lambda_config import get_alarm_config

sns = boto3.client('sns')


EMAIL_PLAIN_TEXT = """
    An important event was detected in stream {0}: {1} with confidence {2} at {3}. 

    An image snapshot can be found at the following link: {4}
    """
SNS_TOPIC_ARN = "arn:aws:sns:eu-west-1:354918369325:event-alarms"


EMAIL_HTML = """
<html>
  <body>
    <h3>Event Alert: {device_id}</h3>
    <p>
      <b>Class:</b> {object_class}<br/>
      <b>Confidence:</b> {confidence:.2f}<br/>
      <b>Time:</b> {alarm_time}
    </p>
    <p>Snapshot:</p>
    <img src="cid:snapshot" alt="snapshot"/>
  </body>
</html>
"""

class AlarmController:

    def __init__(self):
        self.s3 = boto3.client('s3')
        self.bucket_name = "immediate-action-alarm-snapshots"
        # self.sns = boto3.client('sns')
        self.ses = boto3.client('ses', region_name='eu-west-1')


    def send_text_email_alarm(self, device_id : str, object_class : str, confidence : float, alarm_formatted_time : str, image_url : str):
        raise NotImplementedError("This method is deprecated. Use send_email_alarm instead.")
        self.sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Message=EMAIL_PLAIN_TEXT.format(
                device_id, object_class, confidence, alarm_formatted_time, image_url
            ),
            Subject="Event Alert! {0}".format(device_id),
        )

    def send_email_alarm(self, device_id : str, object_class : str, confidence : float, alarm_formatted_time : str, 
                                image_pil : Image.Image, to_addrs : list[str], from_addr : str):
        max_w, max_h = 1280, 720
        w, h = image_pil.size
        if w > max_w or h > max_h:
            image_pil = image_pil.copy()
            image_pil.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

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
            confidence=confidence, alarm_time=alarm_formatted_time)
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

    def send_whatsapp_alarm(self, device_id : str, object_class : str, confidence : float, timestamp : str, image_url : str):
        variables = {
            "1": device_id,
            "2": object_class,
            "3": str(confidence),
            "4": timestamp,
            "42": image_url.replace("https://immediate-action-alarm-snapshots.s3.amazonaws.com/", "")
        }

        if self.twilio_client is None:
            from twilio.rest import Client
            account_sid = 'AC25c10375466eea035619587010999fdf'
            auth_token = '62ff3ca6549deed92f749e8c612fa2ee'
        
            self.twilio_client = Client(account_sid, auth_token)

        from_whatsapp = 'whatsapp:+14155238886'
        to_whatsapp = 'whatsapp:+555196039983'

        message = self.twilio_client.messages.create(
            from_=from_whatsapp,
            to=to_whatsapp,
            content_sid="HX7ebfdb1adfd14320b4bc430c67385077", 
            content_variables=json.dumps(variables)
        )

        return message.sid

    def upload_image_to_s3(self, device_id : str, image_pil : Image.Image, frame_timestamp : str):
        file_name = f"{device_id}/{frame_timestamp}.jpg"
        
        buffer = io.BytesIO()
        image_pil.save(buffer, format='JPEG')
        buffer.seek(0)  

        self.s3.upload_fileobj(buffer, self.bucket_name, file_name, ExtraArgs={'ContentType': 'image/jpeg'})

        # Return pre-signed URL
        return self.s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.bucket_name, 'Key': file_name},
            ExpiresIn=3600  # 1 hour
        )
    
    def _normalize_plate_text(self, plate_text: str) -> str:
        return plate_text.upper().replace("-", "")

    def _evaluate_alarm_rules(self, predictions: dict, rules: dict):
        # 1) Any motion: fire immediately
        if "any_motion" in rules:
            return True, "any_motion", None

        # 2) Object-based rules
        if "object" in rules:
            targets = rules["object"].get("targets", {})  # {label: threshold}
            # Choose the highest-confidence valid detection (stable and deterministic)
            best_det = None
            for model_name, out in predictions.items():
                for det in out.get("detections", []):
                    lbl = det.get("label")
                    conf = float(det.get("confidence", 0.0))
                    thr = targets.get(lbl)
                    if thr is not None and conf >= float(thr):
                        if best_det is None or conf > best_det.get("confidence", 0.0):
                            best_det = det
            if best_det is not None:
                return True, "object_detection", best_det

        # 3) Plate rules 
        if "plate" in rules:
            plates = []
            for model_name, out in predictions.items():
                if model_name.startswith("object_detection"):
                    if "lpr" in model_name:
                        plates.extend([
                            plate
                            for det in out.get("detections", [])
                            if "license_plate" in det and len(det["license_plate"]) > 0
                            for plate in det["license_plate"]
                        ])
                if model_name == "license_plate_recognition":
                    plates.extend(out.get("detections", []))
            print("plates found: ", [p["ocr_text"] for p in plates])
            if plates:
                target_plates = {self._normalize_plate_text(p): v for p, v in rules["plate"].get("targets", {}).items()}
                for plate in plates:
                    plate_text = self._normalize_plate_text(plate.get("ocr_text", ""))
                    if plate_text in target_plates and \
                        float(plate.get("confidence", 0.0)) >= float(target_plates[plate_text]["min_plate_confidence"]) and \
                        float(plate.get("ocr_confidence", 0.0)) >= float(target_plates[plate_text].get("min_ocr_confidence", 0.0)):
                        return True, "plate_detection", {
                            "label": plate_text,
                            "confidence": float(plate.get("confidence", 0.0)),
                            "ocr_confidence": float(plate.get("ocr_confidence", 0.0))
                        }
            

        return False, "", None

    def check_alarm(self, stream_name : str, predictions : dict, frame_timestamp : datetime, image_pil : Image.Image, verbose : bool = False):
        """
        Check if the predictions match the alarm configuration for the device.
        If they do, send an alarm notification.
        """

        start = time.time()
        alarm_config = get_alarm_config(stream_name)
        if not alarm_config:
            if verbose: print(f"No alarm configuration found for device {stream_name}.")
            return
        elapsed = (time.time() - start) * 1000
        if verbose: print(f"[INFO] Retrieved alarm config in {elapsed:.2f} ms")

        # Check if any of the predicted classes match the alarm configuration
        start = time.time() 
        alarm_detected, alarm_type, alarm_detection = self._evaluate_alarm_rules(predictions=predictions, rules=alarm_config.get("rules", {}))
        elapsed = (time.time() - start) * 1000
        # if verbose: print(f"[INFO] Checked predictions for in {elapsed:.2f} ms")

        # TODO: sending an alarm can take a while, so we should not block the main thread
        if alarm_detected:
            alarm_object_class = alarm_detection["label"] if alarm_detection else "unknown"
            alarm_confidence = alarm_detection["confidence"] if alarm_detection else 0.0
            
            print(f"[INFO] Alarm triggered for device {stream_name} at {frame_timestamp}")
            print(f"\tAlarm type: {alarm_type}")
            print(f"\tAlarm time: {frame_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"\tAlarm confidence: {alarm_confidence:.2f}")
            print(f"\tAlarm object class: {alarm_object_class}")
            
            if "email" in alarm_config["channels"]:
                start = time.time()
                self.send_email_alarm(
                    device_id=stream_name,
                    object_class=alarm_object_class,
                    confidence=alarm_confidence,
                    alarm_formatted_time=frame_timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    image_pil=image_pil,
                    to_addrs=alarm_config["channels"]["email"]["recipients"],
                    from_addr="gustavo.fuhr@immediateaction.io"
                )
                if verbose: print(f"[INFO] Sent email alarm in {time.time() - start:.2f} seconds")

            if "whatsapp" in alarm_config["channels"]:
                # for whats app, I need to upload image to S3
                start = time.time()
                image_url = self.upload_image_to_s3(stream_name, image_pil, frame_timestamp)
                if verbose: print(f"[INFO] Uploaded alarm snapshot to S3 in {time.time() - start:.2f} seconds")

                start = time.time()
                self.send_whatsapp_alarm(
                    device_id=stream_name,
                    object_class=alarm_detection["label"],
                    confidence=alarm_detection["confidence"],
                    timestamp=frame_timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    image_url=image_url
                )
                if verbose: print(f"[INFO] Sent WhatsApp alarm in {time.time() - start:.2f} seconds")

        
        return alarm_detected
        