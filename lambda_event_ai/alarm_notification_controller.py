# notifications.py
import json
from io import BytesIO
from typing import List, Optional

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

import boto3
from PIL import Image

from lambda_logging import base_logger

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

class AlarmNotificationController:
    """
    Handles all outbound notifications (SES email and WhatsApp via AWS Social Messaging),
    plus S3 image uploads used by those channels.
    """

    def __init__(
        self,
        logger=None,
        bucket_name: str = "immediate-action-alarm-snapshots",
        ses_region: str = "eu-west-1",
        whatsapp_region: str = "eu-west-1",
        whatsapp_api_version: str = "v20.0",
        whatsapp_origination_phone_number_id: Optional[str] = None,
        whatsapp_template_name: str = "event_alert",
        whatsapp_template_lang: str = "en",
        s3_client=None,
        ses_client=None,
        social_client=None,
    ):
        self.logger = logger or base_logger
        self.bucket_name = bucket_name

        # Allow dependency injection for tests; fall back to real clients.
        self.s3 = s3_client or boto3.client("s3")
        self.ses = ses_client or boto3.client("ses", region_name=ses_region)
        self.social = social_client or boto3.client("socialmessaging", region_name=whatsapp_region)

        self.whatsapp_api_version = whatsapp_api_version
        self.whatsapp_origination_phone_number_id = whatsapp_origination_phone_number_id
        self.whatsapp_template_name = whatsapp_template_name
        self.whatsapp_template_lang = whatsapp_template_lang

    # ---------- shared helpers ----------

    @staticmethod
    def _resize_image_to_send(image_pil: Image.Image, max_w: int = 1280, max_h: int = 720) -> Image.Image:
        w, h = image_pil.size
        if w > max_w or h > max_h:
            image_pil = image_pil.copy()
            image_pil.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        return image_pil

    def upload_image_to_s3(self, image_pil: Image.Image, key: str, generate_presigned_url: bool = False) -> str:
        buf = BytesIO()
        image_pil.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        self.s3.upload_fileobj(buf, self.bucket_name, key, ExtraArgs={"ContentType": "image/jpeg"})

        if not generate_presigned_url:
            return f"s3://{self.bucket_name}/{key}"

        return self.s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket_name, "Key": key},
            ExpiresIn=3600,
        )

    # ---------- EMAIL (SES) ----------

    def _build_email_message(
        self,
        device_id: str,
        object_class: str,
        confidence: float,
        event_formatted_time: str,
        alarm_formatted_time: str,
        image_pil: Image.Image,
        to_addrs: List[str],
        from_addr: str,
    ) -> str:
        img = self._resize_image_to_send(image_pil)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        img_bytes = buf.getvalue()

        msg_root = MIMEMultipart("related")
        msg_root["Subject"] = f"Event Alert! {device_id}"
        msg_root["From"] = from_addr
        msg_root["To"] = ", ".join(to_addrs)

        alt = MIMEMultipart("alternative")
        msg_root.attach(alt)

        text_part = (
            f"Event Alert: {device_id}\n"
            f"Class: {object_class}\n"
            f"Confidence: {confidence:.2f}\n"
            f"Time: {alarm_formatted_time}"
        )
        alt.attach(MIMEText(text_part, "plain", "utf-8"))

        html_part = EMAIL_HTML.format(
            device_id=device_id,
            object_class=object_class,
            confidence=confidence,
            event_timestamp=event_formatted_time,
            alarm_time=alarm_formatted_time,
        )
        alt.attach(MIMEText(html_part, "html", "utf-8"))

        img_mime = MIMEImage(img_bytes, _subtype="jpeg", name="snapshot.jpg")
        img_mime.add_header("Content-ID", "<snapshot>")
        img_mime.add_header("Content-Disposition", "inline", filename="snapshot.jpg")
        msg_root.attach(img_mime)

        return msg_root.as_string()

    def send_email_alarm(
        self,
        device_id: str,
        object_class: str,
        confidence: float,
        event_formatted_time: str,
        alarm_formatted_time: str,
        image_pil: Image.Image,
        to_addrs: List[str],
        from_addr: str,
    ):
        raw = self._build_email_message(
            device_id=device_id,
            object_class=object_class,
            confidence=confidence,
            event_formatted_time=event_formatted_time,
            alarm_formatted_time=alarm_formatted_time,
            image_pil=image_pil,
            to_addrs=to_addrs,
            from_addr=from_addr,
        )
        self.ses.send_raw_email(Source=from_addr, Destinations=to_addrs, RawMessage={"Data": raw})

    # ---------- WHATSAPP (AWS Social Messaging) ----------

    def _register_whatsapp_media(self, s3_key: str) -> str:
        resp = self.social.post_whatsapp_message_media(
            originationPhoneNumberId=self.whatsapp_origination_phone_number_id,
            sourceS3File={"bucketName": self.bucket_name, "key": s3_key},
        )
        return resp["mediaId"]

    def send_whatsapp_alarm(
        self,
        device_id: str,
        object_class: str,
        confidence: float,
        alarm_formatted_time: str,
        event_formatted_time: str,
        image_pil: Image.Image,
        to_numbers: List[str],
    ) -> dict:
        # 1) Resize & upload image
        img = self._resize_image_to_send(image_pil)
        s3_key = f"{device_id}/{alarm_formatted_time.replace(' ', 'T')}_whatsapp.jpg"
        self.upload_image_to_s3(img, s3_key, generate_presigned_url=False)

        # 2) Register media and build template
        media_id = self._register_whatsapp_media(s3_key)
        template = {
            "name": self.whatsapp_template_name,
            "language": {"code": self.whatsapp_template_lang},
            "components": [
                {"type": "header", "parameters": [{"type": "image", "image": {"id": media_id}}]},
                {
                    "type": "body",
                    "parameters": [
                        {"type": "text", "text": device_id},
                        {"type": "text", "text": object_class},
                        {"type": "text", "text": f"{confidence:.2f}"},
                        {"type": "text", "text": alarm_formatted_time},
                        {"type": "text", "text": event_formatted_time},
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
            try:
                resp = self.social.send_whatsapp_message(
                    originationPhoneNumberId=self.whatsapp_origination_phone_number_id,
                    message=json.dumps(payload).encode("utf-8"),
                    metaApiVersion=self.whatsapp_api_version,
                )
                results[number] = resp.get("messageId")
            except Exception as e:
                results[number] = str(e)
                self.logger.error(f"WhatsApp send failed for {number}: {e}")

        return results
