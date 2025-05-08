import sys
import json
import ssl
import time
import threading

import paho.mqtt.client as mqtt

from event_ai_processor import EventAIProcessor
from lambda_function import lambda_handler 

# MQTT configuration
AWS_IOT_ENDPOINT = "arpj530io2lue-ats.iot.eu-west-1.amazonaws.com"
AWS_IOT_PORT = 8883
CLIENT_ID = "test_thing"
TOPIC = "cameras/axis-local/events/motion/start"

# using certs from test_thing
ROOT_CA_PATH="/home/gfuhr/projects/immediate-action-camera/AmazonRootCA1.pem"
CERTIFICATE_PATH="/home/gfuhr/projects/immediate-action-camera/ia-tiny-cam-manager/credentials/test_thing/certificate.pem"
PRIVATE_KEY_PATH="/home/gfuhr/projects/immediate-action-camera/ia-tiny-cam-manager/credentials/test_thing/private.key"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected successfully to AWS IoT Core.")
        client.subscribe(TOPIC)
        print(f"Subscribed to topic: {TOPIC}")
    else:
        print(f"Connection failed with code {rc}")

def on_message(client, userdata, msg):
    print(f"Received message on topic {msg.topic}")
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        payload['topic'] = msg.topic  # Attach topic to the payload (needed for your lambda_handler)
        print(f"Payload: {json.dumps(payload, indent=4)}")
        threading.Thread(target=lambda_handler, args=(payload, None)).start()
    except Exception as e:
        print(f"Error processing message: {e}")

def main():
    client = mqtt.Client(client_id=CLIENT_ID)
    
    client.tls_set(
        ca_certs=ROOT_CA_PATH,
        certfile=CERTIFICATE_PATH,
        keyfile=PRIVATE_KEY_PATH,
        cert_reqs=ssl.CERT_REQUIRED,
        tls_version=ssl.PROTOCOL_TLSv1_2,
        ciphers=None
    )

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(AWS_IOT_ENDPOINT, AWS_IOT_PORT, keepalive=60)

    client.loop_forever()

if __name__ == "__main__":
    main()
