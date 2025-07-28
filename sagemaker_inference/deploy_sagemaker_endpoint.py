import subprocess
import boto3
import sys

# ====== PARAMETERS ======
GPU = True
OBJECT_DETECTION_MODEL = "dfine_x_obj2coco"
AWS_REGION = "eu-west-1"

IMAGE_NAME = "sagemaker_inference_server_loadtest"
ECR_REPO_NAME = IMAGE_NAME

obj_models = {
    "dfine_s_obj2coco": {
        "CHECKPOINT_FILE": "/workspace/dfine_checkpoints/dfine_s_obj2coco.pth",
        "CONFIG_FILE": "/workspace/D-FINE/configs/dfine/objects365/dfine_hgnetv2_s_obj2coco.yml",
    },
    "dfine_l_coco": {
        "CHECKPOINT_FILE": "/workspace/dfine_checkpoints/dfine_l_coco.pth",
        "CONFIG_FILE": "/workspace/D-FINE/configs/dfine/dfine_hgnetv2_l_coco.yml",
    },
    "dfine_l_obj2coco": {
        "CHECKPOINT_FILE": "/workspace/dfine_checkpoints/dfine_l_obj2coco_e25.pth",
        "CONFIG_FILE": "/workspace/D-FINE/configs/dfine/objects365/dfine_hgnetv2_l_obj2coco.yml",
    },
    "dfine_x_obj2coco": {
        "CHECKPOINT_FILE": "/workspace/dfine_checkpoints/dfine_x_obj2coco.pth",
        "CONFIG_FILE": "/workspace/D-FINE/configs/dfine/objects365/dfine_hgnetv2_x_obj2coco.yml",
    }
}

CURRENT_INSTANCE_TYPE = "ml.g4dn.xlarge" if GPU else "ml.m5.xlarge"
device = "cpu" if CURRENT_INSTANCE_TYPE.startswith("ml.m5") else "cuda:0"

# ====== 1. BUILD DOCKER IMAGE AND PUBLISH TO ECR ======

# Build docker image
subprocess.run(["docker", "build", "-t", IMAGE_NAME, "."], check=True)

# Get AWS account ID
account_id = boto3.client("sts").get_caller_identity()["Account"]
ecr_uri = f"{account_id}.dkr.ecr.{AWS_REGION}.amazonaws.com/{ECR_REPO_NAME}"
print(f"ECR URL: {ecr_uri}")

# Create ECR repo (ignore error if exists)
subprocess.run([
    "aws", "ecr", "create-repository",
    "--repository-name", ECR_REPO_NAME,
    "--region", AWS_REGION
], check=False)

# Tag & push docker image
subprocess.run(["docker", "tag", f"{IMAGE_NAME}:latest", ecr_uri], check=True)
subprocess.run(["docker", "push", ecr_uri], check=True)

# ====== 2. DEPLOY TO SAGEMAKER ======
ecr_uri_base = ecr_uri.replace("_loadtest", "")
model_name = IMAGE_NAME.replace("_", "-")
execution_role_arn = "arn:aws:iam::354918369325:role/AmazonSageMaker-ExecutionRole"
env_vars = obj_models[OBJECT_DETECTION_MODEL].copy()
env_vars.update({"DEVICE": device})

sys.path.append("..")
from aws_utils import SageMakerClient

c = SageMakerClient(region_name=AWS_REGION)
c.sagemaker_inference_deploy_pipeline(
    model_name, ecr_uri_base, execution_role_arn, env_vars, CURRENT_INSTANCE_TYPE
)

print(f"Deployment finished: model {model_name}, endpoint should be live soon.")