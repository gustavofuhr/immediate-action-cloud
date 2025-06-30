
import json
import time

import boto3


class SageMakerClient:
    def __init__(self, region_name):
        self.client = boto3.client('sagemaker', region_name=region_name)


    def _delete_model_if_exists(self, model_name):
        try:
            # Check if the model exists by describing it
            self.client.describe_model(ModelName=model_name)
            print(f"Model '{model_name}' exists. Deleting it...")
            self.client.delete_model(ModelName=model_name)
            print(f"Model '{model_name}' deleted successfully.")
        except self.client.exceptions.ClientError as e:
            if "Could not find model" in str(e):
                print(f"Model '{model_name}' does not exist. Proceeding to create it.")
            else:
                raise  # Re-raise other exceptions

    def create_sagemaker_model(self, model_name, ecr_image_uri, execution_role_arn, env_variables, overwrite=True):
        """
        Create a SageMaker Model with the specified parameters.

        :param model_name: The name of the model to create.
        :param ecr_image_uri: The ECR URI of the Docker image to use for the model.
        :param execution_role_arn: The ARN of the IAM role to use for the model.
        :param env_variables: A dictionary of environment variables to set in the model's container.
        """
        if overwrite:
            self._delete_model_if_exists(model_name)

        print("Creating SageMaker Model...")
        model_response = self.client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': ecr_image_uri,
                'Environment': env_variables
            },
            ExecutionRoleArn=execution_role_arn
        )
        print(f"Model created: {model_response['ModelArn']}")

    def _delete_endpoint_config_if_exists(self, endpoint_config_name):
        try:
            self.client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
            self.client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
        except self.client.exceptions.ClientError as e:
            if not "Could not find endpoint configuration" in str(e):
                raise  # Re-raise other exceptions

    def create_endpoint_config(self, endpoint_config_name, model_name, instance_type, overwrite=True):
        if overwrite:
            self._delete_endpoint_config_if_exists(endpoint_config_name)

        print("Creating Endpoint Configuration...")
        endpoint_config_response = self.client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': instance_type
                }
            ]
        )
        print(f"Endpoint configuration created: {endpoint_config_response['EndpointConfigArn']}")

    def create_endpoint(self, endpoint_name, endpoint_config_name, wait_for_deploy=False):
        print("Deploying Endpoint...")
        endpoint_response = self.client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        print(f"Endpoint deployment started: {endpoint_response['EndpointArn']}")

        if wait_for_deploy:
            print("Waiting for endpoint to be in service...")
            self.client.get_waiter('endpoint_in_service').wait(EndpointName=endpoint_name)
            print(f"Endpoint {endpoint_name} is now in service!")


    def sagemaker_inference_deploy_pipeline(self, model_name, ecr_image_uri, execution_role_arn, env_variables, instance_type, wait_for_deploy=False):
        self.create_sagemaker_model(model_name, ecr_image_uri, execution_role_arn, env_variables)
        endpoint_config_name = model_name + "-config"
        self.create_endpoint_config(endpoint_config_name, model_name, instance_type)
        endpoint_name = model_name + "-endpoint"
        self.create_endpoint(endpoint_name, endpoint_config_name, wait_for_deploy)


class AWSBatchJobClient:
    
    def __init__(self, region_name):
        self.client = boto3.client('batch', region_name=region_name)

    def compute_environment_exists(self, compute_environment_name):
        response = self.client.describe_compute_environments(
            computeEnvironments=[compute_environment_name])
        return len(response["computeEnvironments"]) > 0

    def get_compute_environment_status(self, compute_environment_name):
        response = self.client.describe_compute_environments(
            computeEnvironments=[compute_environment_name])
        return response["computeEnvironments"][0]["status"]
    
    def get_compute_environment_state(self, compute_environment_name):
        response = self.client.describe_compute_environments(
            computeEnvironments=[compute_environment_name])
        return response["computeEnvironments"][0]["state"]
    
    def is_compute_environment_updating(self, compute_environment_name):
        return self.get_compute_environment_status(compute_environment_name) == "UPDATING"

    def delete_compute_environment_if_exists(self, compute_environment_name):
        if not self.compute_environment_exists(compute_environment_name):
            return
        
        try:
            print(f"Disabling compute environment: {compute_environment_name}")
            self.client.update_compute_environment(computeEnvironment=compute_environment_name, state='DISABLED')
        
            # Wait for the compute environment to reach the DISABLED state
            while self.is_compute_environment_updating(compute_environment_name):
                print(f"Waiting for compute environment {compute_environment_name} to be disabled...")
                time.sleep(10)
        
            print(f"Deleting compute environment: {compute_environment_name}")
            try:
                self.client.delete_compute_environment(computeEnvironment=compute_environment_name)
                while self.compute_environment_exists(compute_environment_name):
                    print(f"Waiting for compute environment {compute_environment_name} to be deleted...")
                    time.sleep(10)
                    
                print(f"Compute environment {compute_environment_name} deleted successfully.")
            except self.client.exceptions.ClientException as e:
                print(f"Error deleting compute environment {compute_environment_name}: {e}")
        except self.client.exceptions.ClientError as e:
            if not "ComputeEnvironment does not exist" in str(e):
                raise

    def create_compute_environment(self, compute_environment_name, instance_role_arn, instance_type, 
                                        instance_subnets, instance_security_groups, 
                                        min_vcpus = 0, max_vcpus = 16, overwrite=True):
   
        if instance_type not in ["g4dn.xlarge", "c5.large"]:
            raise ValueError("Only 'g4dn.xlarge' and 'c5.large' instance types are supported now.")
        
        if overwrite:
            self.delete_compute_environment_if_exists(compute_environment_name)

        compute_resources = {
            'type': 'EC2',
            'minvCpus': min_vcpus,
            'maxvCpus': max_vcpus,
            'instanceTypes': [instance_type],
            "allocationStrategy": "BEST_FIT_PROGRESSIVE",
            "bidPercentage": 60,
            'subnets': instance_subnets,
            'securityGroupIds': instance_security_groups,
            'instanceRole': instance_role_arn,
        }
        if "g4dn" in instance_type: # GPU instance type
            compute_resources.update({
                "ec2Configuration": [{
                    "imageType": "ECS_AL2_NVIDIA"
                }]
            })

        print(f"Creating compute environment '{compute_environment_name}'...")
        create_env_response = self.client.create_compute_environment(
            computeEnvironmentName=compute_environment_name,
            type='MANAGED',
            computeResources=compute_resources,
            # serviceRole=execution_role_arn
        )
        while True:
            state = self.get_compute_environment_state(compute_environment_name)
            status = self.get_compute_environment_status(compute_environment_name)
            print(f"Compute environment {compute_environment_name} state: {state}, status: {status}")
            if status != "CREATING" and state == 'ENABLED':
                break
            print(f"Waiting for compute environment {compute_environment_name} to be created...")
            time.sleep(10)
        print("Compute Environment Created:", json.dumps(create_env_response, indent=2))

    def job_queue_exists(self, job_queue_name):
        response = self.client.describe_job_queues(
            jobQueues=[job_queue_name])
        return len(response["jobQueues"]) > 0

    def delete_job_queue_if_exists(self, job_queue_name):
        if not self.job_queue_exists(job_queue_name):
            return
        
        try:
            print(f"Disabling job queue: {job_queue_name}")
            self.client.update_job_queue(jobQueue=job_queue_name, state='DISABLED')
        
            # Wait for the job queue to reach the DISABLED state
            while True:
                response = self.client.describe_job_queues(
                    jobQueues=[job_queue_name]
                )
                state = response['jobQueues'][0]['state']
                status = response['jobQueues'][0]['status']
                print(f"Job queue {job_queue_name} state: {state}, status: {status}")
                
                if status != "UPDATING" and state == 'DISABLED':
                    break
                
                print(f"Waiting for job queue {job_queue_name} to be disabled...")
                time.sleep(10)
        
            print(f"Deleting job queue: {job_queue_name}")
            try:
                self.client.delete_job_queue(jobQueue=job_queue_name)
                while self.job_queue_exists(job_queue_name):
                    print(f"Waiting for job queue {job_queue_name} to be deleted...")
                    time.sleep(10)
                print(f"Job queue {job_queue_name} deleted successfully.")
            except self.client.exceptions.ClientException as e:
                print(f"Error deleting job queue {job_queue_name}: {e}")
        except self.client.exceptions.ClientError as e:
            if not "JobQueue does not exist" in str(e):
                raise

    def create_job_queue(self, job_queue_name, compute_environment_name, priority=1, overwrite=True):
        if overwrite:
            self.delete_job_queue_if_exists(job_queue_name)

        print(f"Creating job queue '{job_queue_name}'...")
        create_queue_response = self.client.create_job_queue(
            jobQueueName=job_queue_name,
            priority=priority,
            computeEnvironmentOrder=[
                {
                    'order': 1,
                    'computeEnvironment': compute_environment_name
                }
            ]
        )
        print("Job Queue Created:", json.dumps(create_queue_response, indent=2))

    def job_definition_exists(self, job_definition_name):
        response = self.client.describe_job_definitions(
            jobDefinitions=[job_definition_name])
        return len(response["jobDefinitions"]) > 0
    
    def delete_job_definition_if_exists(self, job_definition_name):
        if not self.job_definition_exists(job_definition_name):
            return
        
        try:
            print(f"Deleting job definition: {job_definition_name}")
            self.client.deregister_job_definition(jobDefinition=job_definition_name)
            while self.job_definition_exists(job_definition_name):
                print(f"Waiting for job definition {job_definition_name} to be deleted...")
                time.sleep(10)
            print(f"Job definition {job_definition_name} deleted successfully.")
        except self.client.exceptions.ClientError as e:
            if not "JobDefinition does not exist" in str(e):
                raise

    def create_job_definition(self, job_definition_name, container_properties, overwrite=True):
        if overwrite:
            self.delete_job_definition_if_exists(job_definition_name)

        print(f"Creating job definition '{job_definition_name}'...")
        create_job_response = self.client.register_job_definition(
            jobDefinitionName=job_definition_name,
            type='container',
            containerProperties=container_properties
        )
        print("Job Definition Created:", json.dumps(create_job_response, indent=2))

    def submit_job(self, job_name, job_queue, job_definition, containerOverrides):
        print(f"Submitting job '{job_name}' to queue '{job_queue}'...")
        submit_job_response = self.client.submit_job(
            jobName=job_name,
            jobQueue=job_queue,
            jobDefinition=job_definition,
            containerOverrides=containerOverrides
        )
        print("Job Submitted:", json.dumps(submit_job_response, indent=2))