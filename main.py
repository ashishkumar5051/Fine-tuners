from fastapi import FastAPI
import os
import json
import boto3
from pydantic import BaseModel
from dotenv import load_dotenv

# endpoint_name = "huggingface-pytorch-tgi-inference-2024-03-06-05-50-57-978"
endpoint_name = "jumpstart-dft-hf-llm-mistral-7b-instruct"

app = FastAPI()
load_dotenv()


class Data(BaseModel):
    question: str
    max_new_tokens: int = 50
    temperature: float = 0.3


@app.post("/v1/chat/completions")
def index(data: Data):

    max_new_tokens = data.max_new_tokens if hasattr(data, 'max_new_tokens') else 50
    temperature = data.temperature if hasattr(data, 'temperature') else 0.3

    prompt = {
        "inputs": data.question,
        "parameters": {"max_new_tokens": max_new_tokens, "temperature": temperature},
    }
    response = generate_response(prompt)
    return response


def generate_response(payload):
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("REGION")

    client = boto3.client(
        "runtime.sagemaker",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region,
    )

    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        # InferenceComponentName="huggingface-llm-mistral-7b-instruct-20210304-045144",
        ContentType="application/json",
        Body=json.dumps(payload).encode("utf-8"),
    )
    model_predictions = json.loads(response["Body"].read())
    generated_text = model_predictions[0]["generated_text"]

    return generated_text
