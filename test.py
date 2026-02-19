import boto3
import json

# Initialize the Bedrock Runtime client
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'  # Replace with your region
)

# Model ID you're testing
model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# Prepare the request
request_body = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 1024,
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is your model version, training data cutoff date, and what are your key capabilities compared to Claude 3.5 Sonnet?"
                }
            ]
        }
    ]
}

print(f"Model ID being invoked: {model_id}
")

try:
    # Invoke the model
    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(request_body)
    )
    
    # Extract response metadata
    print("RESPONSE METADATA:")
    print(f"HTTP Status Code: {response['ResponseMetadata']['HTTPStatusCode']}")
    print(f"Request ID: {response['ResponseMetadata']['RequestId']}")
    
    # Check HTTP headers
    print("
HTTP HEADERS:")
    if 'HTTPHeaders' in response['ResponseMetadata']:
        for header, value in response['ResponseMetadata']['HTTPHeaders'].items():
            print(f"{header}: {value}")
    
    # Parse response body
    response_body = json.loads(response['body'].read())
    print("
MODEL RESPONSE:")
    print(json.dumps(response_body, indent=2))
    
except Exception as e:
    print(f"Error: {str(e)}")