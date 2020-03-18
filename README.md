# onnx-inference-lambda

This project demnonstrates how to run onnx-runtime in AWS Lambda.

You can deploy with 'sls deploy'

Post an image to the created endpoint via curl:
curl  --verbose --trace-time -H "Content-Type: image/jpeg" --data-binary "@tests/plant.jpg" $GENERATED_LAMBDA_ENDPOINT
