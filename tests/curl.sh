curl  --verbose --trace-time -H "Content-Type: image/jpeg" --data-binary "@tests/7214.jpg" \
 https://iadgca1vqd.execute-api.eu-central-1.amazonaws.com/dev/predict

curl  --verbose --trace-time -H "Content-Type: image/jpeg" --data-binary "@tests/plant.jpg" \
 https://iadgca1vqd.execute-api.eu-central-1.amazonaws.com/dev/predict