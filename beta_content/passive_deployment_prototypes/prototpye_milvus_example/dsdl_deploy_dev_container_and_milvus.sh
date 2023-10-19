#! /bin/bash
api_address=$1
external_address=$2

if [ -z "$api_address" ]; then
  read -p "Endpoint Address - Enter the address of your container/environment/ingress relative to your Splunk Search Head: " api_address
fi

if [ -z "$external_address" ]; then
  read -p "External Address - Enter the address of your container/environment/ingress relative to your users: " external_address
fi

docker compose -f ./compose_files/milvus-docker-compose.yml up --detach

# Timeout
timeout=10
timer=0

# Loop to check if the container is up
while [[ $(docker ps --filter "name=dsdl-dev" --format '{{.Names}}') != dsdl-dev && $timer -lt $timeout ]]; do
  echo "Waiting for dsdl-dev to be up..."
  sleep 1
  ((timer++))
done

# Final check
if [[ $(docker ps --filter "name=dsdl-dev" --format '{{.Names}}') == dsdl-dev ]]; then
  echo "Container dsdl-dev is running."
else
  return 0
fi

dev_id=`docker container inspect dsdl-dev | jq '.[] | select(.Name=="/dsdl-dev") | .Id'`
dev_id=${dev_id//\"/}
echo $dev_id

api_port=5000

cat << EOF > ./conf_files/containers.conf
[__dev__]
api_url = https://$api_address:$api_port
api_url_external = https://$external_address:$api_port
cluster = docker
id = $dev_id
image = static
jupyter_url = https://$external_address:8888
mlflow_url = http://$external_address:6060
mode = DEV
runtime = None
spark_url = http://$external_address:4040
tensorboard_url = http://$external_address:6006
EOF
