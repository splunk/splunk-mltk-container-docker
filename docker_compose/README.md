# Splunk App for Data Science and Deep Learning (DSDL) - Docker compose files for additional services

Since DSDL version 5.2.0 you can run additional container based services that allow for extended functionalities of DSDL such as utilizing large language models (LLMs) including retrieval augmented generation (RAG) operations in conjunction to vector or graph databases.

## Prerequisits

Please ensure that the following prerequists are met to make use of additional services with DSDL:
- DSDL 5.2.0 or later is correctly set up in your environment and connected to a docker host.
- In DSDL's setup page the docker network name `dsenv-network` is specified. This mandatory so that DSDL containers can communicate on the same Docker network as defined in the docker compose files in this folder.
- Start the desired additional services with docker compose on the same docker host
- Start at least one DSDL dev or prod container based on the image `Red Hat LLM RAG CPU (5.2.0)`. Start this container after you run docker compose to ensure the docker network dsenv-network is active


## Start additional services

You can use the following docker compose commands to start selected additional services.

### Start Milvus Vector Database service
docker compose -f milvus-docker-compose.yml up --detach

### Start Ollama LLM service
docker compose -f ollama-docker-compose.yml up --detach

### Start Ollama LLM service (GPU)
docker compose -f ollama-docker-compose-gpu.yml up --detach

### Start Neo4j graph database service
docker compose -f neo4j-docker-compose.yml up --detach

## Stop additional services

You can use the following Docker compost commands to stop selected additional services.

### Stop Milvus Vector Database service
docker compose -f milvus-docker-compose.yml down

### Stop Ollama LLM service
docker compose -f ollama-docker-compose.yml down

### Stop Ollama LLM service (GPU)
docker compose -f ollama-docker-compose-gpu.yml down

### Stop Neo4j graph database service
docker compose -f neo4j-docker-compose.yml down
