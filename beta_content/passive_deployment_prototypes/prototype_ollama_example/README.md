Scripts in this prototype deployement deploy a DSDL dev container, ollama container and milvus.
You may choose to deploye one, several or all depending on your needs and which example you are tryig to run.

## Case 1: I am deploying supporting infrastructure (ollama/milvus) into a docker environment which will be used along-side an existing DSDL dev container:

1. Clone/copy this repo onto the docker host connected to DSDL.
2. Ensure Docker and Docker Compose are installed.
    - https://docs.docker.com/engine/install/
    - https://docs.docker.com/compose/install/
3. Read the infra-only compose file: [infra-rag-docker-compose.yml](<compose_files/infra-rag-docker-compose.yml>) and ensure you understand what you are deploying, you may need to adjust settings or remove the sections related to GPU resources for ollama (if you do not have an nvidia GPU enabled environment)
4. Run ```./dsdl_deploy_infra.sh``` to deploy the supporting infrastructure into your docker environment.

5. Adjust dev containter network if nessesary: 

    - For an existing DSDL container to comminicate with ollama and milvus they must be configured to be deployed in the same network, and this network must not be the default network. By default the compose files in this prototype deploy containers to a network named ```dsenv-network```.
    
        If you have previously deployed a dsdl container and are not using compose to deploy it here you may need to run:  
        ``` 
        docker network connect dsenv-network <container-name> 
        ``` 
        to move the container to the correct network.

## Case 2: I am deploying a statically deployed DSDL dev container and all supporting infra (ollama/milvus)

Static deployment of the DSDL dev container is useful in environments where DSDL cannot dynamically deploy containers via the docker management API, perhaps because of security or communication restrictions.

Deploy a full static environment to a docker host:
1. Install docker https://docs.docker.com/engine/install/ (Docker desktop not required)
2. Install docker-compose https://docs.docker.com/compose/install/
3. Read dsdl_deploy_full.sh to ensure you are happy with what we're configuring. 
4. Run dsdl_deploy_full.sh as root or with sufficient privs
5. Deploy the configuration created by this script in ```./conf_files/*```.
    - Stop Splunk on the search head.
    - Copy the contents of ```./conf_files/``` from the docker host into ```splunk/etc/apps/mltk-container/local/``` on the serch head.
    - Start Splunk on the search head.
    - Check the dev container appears with details in the Containers page in the DSDL UI.
    - Run a simple example to validate that fit/apply works.

Note:
- This template uses the default container configuration which uses an insecure certificate, this may or may not be fine for your test environment but should not be used in production.
- This template only deploys a dev container, future guidance will detail how to configure dev and prod containers using dynamic templates and deploy them statically in docker and in K8s using HELM.
- If you stop or restart the docker containers in the environment you will need to update the config on the splunk search head.

Routing requirements:
- Splunk search head must be able to access port 5000 or appropriately mapped ingress on the docker host for API communication
- Users must be able to access appropriate ports or ingress on the docker host to access web applciations hosted in the container
    - Jupyter:8888 (Required)
    - Mlflow:6060 (Optional)
    - Spark:4040 (Optional)
    - Tensorboard:6006 (Optional)