# Passive Deployment of DSDL with Docker Compose

The scripts and configuration files in this directory are an experimental deployment mechanism for DSDL. In some situations customers may not be able to give access to the docker API or k8s control plane for DSDL to deploy new containers, in this case docker compose (or HELM in the future) may be utilised to create containers and sets of containers for dev and prod models without the need for a dynamic control interface from the DSDL app. This can simplify deployment in a high security environment at the cost of dynamic control of containers from the DSDL app, instead relying on updates to the infrastructure to deploy new containers.

Usage steps to deploy on a docker host:
- Install docker https://docs.docker.com/engine/install/ (Docker desktop not required)
- Install docker-compose https://docs.docker.com/compose/install/
- Install requirements: apt-get install jq
- Read dsdl_deploy_static_dev_container.sh to ensure you are happy with what we're configuring. 
- Run dsdl_deploy_static_dev_container.sh as root or with sufficient privs
- Stop Splunk on the search head
- Copy the contents of ./conf_files/ from the docker host into splunk/etc/apps/mltk-container/local/ on the serch head
- Start Splunk on the search head
- Check the dev container appears with details in the Containers page in the DSDL UI
- Run a simple example to validate that fit/apply works

Note:
- This template uses the default container configuration which uses an insecure certificate, this may or may not be fine for your test environment but should not be used in production.
- This template only deploys a dev container, future guidance will detail how to configure dev and prod containers using dynamic templates and deploy them statically in docker and in K8s using HELM.
- If you stop or restart the environment you will need to update the config on the splunk search head

Routing requirements:
- Splunk search head must be able to access port 5000 or appropriately mapped ingress on the docker host for API communication
- Users must be able to access appropriate ports or ingress on the docker host to access web applciations hosted in the container
    - Jupyter:8888 (Required)
    - Mlflow:6060 (Optional)
    - Spark:4040 (Optional)
    - Tensorboard:6006 (Optional)