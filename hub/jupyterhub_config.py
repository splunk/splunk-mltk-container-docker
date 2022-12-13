# Configuration file for jupyterhub
c = get_config()
# Use DummyAuthenticator and SimpleSpawner
#c.JupyterHub.spawner_class = "simple"
#c.JupyterHub.authenticator_class = "dummy"
#c.DockerSpawner.image = 'phdrieger/dsdl-juypterhub:1.0.0'
#c.JupyterHub.spawner_class = 'dockerspawner.DockerSpawner'

# dummy for testing. Don't use this in production!
c.JupyterHub.authenticator_class = "dummy"

# launch with docker
#c.JupyterHub.spawner_class = "docker"
c.JupyterHub.spawner_class = 'dockerspawner.DockerSpawner'

# we need the hub to listen on all ips when it is in a container
c.JupyterHub.hub_ip = '0.0.0.0'
# the hostname/ip that should be used to connect to the hub
# this is usually the hub container's name
c.JupyterHub.hub_connect_ip = 'jupyterhub'

# pick a docker image. This should have the same version of jupyterhub
# in it as our Hub.
#c.DockerSpawner.image = 'phdrieger/dsdl-juypterhub:1.0.0'
c.DockerSpawner.image = 'jupyter/all-spark-notebook:spark-3.2.1'

# tell the user containers to connect to our docker network
c.DockerSpawner.network_name = 'jupyterhub'

# delete containers when the stop
c.DockerSpawner.remove = True