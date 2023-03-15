# DSDL and Jupyterhub Integration
This documentation describes how to set up Splunk DSDL and Jupyterhub Integration.

## Prerequisites

- Jupyterhub set up on a Kubernetes Environment on a Bare Metal Host.
- [Splunk App for Data Science and Deep Learning](https://splunkbase.splunk.com/app/4607)(DSDL) installed. Please download the latest version(v5.0.1) from [here](https://github.com/splunk/splunk-mltk-container/blob/jupyterhub-with-kubernetes/mltk-container_v5.0.1.tgz).
- Make sure Kubernets cluser and Splunk instance running on the same host.


## Set up Jupyterhub with Kubernetes
This section describes how to set up Jupyterhub on Kubernetes on a Bare Metal Host with MicroK8s.

### Host Infomation:
|  **Instance Type** | c5.9xlarge (vcpu 36 / 72 Gib ram)  |
|---|---|
| **Storage**  | 200 GiB |
| OS  | Ubuntu 22.04 LTS  |


### 1. Set up Kubernetes Cluser on a Bare Metal Host with MicroK8s.
Please follow the steps [here](https://z2jh.jupyter.org/en/stable/kubernetes/other-infrastructure/step-zero-microk8s.html) to set up your own Kubernetes Cluster.


1. Login as root user
    ```
    sudo su
    ```

2. Install microk8s
    ```
    snap install microk8s
    ```

3. Enable the necessary MicroK8s Add ons
    ```
    microk8s enable dns
    microk8s enable helm3
    ```

4. Configure networking
    ```
    microk8s enable metallb:<IP_RANGE>
    ```
    ***Note***: Make sure the IP_RANGE cover your host's IP.
    e.g. Host IP: `10.202.21.250`
    ```
    microk8s enable metallb:10.202.21.0-10.202.21.254
    ```

5. Configure Storage
    ```
    systemctl enable iscsid.service
    microk8s enable community
    microk8s enable openebs
    ```

    Choose a directory on your host where you want to store data from your cluster. The path can be on the system disk or a separate disk. Create a YAML file called `local-storage-dir.yaml` with the following contents:
    ```
    ## local-storage-dir.yaml
    apiVersion: storage.k8s.io/v1
    kind: StorageClass
    metadata:
      name: local-storage-dir
      annotations:
        storageclass.kubernetes.io/is-default-class: "true"
        openebs.io/cas-type: local
        cas.openebs.io/config: |
          - name: StorageType
            value: hostpath
          - name: BasePath
            value: /path/to/your/storage
    provisioner: openebs.io/local
    reclaimPolicy: Delete
    volumeBindingMode: WaitForFirstConsumer
    ```

    Apply the customized StorageClass resource to your cluster:
    ```
    microk8s.kubectl apply -f local-storage-dir.yaml
    ```

6. Verify
    ```
    microk8s.kubectl get storageclass
    
    # output
    local-storage-dir (default)    openebs.io/local       Delete          WaitForFirstConsumer   false                  11h
    ```


### 2. Set up `helm`
Please follow the steps [here](https://z2jh.jupyter.org/en/stable/kubernetes/setup-helm.html) to set up helm.

1. Installation
    ```
    curl https://raw.githubusercontent.com/helm/helm/HEAD/scripts/get-helm-3 | bash
    ```

2. Verify
    ```
    helm version
    
    # output
    version.BuildInfo{Version:"v3.11.2", GitCommit:"912ebc1cd10d38d340f048efaf0abda047c3468e", GitTreeState:"clean", GoVersion:"go1.18.10"}
    ```


### 3. Install JupyterHub
Please follow the steps [here](https://z2jh.jupyter.org/en/stable/jupyterhub/installation.html) to install JupyterHub in the Kubernetes cluster using the JupyterHub Helm chart.

1. Create default `config.yaml`. Create a config.yaml file with some helpful comments.
    ```
    # This file can update the JupyterHub Helm chart's default configuration values.
    #
    # For reference see the configuration reference and default values, but make
    # sure to refer to the Helm chart version of interest to you!
    #
    # Introduction to YAML:     https://www.youtube.com/watch?v=cdLNKUoMc6c
    # Chart config reference:   https://zero-to-jupyterhub.readthedocs.io/en/stable/resources/reference.html
    # Chart default values:     https://github.com/jupyterhub/zero-to-jupyterhub-k8s/blob/HEAD/jupyterhub/values.yaml
    # Available chart versions: https://jupyterhub.github.io/helm-chart/
    #
    ```

2. Install JupyterHub
    ```
    helm repo add jupyterhub https://jupyterhub.github.io/helm-chart/
    helm repo update
    
    # install the chart configured by your config.yaml
    
    helm upgrade --cleanup-on-fail   --install jupyterhub jupyterhub/jupyterhub   --namespace default   --create-namespace   --values config.yaml

    # Successful output
    NAME: jupyterhub
    LAST DEPLOYED: Wed Mar 15 14:51:16 2023
    NAMESPACE: default
    STATUS: deployed
    REVISION: 2
    TEST SUITE: None
    NOTES:
    .      __                          __                  __  __          __
          / / __  __  ____    __  __  / /_  ___    _____  / / / / __  __  / /_
    __  / / / / / / / __ \  / / / / / __/ / _ \  / ___/ / /_/ / / / / / / __ \
    / /_/ / / /_/ / / /_/ / / /_/ / / /_  /  __/ / /    / __  / / /_/ / / /_/ /
    \____/  \__,_/ / .___/  \__, /  \__/  \___/ /_/    /_/ /_/  \__,_/ /_.___/
                  /_/      /____/

          You have successfully installed the official JupyterHub Helm chart!
          ...
    ```

    **Troubleshooting**

    If you get the following error message:

    `Error: Kubernetes cluster unreachable`
  

    Please try the following command
    ```
    microk8s.kubectl config view --raw > $HOME/.kube/config
    ```

    Then re-run
    ```
    helm upgrade --cleanup-on-fail   --install jupyterhub jupyterhub/jupyterhub   --namespace default   --create-namespace   --values config.yaml
    ```

    (Resolution link: https://github.com/k3s-io/k3s/issues/1126)
    

3. Verify.

  - Check if the Jupyterhub login page is available at `http://<INSTANCE_IP>/hub/login`. e.g. `http://10.202.21.250/hub/login`. You can use any random username and password to log in.


## Customize Jupyterhub
By making changes to your `config.yaml`, you can customize Jupyterhub deployment.

This section describes how to configure your Jupyterhub to 1) support multiple users, 2) use DSDL image and 3) support persistent shared volume.

**Please download the [jupyterhub_setup/config.yaml](https://github.com/splunk/splunk-mltk-container-docker/blob/jupyterhub-with-kubernetes/jupyterhub_setup/config.yaml) as reference.**

### 1. Support multiple users
- Please add the following block to your `config.yaml`, and replace the <USERNAME> and <PASSWORD> with your desired values.
    ```
    hub:
      config:
        Authenticator:
          admin_users:
            - <USERNAME1>
            - <USERNAME2>
          allowed_users:
            - <USERNAME3>
            - <USERNAME4>
        DummyAuthenticator:
          password: <PASSWORD>
        JupyterHub:
          authenticator_class: dummy
    ```
    
- Apply the change
    ```
    helm upgrade --cleanup-on-fail   --install jupyterhub jupyterhub/jupyterhub   --namespace default   --create-namespace   --values config.yaml
    ```

- Verify

  Navigate to the login page, now you can only use defined users and passwords to log in. 

### 2. Use DSDL image
The main feature of this integration is configuring Jupyterhub to use the DSDL image, which allows users to use all the prebuild ML models inside DSDL. 

- Please add the Please add the following block to your `config.yaml`.
    ```
    singleuser:
      # Define the default image
      image:
        name: jupyter/minimal-notebook
        tag: 2343e33dec46
      profileList:
        - display_name: "Minimal environment"
          description: "To avoid too much bells and whistles: Python."
          default: true
        - display_name: "Gold Image CPU - Jupyterhub 5.0.0"
          description: "Splunk DLDS Gold Image CPU Jupyterhub"
          kubespawner_override:
            image: yling454/mltk-container-golden-image-cpu-jupyterhub:5.1.7
            http_timeout: 3600
          cmd: null
    ```

- Apply the change
    ```
    helm upgrade --cleanup-on-fail   --install jupyterhub jupyterhub/jupyterhub   --namespace default   --create-namespace   --values config.yaml
    ```

- Verify
    It takes a while to fetch the DSDL image. Once it's finished, you can see two server options when you log in.
     - Minimal environment
     - Gold Image CPU - Jupyterhub 5.0.0

    Select `Gold Image CPU - Jupyterhub 5.0.0`, and the server should be up and running.

### 3. Support Persistent Shared Volume
In order to support persistent shared volume, we need to add an additional storage volume in your deployment and then make it become shareable for each user.

Please follow the steps below to create `PersistentVolume` and `PersistentVolumeClaim` outside of Jupyterhub.

#### 1. Create `PersistentVolume`
- Choose a directory on your host where you want to store data from your cluster. The path can be on the system disk or a separate disk. Create a YAML file called `shared-storage-dir.yaml` with the following contents:
  ```
  apiVersion: storage.k8s.io/v1
  kind: StorageClass
  metadata:
    name: shared-storage-dir
    annotations:
      storageclass.kubernetes.io/is-default-class: "true"
      openebs.io/cas-type: local
      cas.openebs.io/config: |
        - name: StorageType
          value: hostpath
        - name: BasePath
          value: <PATH/TO/YOUR/STORAGE>
  provisioner: openebs.io/local
  reclaimPolicy: Delete
  volumeBindingMode: WaitForFirstConsumer
  ```

- Run `microk8s.kubectl apply -f shared-storage-dir.yaml`

- Verify 
    ```
    microk8s.kubectl get storageclass
    
    # output
    shared-storage-dir (default)   openebs.io/local       Delete          WaitForFirstConsumer   false                  15s
    ```

#### 2. Create `PersistentVolumeClaim`
- Create a YAML file called `pvc.yaml` with the following contents:
  ```
  apiVersion: v1
  kind: PersistentVolumeClaim
  metadata:
    name: jupyterhub-shared-volume
  spec:
    accessModes:
      - ReadWriteOnce
    volumeMode: Filesystem
    resources:
      requests:
        storage: 8Gi
    storageClassName: shared-storage-dir
  ```
  
- Run `microk8s.kubectl apply -f pvc.yaml`

- Verify
    ```
    microk8s.kubectl get pvc
    
    # output
    jupyterhub-shared-volume   Pending                                                                        shared-storage-dir   9s
    ```

#### 3. Mount the `PersistentVolumeClaim` to user pods.
- Please add the following block to your `config.yaml` under `singleuser` section.
    ```
      storage:
        extraVolumes:
          - name: shared-storage-dir
            persistentVolumeClaim:
              claimName: jupyterhub-shared-volume
        extraVolumeMounts:
          - name: shared-storage-dir
            mountPath: /srv   # Destination DIR inside container
    ```

- Apply the change to your JupyterHub deployment
Run the following command
    ```
    helm upgrade --cleanup-on-fail   --install jupyterhub jupyterhub/jupyterhub   --namespace default   --create-namespace   --values config.yaml
    ```

- Verify
  - Log in with user1 and create a new file under the `shared` folder.

  - Verify `jupyterhub-shared-volume`'s status 
      ```
      microk8s.kubectl get pvc
    
      # output
      jupyterhub-shared-volume   Bound    pvc-c6b1ddf6-e7ae-4cfd-94f3-14746e1cd9d0   8Gi        RWO            shared-storage-dir   4m59s
      ```
  - Log in with user2, and you would see the file that you just created using user1 still shows up in user2's `shared` folder.


## Set up DSDL and Jupyterhub Connection
Make sure you download the latest DSDL(v5.0.1) from [here](https://github.com/splunk/splunk-mltk-container/blob/jupyterhub-with-kubernetes/mltk-container_v5.0.1.tgz).

1. Open the Web UI of your Splunk Instanc.
2. Access the DSDL App from the list of applications.
3. Click on `Configuration` > `Set up`, and check the checkbox `Yes` to proceed with the setup process.
4. Under the `Kubernetes` section, enter the following details.
    - **Authentication Mode**: Select `User Token`.
    - **Cluster Base URL**: Enter your cluser base URL. (You can get it from `$HOME/.kube/config`).
    - **Service Type**: Select `Node Port`.
    - **Node Port Internal Hostname**: Enter `127.0.0.1`
    - **Node Port External Hostname**: Enter the hostname/Ip of the instance that your Splunk and Jupyterhub are running on.
    - **Namespace**: Enter the Kubernetes namespace that is configured for your Jupterhub. For example `default`.
    - **Storage Class**: Enter `microk8s-hostpath`.
    - **Image Pull Secrets**: Enter `None`.
    - **In Cluster Mode**: Select `No`.
    - **Jupyterhub External IP**: Enter your Jupyterhub External IP. It should be the IP of the instance that your Jupyterhub is running on.

5. Scroll down and click `Test & Save` button. You would see a pop-up window with a successful message: **Successfully established connection to container environment.**

6. Click `OK`, Click `Container Management` on the redirected page.

7. Click the `START JUPYTER HUB` button under the `Deployment Jupyterhub` panel. 

8. You would see the `JUPYTER HUB RUNNING` status and the `OPEN JUPYTER LAB` button. Click the `OPEN JUPYTER LAB` button, you will be redirected to the JupyterHub login page.