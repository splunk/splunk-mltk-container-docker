# Start from base image: basic linux or nvidia-cuda
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG TAG

# Setup Anconda Base
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
RUN apt-get update && apt-get install -y \
	wget \
	vim \
	bzip2
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Install basic frameworks
RUN conda install -n base nb_conda waitress nodejs datashader tensorflow-gpu pytorch gensim dask-ml 

# Setup jupyter lab extensions
RUN conda install -n base -c conda-forge dask-labextension
RUN jupyter labextension install dask-labextension
RUN jupyter serverextension enable --py --sys-prefix dask_labextension --user

RUN conda install -n base -c conda-forge jupyterlab-nvdashboard
RUN jupyter labextension install jupyterlab-nvdashboard

RUN pip install jupyter-tensorboard
RUN jupyter labextension install jupyterlab_tensorboard
RUN jupyter serverextension enable --py --sys-prefix jupyter_tensorboard
RUN jupyter tensorboard enable --user 

RUN jupyter lab clean

# Install additional frameworks
RUN conda install -n base -c conda-forge dask-xgboost spacy fbprophet pomegranate shap lime umap-learn tslearn kmodes
RUN python -m spacy download en_core_web_sm

RUN mkdir /dltk
# Define working directory
WORKDIR /srv

# Copy bootstrap entry point script
COPY bootstrap.sh /dltk/
COPY app /dltk/app
COPY notebooks /dltk/notebooks

# Copy jupyter config
COPY config/jupyter_notebook_config.py /dltk/.jupyter/jupyter_notebook_config.py
# Copy jupyter notebook conversion template to export python module
COPY config/jupyter_notebook_conversion.tpl /dltk/.jupyter/jupyter_notebook_conversion.tpl

# Handle user rights
RUN chgrp -R 0 /dltk && \
    chmod -R g=u /dltk
RUN chgrp -R 0 /srv && \
    chmod -R g=u /srv
RUN chmod g+w /etc/passwd
USER 1001

# Install additional user libraries
# RUN pip install ginza

# Expose container port 5000 (MLTK Container Service) and 8888 (Notebook) and 6006 (Tensorboard)
EXPOSE 5000 8888 6006

# Define bootstrap as entry point to start container
ENTRYPOINT ["/dltk/bootstrap.sh"]
