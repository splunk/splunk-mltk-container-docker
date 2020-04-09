ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG TAG

RUN conda install -n base nb_conda waitress datashader tensorflow-gpu pytorch gensim dask-ml 
RUN conda install -c conda-forge dask-xgboost spacy nodejs 

RUN conda install -c conda-forge dask-labextension
RUN jupyter labextension install dask-labextension
RUN jupyter serverextension enable --py --sys-prefix dask_labextension

RUN conda install -c conda-forge fbprophet pomegranate shap lime 

## experimental for jupyterlab extensions
#RUN pip install jupyter-tensorboard
#RUN jupyter labextension install jupyterlab_tensorboard
#RUN jupyter serverextension enable jupyter_tensorboard --system
#RUN jupyter tensorboard enable --system

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

# Expose container port 5000 (MLTK Container Service) and 8888 (Notebook)
EXPOSE 5000 8888 6006

RUN chgrp -R 0 /dltk && \
    chmod -R g=u /dltk
RUN chgrp -R 0 /srv && \
    chmod -R g=u /srv
RUN chmod g+w /etc/passwd

USER 1001
# Define bootstrap as entry point to start container
# TODO define switch for dev / prod (flask only)
ENTRYPOINT ["/dltk/bootstrap.sh"]
