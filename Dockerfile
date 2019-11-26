ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG TAG

# Installing packages
RUN conda install jupyterlab flask h5py tensorboard nb_conda 
RUN conda install -c rapidsai -c nvidia -c conda-forge -c defaults rapids=0.10 python=3.6

# Install NLP libs
RUN if [ ${TAG} = "nlp" ]; then pip install flair spacy nltk gensim && python -m spacy download en_core_web_sm; fi

# Define working directory
WORKDIR /srv

# Copy bootstrap entry point script
COPY bootstrap.sh /root/

# Copy flask app for serving
COPY app ./app
COPY notebooks ./notebooks

# Copy jupyter config
COPY config/jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py
# Copy jupyter notebook conversion template to export python module
COPY config/jupyter_notebook_conversion.tpl /root/.jupyter/jupyter_notebook_conversion.tpl

# Expose container port 5000 (MLTK Container Service) and 8888 (Notebook)
EXPOSE 5000 8888 6006

# Define bootstrap as entry point to start container
# TODO define switch for dev / prod (flask only)
ENTRYPOINT ["/root/bootstrap.sh"]
