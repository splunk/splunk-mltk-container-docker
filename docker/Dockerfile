# syntax=docker/dockerfile:1.10
#
# Splunk DSDL container images: one multi-stage template for every image.
#
# Stages:
#   builder    UBI 9 + CPython RPM + compilers + uv. The ONLY stage that contains
#              compilers, -devel packages and kernel headers. Builds the venvs.
#   rootfs     Exact RPM closure installed into a ubi-micro root (dnf --installroot):
#              no package manager, no PAM/su/ssh, no setuid binaries, no toolchain.
#   venv-*     Python environments installed from the hashed uv locks in deps/<profile>.
#   payload    DSDL app, notebooks, Jupyter config, entrypoint and ESCU models.
#   targets    cpu, llm, agentic, spark, gpu, rapids (built through docker-bake.hcl).
#
# Base images are pinned by digest (tools/update-digests.sh). Rebuild monthly so
# OS and Python fixes reach the published images.

ARG UBI_IMAGE=registry.access.redhat.com/ubi9/ubi@sha256:16e9e43d99e181463513cc84a514859b1abb9460f1c4d9e208da4b4e51cd5cc3
ARG MICRO_IMAGE=registry.access.redhat.com/ubi9/ubi-micro@sha256:5be877a2a280ae36eb9ab80ca4efb28bbf77c57abbba38137cdc37a03b8465e6
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.13.3@sha256:bee0a2a458fd956c5df7b8b4c5fed2257b0af34866e3da7b51db0be3acb375d1
ARG PYTHON_VERSION=3.12

FROM ${UV_IMAGE} AS uv
FROM ${MICRO_IMAGE} AS micro

# ---------------------------------------------------------------------------
# builder: compilers and headers live here and never reach a runtime image
# ---------------------------------------------------------------------------
FROM ${UBI_IMAGE} AS builder
ARG PYTHON_VERSION
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
RUN dnf -y --setopt=install_weak_deps=False --nodocs install \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-devel gcc gcc-c++ make diffutils \
 && dnf clean all && rm -rf /var/cache/dnf
COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_PYTHON=/usr/bin/python${PYTHON_VERSION} \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_PROGRESS=1 \
    UV_PROJECT_ENVIRONMENT=/opt/dltk/venv

# ---------------------------------------------------------------------------
# rootfs: the runtime operating system, assembled from an exact RPM list
# ---------------------------------------------------------------------------
FROM builder AS rootfs
ARG PYTHON_VERSION
ARG ROOTFS_ALLOWLIST_CHECK=1
COPY --from=micro / /rootfs
COPY docker/rootfs/ /policy/
RUN ROOTFS_ALLOWLIST_CHECK=${ROOTFS_ALLOWLIST_CHECK} bash /policy/build-rootfs.sh base ${PYTHON_VERSION}

# JRE for the Spark add-on, recorded as a file delta on top of the runtime rootfs
FROM rootfs AS rootfs-jre
ARG ROOTFS_ALLOWLIST_CHECK=1
RUN ROOTFS_ALLOWLIST_CHECK=${ROOTFS_ALLOWLIST_CHECK} bash /policy/build-rootfs.sh jre

# ---------------------------------------------------------------------------
# Python environments from the uv locks: every package downloaded from an index or URL
# is verified against the sha256 in uv.lock. The local packages (dsdlsupport,
# vendor/timesfm) and pure-Python sdists are built from source in this stage.
# ---------------------------------------------------------------------------
FROM builder AS venv-base
COPY docker/finalize-venv.sh /usr/local/bin/finalize-venv
COPY package-dsdlsupport/ /src/package-dsdlsupport/
COPY vendor/ /src/vendor/

FROM venv-base AS venv-cpu
ARG TARGETARCH
COPY deps/cpu/ /src/deps/cpu/
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-${TARGETARCH} \
    cd /src/deps/cpu && uv sync --locked --no-install-project --no-editable \
        --group dev --group core --group dl --group nlp \
 && finalize-venv

# files that the "spark" group adds on top of the cpu venv
FROM venv-cpu AS venv-spark-delta
ARG TARGETARCH
COPY docker/spark-jars.sh docker/spark-jars-trim.txt docker/spark-jar-overrides.txt /src/spark/
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-${TARGETARCH} \
    touch /tmp/.spark-marker && sleep 1 \
 && cd /src/deps/cpu && uv sync --locked --no-install-project --no-editable \
        --group dev --group core --group dl --group nlp --group spark \
 && bash /src/spark/spark-jars.sh /src/spark \
 && finalize-venv \
 && mkdir /delta \
 && (cd / && find opt/dltk/venv -newer /tmp/.spark-marker \( -type f -o -type l \) -print0 | tar --null -T - -cf - | tar -xf - -C /delta)

FROM venv-base AS venv-llm
ARG TARGETARCH
COPY deps/llm/ /src/deps/llm/
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-${TARGETARCH} \
    cd /src/deps/llm && uv sync --locked --no-install-project --no-editable --all-groups \
 && finalize-venv

FROM venv-base AS venv-gpu
ARG TARGETARCH
COPY deps/gpu-cu12/ /src/deps/gpu-cu12/
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-${TARGETARCH} \
    cd /src/deps/gpu-cu12 && uv sync --locked --no-install-project --no-editable --all-groups \
 && finalize-venv

FROM venv-base AS venv-rapids
ARG TARGETARCH
COPY deps/rapids-cu12/ /src/deps/rapids-cu12/
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-${TARGETARCH} \
    cd /src/deps/rapids-cu12 && uv sync --locked --no-install-project --no-editable --all-groups \
 && finalize-venv

# ---------------------------------------------------------------------------
# payload: identical in every image (one shared layer)
# ---------------------------------------------------------------------------
FROM builder AS payload
COPY app/ /dltk/app/
COPY notebooks/ /dltk/notebooks/
COPY config/jupyter_server_config.py /dltk/.jupyter/jupyter_server_config.py
COPY config/jupyter_notebook_config.py /dltk/.jupyter/jupyter_notebook_config.py
COPY config/jupyter_notebook_template.tpl /dltk/.jupyter/jupyter_notebook_conversion.tpl
COPY config/null.tpl /dltk/.jupyter/null.tpl
COPY escu_models/ /dltk/app/model/data/
COPY docker/entrypoint.py docker/bootstrap_fast.sh /dltk/
RUN find /dltk \( -name .DS_Store -o -name __pycache__ \) -prune -exec rm -rf {} + \
 # ESCU: keep the flat 5.2.x layout and add the per-model folders the notebooks open
 && cd /dltk/app/model/data \
 && for model in *.pt; do name="${model%.pt}"; mkdir -p "$name"; ln -sf "../$model" "$name/$model"; done \
 # model files must be real artefacts, never Git LFS pointer stubs
 && if grep -rlI --include='*' '^version https://git-lfs.github.com/spec/v1' /dltk/app/model/data; then \
        echo "Git LFS pointer file found in model data" >&2; exit 1; fi \
 && sed -i 's/\r$//' /dltk/bootstrap_fast.sh /dltk/entrypoint.py \
 && chmod 0755 /dltk/bootstrap_fast.sh \
 && ln -s bootstrap_fast.sh /dltk/bootstrap_transformers.sh \
 && ln -s bootstrap_fast.sh /dltk/bootstrap_rapids.sh \
 && chmod -R u=rwX,go=rX /dltk \
 # /srv must be group-writable for arbitrary UIDs (gid 0); COPY keeps the mode of
 # copied directories but not of the destination itself, hence the /out wrapper
 && install -d -m 2775 /out/srv

# ---------------------------------------------------------------------------
# runtime targets
# ---------------------------------------------------------------------------
FROM scratch AS runtime
ARG DSDL_VERSION=0.0.0-dev
ARG VCS_REF=unknown
COPY --from=rootfs /rootfs/ /
COPY --link --from=payload /dltk /dltk
COPY --link --from=payload --chown=1001:0 /out/ /
ENV PATH=/opt/dltk/venv/bin:/usr/bin:/bin \
    VIRTUAL_ENV=/opt/dltk/venv \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=Etc/UTC \
    DSDL_VERSION=${DSDL_VERSION} \
    DSDL_APP_MODULE=app.main:app \
    MODE_DEV_PROD=PROD \
    HOME=/srv/.dsdl-home \
    JUPYTER_CONFIG_DIR=/dltk/.jupyter \
    JUPYTER_RUNTIME_DIR=/tmp/jupyter-runtime \
    JUPYTERLAB_SETTINGS_DIR=/srv/.dsdl-home/.jupyter/lab/user-settings \
    JUPYTERLAB_WORKSPACES_DIR=/srv/.dsdl-home/.jupyter/lab/workspaces \
    MPLCONFIGDIR=/tmp/matplotlib \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1
LABEL org.opencontainers.image.source="https://github.com/splunk/splunk-mltk-container-docker" \
      org.opencontainers.image.vendor="Splunk" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${DSDL_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}"
USER 1001:0
WORKDIR /srv
EXPOSE 5000 8888
ENTRYPOINT ["/dltk/bootstrap_fast.sh"]

FROM runtime AS cpu
COPY --link --from=venv-cpu /opt/dltk/venv /opt/dltk/venv
LABEL org.opencontainers.image.title="Splunk DSDL Golden CPU"

FROM cpu AS spark
COPY --link --from=rootfs-jre /delta/ /
COPY --link --from=venv-spark-delta /delta/ /
ENV JAVA_HOME=/usr/lib/jvm/jre-21 \
    SPARK_USER=dltk \
    HADOOP_USER_NAME=dltk
LABEL org.opencontainers.image.title="Splunk DSDL Spark"
EXPOSE 4040

FROM runtime AS llm
COPY --link --from=venv-llm /opt/dltk/venv /opt/dltk/venv
LABEL org.opencontainers.image.title="Splunk DSDL LLM RAG"

FROM llm AS agentic
ENV DSDL_APP_MODULE=app.main_agentic:app
LABEL org.opencontainers.image.title="Splunk DSDL Agentic AI"

FROM runtime AS cuda-runtime
# The CUDA 12.9 user space comes from the nvidia-*-cu12 wheels; libcuda is injected by the
# NVIDIA container toolkit. There is no cuda-compat, so the images rely on CUDA 12
# minor-version compatibility: cuda>=12.2 means driver R535 or newer (the RAPIDS 26.10
# minimum); Blackwell needs R570+. The toolkit checks it with `docker run --gpus`, not in
# CDI mode.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=12.2" \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

FROM cuda-runtime AS gpu
COPY --link --from=venv-gpu /opt/dltk/venv /opt/dltk/venv
LABEL org.opencontainers.image.title="Splunk DSDL Golden GPU"

FROM cuda-runtime AS rapids
COPY --link --from=venv-rapids /opt/dltk/venv /opt/dltk/venv
LABEL org.opencontainers.image.title="Splunk DSDL RAPIDS"
