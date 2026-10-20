# Splunk App for Data Science and Deep Learning

Splunk App for Data Science and Deep Learning (DSDL) 5.2.5 formerly known as Deep Learning Toolkit for Splunk (DLTK) versions 2.3.0 - 3.9.0 and (DSDL) 5.0.0 - 5.2.5 published on [splunkbase](https://splunkbase.splunk.com/app/4607/).
There is also a prior DLTK version 4.x which was open sourced on [GitHub](https://github.com/splunk/deep-learning-toolkit) but is not actively maintained. Please make use of the latest version 5.2.5

Copyright (C) 2005-2026 Splunk Inc. All rights reserved.  
Author: [Philipp Drieger]()
Contributors: [Josh Cowling](), [Huaibo Zhao](), [Tatsu Murata]() 

# About
This repository contains the collection of resources, scripts, and testing frameworks that are used build and deploy the default container images used by the DSDL app. It can be used to modify, secure, update and change these containers and their build process to meet the needs of your enterprise environment.

## Resources
This repository contains the container endpoint (`./app`), jupyter notebook configuration (`./config`) and examples (`./notebooks`), build scripts and the main Dockerfiles to create the [existing pre-built container images](https://hub.docker.com/u/splunk?page=1&search=mltk-container) for TensorFlow, PyTorch, NLP libraries and many other data science libraries for CPU and GPU.

### Images
From 5.3.0 all DSDL images are built from one multi-stage template (`docker/Dockerfile`) on Red Hat UBI 9 with Python 3.12. Images that only differed in their library mix were merged; the 5.2.x image names and `images.conf` stanzas keep working because the merged images are also published under the old names.

| Build target | Replaces (5.2.x names, still published) | Platforms | Contents |
| --- | --- | --- | --- |
| `cpu` | `golden-cpu`, `golden-cpu-arm` (`golden-cpu:<v>-arm64`), `escu-cpu`, `golden-cpu-transformers` | amd64, arm64 | classic ML, TensorFlow and PyTorch (CPU), transformers, spaCy 3, Japanese tokenizers, ESCU pretrained models |
| `llm` | `ubi-llm-rag` (stanza `llm-rag`) | amd64, arm64 | the `cpu` content plus llama-index, vector stores and LLM provider SDKs |
| `agentic` | `ubi-llm-rag:agentic-ai-<v>` (stanza `agentic-ai`) | amd64, arm64 | the `llm` image serving the agentic API (`app/main_agentic.py`) |
| `gpu` | `golden-gpu`, `golden-gpu-transformers` | amd64 | PyTorch and TensorFlow with CUDA 12 (from wheels), transformers fine-tuning, TimesFM; also runs without a GPU |
| `rapids` | `golden-gpu-rapids` | amd64 | NVIDIA RAPIDS (cuDF, cuML, cuGraph) from pip wheels |
| `spark` | `spark` | amd64, arm64 | the `cpu` image plus a headless Java 21 runtime and PySpark 4.2 |

`images.toml` is the single source of truth for targets, platforms, published names, stanza ids, titles and runtimes. `tools/render.py` generates `docker-bake.json` and `images_conf_files/images.conf` from it.

### How the images are hardened
- **No build tools at runtime.** Compilers, `-devel` packages and kernel headers exist only in the `builder` stage. In 5.2.5 they caused about 75 % of all scanner findings.
- **Minimal OS.** The runtime is an exact list of about 70 UBI RPMs installed into a `ubi-micro` root (`docker/rootfs/`): no package manager, PAM, `su`/`sudo`, ssh client, Node.js, setuid binaries or second Python. The build fails if the RPM set changes (`docker/rootfs/allow.txt`).
- **Locked dependencies.** Every Python package comes from a hashed `uv.lock` (`deps/<profile>/`), resolved for Python 3.12 on amd64 and arm64. PyPI is the only default index; the PyTorch index serves only `torch`.
- **No shared secrets.** No TLS private key is baked into the images (see Certificates). `/etc/passwd` is read-only; arbitrary UIDs (OpenShift) are supported through nss_wrapper.
- **Pinned bases.** Base images are pinned by digest (`tools/update-digests.sh`); rebuild monthly so OS and Python fixes reach the images.

### Building containers
Requirements: Docker with buildx (Docker Desktop or Docker Engine 25+). Building the amd64 images on an arm64 machine works through emulation, but is slow.

```
./build.sh                               # list targets and published names
./build.sh golden-cpu                    # 5.2.x name: builds the cpu target for linux/amd64
./build.sh golden-cpu-arm                # builds the cpu target for linux/arm64
./build.sh cpu local_build/ 1.0.0        # a target name builds all of its platforms
./build.sh all splunk/ 5.3.0             # every target
./build.sh --push all splunk/ 5.3.0      # push with SBOM and provenance attestations
```

Every build writes `images_conf_files/images.conf`; copy it to `<splunk>/etc/apps/mltk-container/local/images.conf` to make the images available in the DSDL app.

### Changing the Python libraries of an image
1. Edit the profile in `deps/<profile>/pyproject.toml` (`cpu`, `llm`, `gpu-cu12` or `rapids-cu12`). Declare top-level intent only; add an upper version cap only with a comment explaining why.
2. Run `tools/lock.sh <profile>` to resolve the hashed lock (`tools/lock.sh --upgrade <profile>` moves everything to the newest allowed versions, `tools/lock.sh --check` verifies that all locks are current).
3. Rebuild, then run the smoke test and the scan (below).

To build your own image, add a profile in `deps/`, a target stage at the end of `docker/Dockerfile` and a `[targets.<name>]` entry in `images.toml`, then run `tools/render.py`.

### Supporting scripts
| Script | Description | Example |
| --- | --- | --- |
| `build.sh` | Build (or push) images from `images.toml` | `./build.sh golden-cpu splunk/ 5.3.0` |
| `tools/lock.sh` | Resolve, check or upgrade the hashed Python locks | `tools/lock.sh --upgrade cpu` |
| `tools/sync-cpu-constraints.py` | Pin the packages the gpu, rapids and llm profiles share with the cpu image to the cpu lock versions (run after changing the cpu lock) | `tools/sync-cpu-constraints.py gpu-cu12 rapids-cu12 llm` |
| `tools/smoke-test.sh` | Headless test of a built image: security invariants, import of every `app/model` module against the 5.2.5 baseline in `tests/expected/`, PROD and DEV start, arbitrary UID with a read-only root filesystem | `tools/smoke-test.sh splunk/mltk-container-golden-cpu:5.3.0 cpu` |
| `tools/scan.sh` | Trivy scan with a release gate (fails on end-of-life OS, fixable CRITICAL findings and secrets); reports in `scan_logs/` | `tools/scan.sh --gate splunk/mltk-container-golden-cpu:5.3.0` |
| `tools/pull.sh` | Pull every published image reference (for example to mirror them for an air-gapped installation) | `tools/pull.sh splunk/ 5.3.0` |
| `tools/update-digests.sh` | Re-pin the base images to their current digests | `tools/update-digests.sh` |
| `tools/update-rootfs-allowlist.sh` | Regenerate `docker/rootfs/allow.txt` after an intended OS package change | `tools/update-rootfs-allowlist.sh` |
| `test_container.sh` | Playwright UI tests against a live Splunk instance (`testing/`) | `./test_container.sh golden-cpu 5.3.0` |

### Certificates
The API (port 5000) and JupyterLab (port 8888) use HTTPS unless `ENABLE_HTTPS=false`. At start the container uses, in this order:
1. the certificate and key given by `API_SSL_CERT` and `API_SSL_KEY` (for example mounted from a secret),
2. `/dltk/.jupyter/dltk.pem` and `/dltk/.jupyter/dltk.key` if a derived image provides them,
3. a self-signed pair generated for this installation into `/srv/.dsdl/tls` (valid for one year, renewed automatically before it expires).

Images up to 5.2.5 contained a private key generated at build time, shared by everybody who pulled the same image. If you rely on those images, provide your own certificate.

### Installing extra Python packages in DEV mode
The image content is read-only. In DEV mode `pip install <package>` installs into `/srv/.dsdl/site-packages` on the persistent volume. That directory is searched after the image's own packages, so it can add packages but not replace the image's versions. For permanent changes, edit the profile and rebuild.

### Run and test your container locally
You can run your container locally, e.g. with `docker run -it --rm --name mltk-container-golden-cpu -p 5000:5000 -p 8888:8888 -p 6006:6006 -e MODE_DEV_PROD=DEV -e JUPYTER_PASSWD=<password> -v mltk-container-data:/srv splunk/mltk-container-golden-cpu:5.3.0`

## Further documentation and usage
Please find further information and documentation on splunkbase: [Download and install the Splunk App for Data Science and Deep Learning](https://splunkbase.splunk.com/app/4607/) and refer to the [official documentation pages](https://docs.splunk.com/Documentation/DSDL) on how to install and use the app.
