# Splunk App for Data Science and Deep Learning

Splunk App for Data Science and Deep Learning (DSDL) 5.2.5 formerly known as Deep Learning Toolkit for Splunk (DLTK) versions 2.3.0 - 3.9.0 and (DSDL) 5.0.0 - 5.2.5 published on [splunkbase](https://splunkbase.splunk.com/app/4607/).
There is also a prior DLTK version 4.x which was open sourced on [GitHub](https://github.com/splunk/deep-learning-toolkit) but is not actively maintained. Please make use of the latest version 5.2.5

This branch builds the consolidated **5.3.0** container images (see [Images](#images), [Upgrading from 5.2.x](#upgrading-from-52x) and the build report in [BUILD.md](BUILD.md)).

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
| `gpu` | `golden-gpu`, `golden-gpu-transformers` | amd64 | PyTorch 2.13 and TensorFlow 2.21 from wheels on the CUDA 12.9 user space it shares with `rapids`, peft/bitsandbytes/accelerate, TimesFM; also runs without a GPU |
| `rapids` | `golden-gpu-rapids` | amd64 | NVIDIA RAPIDS 26.10 (cuDF, cuML, cuGraph) and PyTorch 2.13 from pip wheels on CUDA 12.9, with pandas 3.0 (a RAPIDS requirement) |
| `spark` | `spark` | amd64, arm64 | the `cpu` image plus a headless Java 21 runtime and PySpark 4.2 |

**GPU support.** The `gpu` and `rapids` images share one CUDA 12.9 user space from the NVIDIA wheels (cuda-toolkit 12.9.1, cuDNN 9.20, NCCL 2.29). This is the CUDA that RAPIDS 26.10 is built with (`rapidsai/base:26.10-cuda12`); in 5.2.5 the three GPU images ran CUDA 12.4, 11.3 and 12.2 base images. No CUDA and no cuda-compat are installed in the OS layer, so the images rely on CUDA minor-version compatibility.

- **Driver:** R535 or newer (`NVIDIA_REQUIRE_CUDA=cuda>=12.2`, the RAPIDS 26.10 minimum), R570 or newer for Blackwell (R575+ for B300). R535 and R570 are end of life; use R580 (LTSB) or a newer supported branch. The NVIDIA container toolkit enforces the minimum with `docker run --gpus`; in CDI mode an older driver fails at CUDA initialisation instead.
- **GPUs:** Turing (sm_75, e.g. T4) to Blackwell. PyTorch 2.13 (`+cu129`, the newest PyTorch with a CUDA 12.9 build; torch 2.14 has CUDA 12.6 and CUDA 13 builds only) ships kernels for sm_75, sm_80, sm_86, sm_90, sm_100 and sm_120. **Maxwell, Pascal and Volta GPUs (e.g. M60, P100, P4, V100) are not supported:** the PyTorch build has no kernels for them, and cuDNN 9.20 supports Turing and newer only, which also applies to TensorFlow's convolutions and RNNs. In 5.2.5, `golden-gpu` (torch cu121) covered sm_50 to sm_90 and `golden-gpu-transformers` (torch cu128) sm_70 to sm_120. In `rapids`, cuDF, cuML and cuGraph still run on Volta; PyTorch in the same image does not.
- **TensorFlow 2.21** (built with CUDA 12.5) uses the same CUDA 12.9 libraries. It has no native Hopper or Blackwell kernels. On Hopper it JIT-compiles its PTX, which may need a driver newer than R535 (not verified). On Blackwell its XLA compiler, which Keras 3 uses on a GPU by default, fails (tensorflow/tensorflow#111958, fixed in TensorFlow 2.22): use PyTorch there. Set `TF_FORCE_GPU_ALLOW_GROWTH=true` when TensorFlow and PyTorch share a GPU in one process. The image imports Triton (used by bitsandbytes and `torch.compile`) before TensorFlow, because importing it after TensorFlow crashes the process; both bundle LLVM (`docker/finalize-venv.sh`).
- **PyTorch 2.13** is one release behind the CPU images (2.14.1). The cu129 build is a Linux-only build outside PyTorch's support matrix and 2.13 gets no patch releases; the hardening in the PyTorch 2.14.0 security notes is not in it. A CUDA 13 line (PyTorch 2.15+, RAPIDS cu13, driver R580+; TensorFlow CPU-only until TensorFlow ships CUDA 13 wheels) is the planned next step.

**Transformer base models.** The bert/T5 base models that the DSDL text classification and summarization assistants and `bert_phishing` load (`app/model/data/classification/...`, `app/model/data/summarization/...`) are not shipped: in 5.2.x they were Git LFS pointer files and never real models either. Place the models on the `/srv` volume under `/srv/app/model/data/`; the container start keeps them. `testing/test_2_transformers.py` needs them.

`images.toml` is the single source of truth for targets, platforms, published names, stanza ids, titles and runtimes. `tools/render.py` generates `docker-bake.json` and `images_conf_files/images.conf` from it.

### How the images are hardened
- **No build tools at runtime.** Compilers, `-devel` packages and kernel headers exist only in the `builder` stage. In 5.2.5 they caused about 75 % of all scanner findings.
- **Minimal OS.** The runtime is an exact list of about 70 UBI RPMs installed into a `ubi-micro` root (`docker/rootfs/`): no package manager, PAM, `su`/`sudo`, ssh client, Node.js, setuid binaries or second Python. The build fails if the RPM set changes (`docker/rootfs/allow.txt`, and `allow-jre.txt` for the Java runtime of the Spark image) or if an installed RPM is missing files.
- **Locked dependencies.** Every Python package is pinned in a `uv.lock` (`deps/<profile>/`), resolved for Python 3.12; the build fails if a lock is out of date. Every downloaded wheel and sdist is checked against its sha256 in the lock. The two local packages (`package-dsdlsupport`, `vendor/timesfm`) and pure-Python sdists are built from source with version-pinned build backends. PyPI is the only default index; the PyTorch index serves only `torch`. The `gpu` image takes `torchvision` from a direct PyTorch URL, because the CUDA 12.9 index publishes it without a hash; the lock records its sha256.
- **No shared secrets.** No TLS private key is baked into the images (see Certificates). `/etc/passwd` is read-only; arbitrary UIDs (OpenShift) are supported through nss_wrapper.
- **Pinned bases.** Base images are pinned by digest (`tools/update-digests.sh`); rebuild monthly so OS and Python fixes reach the images.

### Building containers
Requirements: Docker with buildx (Docker Desktop, or Docker Engine 25+ with the buildx plugin). A local build loads one platform into Docker: a 5.2.x name builds the platform it was published for, a target name the host's platform if the target supports it. `PLATFORM=linux/amd64` (or `linux/arm64`) selects another platform; building a foreign platform needs QEMU (`docker run --privileged --rm tonistiigi/binfmt --install all`) and is slow. `PLATFORM=all` and `--push` build every declared platform and need the containerd image store (Docker Desktop: Settings > General) or a `docker-container` buildx builder (`docker buildx create --use`).

```
./build.sh                               # list targets and published names
./build.sh golden-cpu                    # 5.2.x name: builds the cpu target for linux/amd64
./build.sh golden-cpu-arm                # cpu target for linux/arm64, tagged golden-cpu:<v>-arm64
./build.sh cpu local_build/ 1.0.0        # a target name: the host's platform
PLATFORM=all ./build.sh cpu              # all platforms of the target
./build.sh all splunk/ 5.3.0             # every target
./build.sh --push all splunk/ 5.3.0      # push with SBOM and provenance attestations
```

Every build writes `images_conf_files/images.conf` with the stanzas of the images it built; copy them to `<splunk>/etc/apps/mltk-container/local/images.conf` to make the images available in the DSDL app.

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
The API (port 5000) and JupyterLab (port 8888) use HTTPS unless `ENABLE_HTTPS=false`. At every start the container uses, in this order:
1. the certificate and key given by `API_SSL_CERT` and `API_SSL_KEY` (for example mounted from a secret),
2. `/dltk/.jupyter/dltk.pem` and `/dltk/.jupyter/dltk.key` if a derived image provides them,
3. a self-signed pair generated for this installation into `/srv/.dsdl/tls` (valid for one year; regenerated at container start when it expires within 30 days or does not match its key, so a container that runs for a year needs a restart).

If `API_SSL_CERT`/`API_SSL_KEY` are set but the files do not exist, the container logs a warning and continues with option 2 or 3.

Images up to 5.2.5 contained a private key generated at build time, shared by everybody who pulled the same image. If you rely on those images, provide your own certificate.

### Installing extra Python packages in DEV mode
The image content is read-only. In a running container `pip install <package>` (in a notebook or a Jupyter terminal) installs into `/srv/.dsdl/site-packages` on the persistent volume, and its command line tools land in `/srv/.dsdl/site-packages/bin`, which is on `PATH`. That directory is searched after the image's own packages, so it can add packages but not replace the image's versions. pip installs the package's dependencies into that directory too, even when the image already has them; use `pip install --no-deps <package>` when the image provides the dependencies. Only pure-Python packages and wheels can be installed: the images have no compiler.

For permanent changes, edit the profile and rebuild, or derive an image: `FROM splunk/mltk-container-golden-cpu:5.3.0`, `USER root`, `RUN pip install <package>` (installs into the image's environment), `USER 1001`.

### Upgrading from 5.2.x
The 5.2.x image names and `images.conf` stanzas keep working, but the contents changed:

- **Platform:** all images run Red Hat UBI 9 with Python 3.12 (5.2.5 used Python 3.8 to 3.11 on UBI 9, Debian 11/13 and Ubuntu 20.04/22.04). There is no conda, no R, no Node.js, no compiler, no `curl`/`wget`/`openssl` command line tools, and no ssh client (`git` works over HTTPS only). The runtime user is 1001 with group 0 in every image, including RAPIDS (5.2.5: `rapids:conda`). `HOME` is `/srv/.dsdl-home` on the persistent volume.
- **Major library versions:** TensorFlow 2.21 with Keras 3 (Keras 2 models load through `tf_keras`), PyTorch 2.14 (2.13 in `gpu` and `rapids`, the newest release with a CUDA 12.9 build), transformers 5, scikit-learn 1.9, numpy 2.5, pandas 2.3 (the 5.2.5 `spark` and `agentic-ai` images had pandas 3.0; `rapids` now has pandas 3.0), spaCy 3.8 with `en_core_web_sm`, xgboost 3.4, LangChain 1.x / LangGraph 1.2 / MCP 1.30 in the llm images (llama-index stays on 0.12), Spark 4.2 with Java 21. **Models persisted with 5.2.x (pickles, Keras 2 files) may need re-saving or retraining.**
- **Removed libraries** (not imported by the shipped models and notebooks, or broken in every 5.2.5 image): pymc3/theano-pymc, dask-xgboost (it forced xgboost 0.90), spaCy 2 and spacytextblob 0.1, tf2onnx, PyPDF2 (use pypdf), wandb, pydgraph, torch-geometric, pygod, pytorch-tabular, and in the llm images the deprecated LlamaIndex integrations (llama-index meta package, `llms-gemini`/`embeddings-gemini`/google-generativeai (replaced by Google GenAI), Vertex, Databricks, vLLM, instructor embeddings, the agent/program/question-gen OpenAI packages, arxiv, the Neo4j graph/vector stores, `llms-huggingface`, `llms-openai-like`, `llms-deepseek`). Each profile in `deps/` documents the reason.
- **Spark:** local mode only. The Hive metastore and `enableHiveSupport()`, the embedded Derby database, the Thrift JDBC server, Spark Connect, the Scala/R shells (`spark-shell`, `sparkR`, `spark-sql`, `beeline`) and the R kernel of the 5.2.5 jupyter base image are not included (`docker/spark-jars-trim.txt`). Bundled netty, Jackson and lz4-java jars are replaced by patched releases (`docker/spark-jar-overrides.txt`).
- **GPU:** CUDA 12.9 from wheels, the CUDA of RAPIDS 26.10, in `gpu` and `rapids` (5.2.5: CUDA 12.4, 11.3 and 12.2 base images). Turing to Blackwell for PyTorch and RAPIDS (TensorFlow 2.21: not on Blackwell); Maxwell, Pascal and Volta GPUs (e.g. V100, P100) are no longer supported (see [Images](#images)).
- **Certificates:** no private key is built into the images any more; see [Certificates](#certificates).
- **Accepted residual findings:** pip vendors older urllib3, msgpack and setuptools copies (4 HIGH findings in every image until pip updates them); nltk 3.10 has a HIGH finding without a fix; the llm images keep llama-index-core 0.12 (a HIGH fixed only in 0.13, which needs a code port). See [BUILD.md](BUILD.md) for the per-image results.

### Run and test your container locally
You can run your container locally, e.g. with `docker run -it --rm --name mltk-container-golden-cpu -p 5000:5000 -p 8888:8888 -p 6006:6006 -e MODE_DEV_PROD=DEV -e JUPYTER_PASSWD=<password> -v mltk-container-data:/srv splunk/mltk-container-golden-cpu:5.3.0`

**Apple silicon (Docker Desktop on a Mac):** use the arm64 images there: `golden-cpu-arm` (`golden-cpu:<v>-arm64`), and `llm-rag`, `agentic-ai` and `spark`, which are multi-platform and run natively. An amd64 image runs under Rosetta emulation, and Rosetta deadlocks in Jupyter kernels: JupyterLab opens, but a notebook stays at "Connecting" and no cell runs (the 5.2.x images show the same). `gpu` and `rapids` are amd64 only and need an NVIDIA GPU host anyway. To build an arm64 image locally: `PLATFORM=linux/arm64 ./build.sh agentic local/ <version>`.

## Further documentation and usage
Please find further information and documentation on splunkbase: [Download and install the Splunk App for Data Science and Deep Learning](https://splunkbase.splunk.com/app/4607/) and refer to the [official documentation pages](https://docs.splunk.com/Documentation/DSDL) on how to install and use the app.
