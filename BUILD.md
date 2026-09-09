# Image Build Report

## Result

All 11 images declared in `tag_mapping.csv` were built locally from the current
working tree with repository prefix `splunk/` and version `5.2.5`. The normal
build was attempted first for each mapping. Where a build or runtime smoke test
exposed a dependency problem, the requirements were compiled, the underlying
issue was corrected, and the image was rebuilt.

The working tree already contained uncommitted changes when this work began.
Those changes were retained and were part of the build context.

## Final image inventory

The IDs below come from `docker image inspect` after the final rebuilds.

| Mapping | Platform | Local image ID | Size |
|---|---|---|---:|
| `golden-cpu` | `linux/amd64` | `sha256:f8442f1c228f3610bc571d669b1acf0f8beaac5ccaf433397168287980508062` | 2.16 GB |
| `golden-gpu` | `linux/amd64` | `sha256:4b13f6b2e7b564876f2c976e1d5296cd489c473b4f552fc7c99f9747183d78e2` | 6.55 GB |
| `golden-cpu-transformers` | `linux/amd64` | `sha256:6027a2bc3e62c68c3d6993f1d4a1746ce5453700623c01864505b86925152e2d` | 1.85 GB |
| `golden-gpu-transformers` | `linux/amd64` | `sha256:4f8b88cd7853907a63437ee710547b5c8d86cd10fe71cd2c0fdab431e805d01d` | 8.00 GB |
| `golden-gpu-rapids` | `linux/amd64` | `sha256:1464533cd1dc7499f1fa9480957b25bd5c6b500ab94ff3207b5824e373b3c11c` | 7.22 GB |
| `ubi-llm-rag` | `linux/amd64` | `sha256:2e4e186a5cd0c02b5471ebbc4c3195fa26e609c5e8a72d9dae9acde1046b54d9` | 1.57 GB |
| `escu-cpu` | `linux/amd64` | `sha256:8ae3b128961a7b59b286b9a1572650a6e879a671d659a941f14a71b3256fbd43` | 4.77 GB |
| `spark` | `linux/amd64` | `sha256:60ee9c1e0d9be8c6bf76b92205adb028109a84f541b29f054236458bb71222d3` | 3.11 GB |
| `golden-cpu-arm` | `linux/arm64` | `sha256:975c2b11ab2a47f8ea2bf1c2afc5a17a514ca68eaa080749b556c260bcabf19d` | 1.11 GB |
| `agentic-ai` | `linux/amd64` | `sha256:1ea1881cf56b24a894caaa1e3ea0391e895ccaa912047999cefbeee93e746dc7` | 1.91 GB |
| `agentic-ai-debian` | `linux/amd64` | `sha256:a476f8caec63e34204f38537c469ec6d1e0aa6c630355ebcd989103240be50b9` | 1.80 GB |

Every image is tagged `splunk/mltk-container-<mapping>:5.2.5`.

### Repository compatibility aliases

The initial build names follow the internal `tag_mapping.csv` identifiers. A
comparison with the DSDL 5.2.4 app configuration showed that three variants are
published as tags in established repositories rather than as separate Docker
repositories. The following additional local tags were created for 5.2.5:

| Build tag | Canonical 5.2.5 repository tag |
|---|---|
| `splunk/mltk-container-golden-cpu-arm:5.2.5` | `splunk/mltk-container-golden-cpu:5.2.5-arm64` |
| `splunk/mltk-container-agentic-ai:5.2.5` | `splunk/mltk-container-ubi-llm-rag:agentic-ai-5.2.5` |
| `splunk/mltk-container-agentic-ai-debian:5.2.5` | `splunk/mltk-container-ubi-llm-rag:agentic-ai-debian-5.2.5` |

The first two follow the exact 5.2.4 repository conventions. The Debian
agentic image is new in 5.2.5 and follows the existing agentic variant naming
scheme inside `mltk-container-ubi-llm-rag`. The original build tags were kept as
aliases. The `[ubi-llm-rag]` configuration stanza was also restored to the
existing public stanza identifier `[llm-rag]`. `build.sh` now applies these
canonical aliases and configuration names automatically on future rebuilds.

`push_5.2.5_images.sh` publishes the ten release images under these canonical
names. It intentionally excludes the `agentic-ai-debian` test image, verifies
that every publishable tag exists locally before the first push, and supports a
non-mutating `--dry-run` check.

## Build history and corrections

### Base and build infrastructure

- The requirements compiler now fails immediately when its Docker build,
  container creation, or lockfile copy fails, and it removes its tag-specific
  temporary container on failure.
- Compiled locks are generated during the compiler-image build. The helper now
  uses `docker create` followed by `docker cp`, so a base image's inherited
  entrypoint cannot start Jupyter and block lockfile retrieval. This was exposed
  by the Spark compiler image.
- Debian requirements compilation is noninteractive (`Etc/UTC`) and configures
  bounded apt retries and network timeouts. Pip compilation and the general
  Debian image install also use bounded retries without bypassing package hashes.
- Temporary `dos2unix` install/remove transactions were replaced with `sed`
  line-ending normalization in the affected Transformer, RAPIDS, Spark, and
  Red Hat agentic Dockerfiles. This removed fragile late package-manager work.
- Docker's internal storage filled while building the large CUDA images. Only
  unused BuildKit cache and task-created temporary compiler images were removed;
  requested images and Docker volumes were retained. The cumulative unused
  build cache reclaimed during this run was 135.524 GB.

### `golden-cpu` and `golden-cpu-arm`

- Both normal builds succeeded. The x86 image was built for `linux/amd64`; the
  ARM mapping produced a native `linux/arm64` image.

### `golden-gpu`

- The first normal build rejected the downloaded
  `nvidia-cublas-cu12==12.1.3.1` artifact because its SHA-256 did not match the
  compiled lock. A later compiler download encountered another corrupt artifact.
- The compiler was made resilient to transient apt/PyPI failures while retaining
  integrity checks. The GPU lock was recompiled and the second image build
  succeeded.

### CPU and GPU Transformer images

- The CPU Transformer's first normal build failed while compiling the legacy
  `numexpr==2.7.3` sdist against current build tooling. The source requirement
  was updated to `numexpr==2.8.4`, which has a compatible wheel, and the image
  succeeded on its second build.
- The GPU Transformer image builds CPython 3.11.12. Its old compiler path did
  not reliably resolve for that interpreter, so the requirements compiler now
  uses `python:3.11-slim` and pip is consistently invoked through
  `python3 -m pip`. The missing `curl` prerequisite and late `dos2unix` apt
  transaction were also corrected before the final successful build.

### `escu-cpu`

- The first three attempts failed because the configured Debian Bullseye
  security route returned 404 for the indexed `e2fsprogs` package. The security
  source was changed to `https://security.debian.org/debian-security`, with
  bounded apt retries/timeouts, and attempt four succeeded.

### Agentic images

- On the Red Hat image, OpenTelemetry's site hook interfered with the isolated
  build interpreter used for the repository-local `dsdlsupport` package. The
  declared Hatchling backend is now installed explicitly and isolation is
  bypassed only for that trusted local package.
- Independent post-lock upgrades of `langchain-aws` and
  `langchain-mcp-adapters` broke the resolved LangChain constraints. Those
  upgrades were removed because both packages are already part of the compiled
  agentic lock. The Red Hat image succeeded on its fourth complete build.
- On the Debian image, installing NodeSource's Debian 13 package replaced the
  DHI Python 3.11 runtime with Debian Python 3.13. Node is not required for the
  prebuilt JupyterLab 4 runtime assets, so it was removed from this image's
  build dependencies.
- The DHI image's optional `pip3` Socket Firewall wrapper required an external
  token. Module-form pip calls bypass that optional wrapper, and a clean
  `/opt/dltk-venv` prevents unrelated packages in the broad DHI development
  environment from contaminating the resolved lock. The Debian image reuses
  the combined CPython 3.11 agentic lock and succeeded on its third build.

### `golden-gpu-rapids`

- The first image built, but importing cuDF exposed a binary ABI failure caused
  by NumPy 2.0.2 combined with pandas 1.5.3. The RAPIDS 24.04 base's compatible
  core versions were made explicit: NumPy 1.26.4, pandas 2.2.1, PyArrow 14.0.2,
  protobuf 4.25.3, urllib3 1.26.18, and pynvml 11.4.1.
- Requirements were recompiled and the corrected image passed imports for cuDF,
  cuML, and cuGraph. Its `pip check` result was compared with the untouched
  RAPIDS base. The final image has the same base metadata warnings for CUDA
  helper distributions, plus an xgboost platform warning caused by running the
  amd64 image through QEMU on an ARM host; no new dependency-version conflict
  was introduced by the rebuilt lock.

### `spark`

- The first image installed PySpark 4.2.0 over a Spark 3.5.1 JVM distribution.
  A real local `SparkSession` then failed with `JavaPackage` not callable.
- Pinning `pyspark==3.5.1` aligned Python and JVM Spark, and a second image ran a
  local job. Dependency checking then exposed upgrades incompatible with the
  upstream notebook environment: NumPy 2.4.6 versus Numba 0.60, IPython 9.17.1
  versus ipympl 0.9.4, and JupyterLab 4.6.3 versus Notebook 7.2.1.
- The untouched base versions (NumPy 1.26.4, IPython 8.26.0, and JupyterLab
  4.2.4) were pinned alongside PySpark 3.5.1. After recompilation, the third
  image passed a real local Spark job and returned to the base image's sole
  `certipy` platform warning.
- The generated display title was corrected from Spark 3.5.0 to Spark 3.5.1 to
  match the base tag and verified runtime.

## Runtime verification

- All 11 requested tags were inspected successfully. Ten report
  `linux/amd64`; `golden-cpu-arm` reports `linux/arm64`.
- `agentic-ai`: Python 3.11.13; `langchain-core` 0.3.86; standard-library
  `json`/`traceback` and `dsdlsupport` imports passed; `pip check` reported no
  broken requirements.
- `agentic-ai-debian`: Python 3.11.16; `langchain-core` 0.3.86; the same imports
  passed; `pip check` reported no broken requirements.
- `golden-gpu-transformers`: Python 3.11.12, Torch 2.9.0+cu128, TensorFlow
  2.20.0, and Transformers 4.57.6 imported; `dsdlsupport` imported and
  `pip check` reported no broken requirements.
- `golden-gpu-rapids`: Python 3.9.19, NumPy 1.26.4, pandas 2.2.1, PyArrow
  14.0.2, cuDF 24.04.01, cuML 24.04.00, cuGraph 24.04.00, and `dsdlsupport`
  imported successfully.
- `spark`: a local Spark 3.5.1 session started, `spark.range(3).count()` returned
  `3`, and `dsdlsupport` imported. `pip check` reports only
  `certipy 0.1.3 is not supported on this platform`, matching the untouched
  upstream base image.
- The build host is an ARM-based macOS machine and has no NVIDIA runtime.
  CUDA images were therefore built for amd64 under emulation and import-tested,
  but actual GPU kernel execution could not be validated on this host. The
  expected no-driver `libcuda.so.1` warning occurred during RAPIDS imports.
- Initial GPU Transformer smoke attempts also encountered exhausted Docker
  internal temporary storage while Matplotlib initialized. After the safe cache
  cleanup, the same test passed with a bounded tmpfs; this was a host storage
  condition rather than an image dependency failure.

## Configuration and repository checks

- `images_conf_files/images.conf` contains 11 stanzas, one for every mapping,
  and all generated image references and display versions use `5.2.5`.
- `build.sh` and `compile_image_python_requirements.sh` pass `bash -n`.
- `git diff --check` passes.
- BuildKit still emits two non-fatal lint warnings for several existing
  Dockerfiles: `InvalidDefaultArgInFrom` for an unset default `BASE_IMAGE`, and
  `LegacyKeyValueFormat` for older `ENV key value` syntax. These do not prevent
  any requested image from building.
