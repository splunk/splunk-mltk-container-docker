# Image Build Report (5.3.0, branch `v5.3.0-dev`)

## Result

The 11 images of 5.2.5 are built from **one multi-stage Dockerfile** (`docker/Dockerfile`) as
**6 build targets**. All 10 published 5.2.5 image references and `images.conf` stanzas are kept:
images that only differed in their library mix were merged and are published under every old
name. The internal `agentic-ai-debian` test image is retired.

| | 5.2.5 (10 published references) | 5.3.0 (same 10 references) | Change |
|---|---:|---:|---:|
| Trivy findings | 32,357 | 1,914 | −94.1 % |
| CRITICAL | 157 | 0 | −100 % |
| HIGH + CRITICAL | 3,383 | 132 | −96.1 % |
| Distinct images and size | 10 images, 38.3 GB | 6 images (cpu amd64 + arm64, llm, gpu, rapids, spark), 19.2 GB | −50 % |
| Operating systems | 6 (UBI 9, Debian 11 and 13, DHI Debian 13, Ubuntu 20.04 and 22.04) | 1 (UBI 9.8) | |
| Python versions | 4 (3.8, 3.9, 3.10, 3.11) | 1 (3.12) | |

The 5.2.5 numbers understate that baseline: Trivy reports no OS findings at all for the two
Ubuntu 20.04 images (end of life), so `golden-gpu` and `golden-gpu-transformers` show only
their Python findings there.

## Images

Built locally for linux/amd64 (and linux/arm64 for `cpu`) from this branch, smoke-tested with
`tools/smoke-test.sh` and scanned with Trivy 0.64.1 (`tools/scan.sh --gate`, vulnerability DB of
2026-10-20). Sizes from `docker image inspect` (same method as the 5.2.5 report).

| Target | Published as (5.2.x names) | Size | Modules importing (5.2.5 union) | Findings | CRIT | HIGH |
|---|---|---:|---:|---:|---:|---:|
| `cpu` amd64 | `golden-cpu`, `escu-cpu`, `golden-cpu-transformers` | 1.70 GB | 47 (40) | 184 | 0 | 11 |
| `cpu` arm64 | `golden-cpu:<v>-arm64` | 1.35 GB | 47 (40) | 180 | 0 | 11 |
| `llm` | `ubi-llm-rag` (stanza `llm-rag`) | 1.77 GB | 61 (46) | 185 | 0 | 12 |
| `agentic` | `ubi-llm-rag:agentic-ai-<v>` | 1.77 GB | 61 (46) | 185 | 0 | 12 |
| `gpu` | `golden-gpu`, `golden-gpu-transformers` | 5.23 GB | 51 (48) | 184 | 0 | 11 |
| `rapids` | `golden-gpu-rapids` | 7.03 GB | 36 (22) | 191 | 0 | 10 |
| `spark` | `spark` | 2.12 GB | 51 (44) | 253 | 0 | 32 |

"Modules importing" counts the `app/model` modules that import in the image; the number in
brackets is the union of the modules that imported in the 5.2.5 images the target replaces
(`tests/expected/<target>.txt`). No module regressed in any image.

### Per published reference

| Reference | 5.2.5 base | 5.2.5 findings / H+C | 5.3.0 findings / H+C |
|---|---|---:|---:|
| `golden-cpu:<v>` | UBI 9 | 881 / 127 | 184 / 11 |
| `golden-cpu:<v>-arm64` | UBI 9 | 990 / 188 | 180 / 11 |
| `escu-cpu:<v>` | Debian 11 (end of life) | 6,912 / 720 | 184 / 11 |
| `golden-cpu-transformers:<v>` | Debian 13 | 4,239 / 379 | 184 / 11 |
| `ubi-llm-rag:<v>` | UBI 9 | 1,105 / 224 | 185 / 12 |
| `ubi-llm-rag:agentic-ai-<v>` | UBI 9 | 7,079 / 253 | 185 / 12 |
| `golden-gpu:<v>` | Ubuntu 20.04 (end of life, OS not scanned) | 357 / 152 | 184 / 11 |
| `golden-gpu-transformers:<v>` | Ubuntu 20.04 (end of life, OS not scanned) | 289 / 138 | 184 / 11 |
| `golden-gpu-rapids:<v>` | Ubuntu 22.04 + conda | 4,664 / 508 | 191 / 10 |
| `spark:<v>` | Ubuntu 22.04 + conda | 5,841 / 694 | 253 / 32 |

## Where the reduction comes from

1. **No build tools in runtime images.** Compilers, `-devel` packages and kernel headers stay in
   the builder stage. In 5.2.5 they were about 75 % of all findings (kernel headers alone 67.6 %).
2. **A minimal operating system.** The runtime is an exact list of 71 UBI 9 RPMs in a ubi-micro
   root (`docker/rootfs/allow.txt`), without package manager, PAM, su/sudo, ssh, Node.js, setuid
   files or a second Python. The Spark image adds 10 RPMs for Java 21 (`allow-jre.txt`).
3. **One Python and fresh, locked libraries.** Python 3.12 everywhere, four `uv.lock` profiles with
   hashes (`deps/`), shared packages at the same versions; legacy pins removed (pymc3, TF 2.8 to
   2.10, spaCy 2, xgboost 0.90, ...).
4. **Spark:** no jupyter/conda base, unused Spark components removed, bundled netty, Jackson and
   lz4-java replaced by hash-pinned patch releases.
5. **GPU:** CUDA from the NVIDIA wheels only (5.2.5 shipped a second, unused CUDA in the OS layer).

## Residual HIGH findings

- **OS packages without a released fix** (Red Hat status "affected"): expat (2), libuuid (2) and
  python3.12 (2) in every image. They are picked up by the monthly
  rebuild (`tools/update-digests.sh`) once Red Hat ships fixes.
- **pip's vendored copies** of urllib3 (2), msgpack and setuptools: 4 HIGH in every image, fixed
  when pip ships a release with newer vendored libraries.
- **nltk 3.10.3:** one HIGH without a fix (cpu, llm, gpu, spark).
- **llama-index-core 0.12:** one HIGH in the llm images, fixed only in 0.13, which needs a port of
  the LLM/RAG modules (they use APIs removed in 0.13).
- **Spark:** jars that Hadoop and Parquet shade into their own artefacts (hadoop-client-runtime,
  parquet-jackson); fixed only by Spark/Hadoop releases.

No image has a CRITICAL finding, a secret, or a finding on an end-of-life OS.

## Verification

For every image (`tools/smoke-test.sh`):
- security invariants: no setuid/setgid files, compilers, headers, package manager, Node.js, ssh,
  su/sudo, curl/wget/openssl/perl; one Python; read-only `/etc/passwd` and image content; no
  private key; no world-writable files; time zone database present; every ELF dependency resolves;
- every `app/model` module imported; none of the 5.2.5 modules regressed;
- functional checks: torch and TensorFlow compute; the ESCU pretrained models (the 3 torch models
  and the DGA notebook end to end, which failed in every 5.2.5 image); spaCy 3 with its model;
  a Spark SQL/FPGrowth/ALS job; the CUDA 12.6 build of the GPU stack (GPU architectures sm_50 to
  sm_90); cuDF import; the agentic application;
- PROD start (HTTPS API on 5000) as the default user and as an arbitrary UID with a read-only
  root filesystem; DEV start with JupyterLab (8888), MLflow UI (6000) and TensorBoard (6006).

The branch was also reviewed in five areas (Dockerfile/rootfs, entrypoint vs. the 5.2.x bootstrap
contract, build tooling, dependency profiles, documentation/compatibility) with an adversarial
verifier per area; the confirmed findings are fixed in commit 8984d9a and later.

## Not verified here

- **No NVIDIA GPU** on the build host: the GPU and RAPIDS images were import- and CPU-tested only.
  Driver matrix (R535+), TensorFlow and PyTorch sharing CUDA in one process, and RAPIDS kernels
  need a GPU run before release. Blackwell GPUs are not supported by the CUDA 12.6 build.
- **arm64** was built and tested for `cpu` only; the `llm`, `agentic` and `spark` locks resolve for
  arm64 and the llm lock was install- and import-tested on arm64 during locking.
- **Playwright UI tests** (`testing/`, against a Splunk instance) and calls to real LLM providers
  were not run. Images were not pushed or signed.
- **Known application issues** unchanged by this branch: `ServiceContext` (llama-index 0.12) fails
  at fit time in `RAG_Encoding`, `RAG_Inferencing` and `llm_rag_script_saas_llm`;
  `timesfm_multi_resolution` imports a missing name; the agentic routes `/chatbox`, `/logReview`
  and `/mcp/status` have no token check.

## Reproduce

```
./build.sh golden-cpu local/ 5.3.0                  # one target, one platform
tools/smoke-test.sh local/mltk-container-golden-cpu:5.3.0 cpu
tools/scan.sh --gate local/mltk-container-golden-cpu:5.3.0
tools/lock.sh --check                               # locks and cpu-constraint blocks are current
```

Build the images one at a time: a GPU or RAPIDS build needs about 30 to 40 GB of Docker disk,
including the build cache (`docker builder prune` frees it afterwards).
