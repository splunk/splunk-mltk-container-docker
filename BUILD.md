# Image Build Report (branch `v5.2.5-dev`, images built as 5.3.0)

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
| Distinct images and size | 10 images, 38.3 GB | 6 images (cpu amd64 + arm64, llm, gpu, rapids, spark), 20.3 GB | −47 % |
| Operating systems | 6 (UBI 9, Debian 11 and 13, DHI Debian 13, Ubuntu 20.04 and 22.04) | 1 (UBI 9.8) | |
| Python versions | 4 (3.8, 3.9, 3.10, 3.11) | 1 (3.12) | |

The 5.2.5 numbers understate that baseline: Trivy reports no OS findings at all for the two
Ubuntu 20.04 images (end of life), so `golden-gpu` and `golden-gpu-transformers` show only
their Python findings there.

## Images

Built locally for linux/amd64 (and linux/arm64 for `cpu`) from this branch, smoke-tested with
`tools/smoke-test.sh` and scanned with Trivy 0.64.1 (`tools/scan.sh --gate`, vulnerability DB of
2026-10-20; `gpu` and `rapids` rebuilt and rescanned with the DB of 2026-10-21 after the move to
CUDA 12.9, see [GPU images on the RAPIDS CUDA version](#gpu-images-on-the-rapids-cuda-version)).
Sizes from `docker image inspect` (same method as the 5.2.5 report).

| Target | Published as (5.2.x names) | Size | Modules importing (5.2.5 union) | Findings | CRIT | HIGH |
|---|---|---:|---:|---:|---:|---:|
| `cpu` amd64 | `golden-cpu`, `escu-cpu`, `golden-cpu-transformers` | 1.70 GB | 47 (40) | 184 | 0 | 11 |
| `cpu` arm64 | `golden-cpu:<v>-arm64` | 1.35 GB | 47 (40) | 180 | 0 | 11 |
| `llm` | `ubi-llm-rag` (stanza `llm-rag`) | 1.77 GB | 61 (46) | 185 | 0 | 12 |
| `agentic` | `ubi-llm-rag:agentic-ai-<v>` | 1.77 GB | 61 (46) | 185 | 0 | 12 |
| `gpu` | `golden-gpu`, `golden-gpu-transformers` | 5.75 GB | 51 (48) | 184 | 0 | 11 |
| `rapids` | `golden-gpu-rapids` | 7.57 GB | 36 (22) | 191 | 0 | 10 |
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
5. **GPU:** CUDA from the NVIDIA wheels only (5.2.5 shipped a second, unused CUDA in the OS layer);
   `gpu` and `rapids` share one CUDA 12.9.1 user space, the CUDA of RAPIDS 26.10 (5.2.5: three
   images on CUDA 12.4, 11.3 and 12.2).

## GPU images on the RAPIDS CUDA version

5.2.5 built its GPU images on three CUDA versions: `golden-gpu` on nvidia/cuda 12.4.1 (Ubuntu 20.04,
torch 2.4.1+cu121), `golden-gpu-transformers` on nvidia/cuda 11.3.1 (Ubuntu 20.04, with torch
2.9.0+cu128 in its venv) and `golden-gpu-rapids` on rapidsai/base 24.04 (nvidia/cuda 12.2.2, Ubuntu
22.04). The latest stable RAPIDS, 26.10, is built with CUDA 12.9.1 (`rapidsai/base:26.10-cuda12`;
the CUDA 13 variant `26.10-cuda13` uses 13.3.0).

`gpu` (`golden-gpu`, `golden-gpu-transformers`) and `rapids` now use exactly that CUDA: cuda-toolkit
12.9.1, cuDNN 9.20.0.48, NCCL 2.29.7 and nvJitLink 12.9.86, at the same versions in both locks and
without any CUDA 13 package (smoke check `cuda_user_space`). This takes torch 2.13.0+cu129, the
newest PyTorch with a CUDA 12.9 build (2.14 ships CUDA 12.6 and CUDA 13 builds only). The CPU
images keep torch 2.14.1.

| | before (torch 2.14.1+cu126) | now (torch 2.13.0+cu129) |
|---|---|---|
| CUDA libraries | 12.6.3 (nvJitLink 12.9) | 12.9.1, as RAPIDS 26.10 |
| cuDNN | 9.10.2 | 9.20.0 |
| PyTorch kernels | sm_50 to sm_90 (Maxwell to Hopper) | sm_75 to sm_120 (Turing to Blackwell) |
| TensorFlow cuDNN ops on Pascal/Volta | yes | no (cuDNN 9.11 and later need Turing) |
| Minimum driver | R535 | R535 (Blackwell: R570) |
| Size `gpu` / `rapids` | 5.23 / 7.03 GB | 5.75 / 7.57 GB |
| Trivy `gpu` / `rapids` (findings, HIGH) | 184, 11 / 191, 10 | 184, 11 / 191, 10 |

Versus 5.2.5, PyTorch gains Blackwell for `golden-gpu` and Hopper and Blackwell for
`golden-gpu-rapids`; it loses Volta (V100) in all three, and Maxwell and Pascal in `golden-gpu`
and `golden-gpu-rapids`. TensorFlow 2.21 does
not run on Blackwell with either build (its XLA stops on sm_120; fixed in TensorFlow 2.22).
torch 2.13 is one minor behind the CPU images, the cu129 build sits outside PyTorch's support matrix,
and 2.13 gets no patch releases. Scanners report no advisory for either torch version today.

Alternatives checked:
- **torch 2.14.1+cu126 with the CUDA libraries overridden to 12.9.1** (uv `override-dependencies`
  on `cuda-toolkit`). It resolves, with cuDNN 9.10.2. It keeps Maxwell to Hopper, the torch version
  of the CPU images and the 2.14.0 security hardening. It has no Blackwell kernels, and torch then
  runs on CUDA libraries it was not released with.
- **CUDA 13** (torch 2.14.1+cu130/cu132, RAPIDS cu13 as `rapidsai/base:26.10-cuda13`). TensorFlow
  has no CUDA 13 GPU wheels (2.21, and 2.22rc0 neither); it also needs driver R580+ and Turing or
  newer. This is the planned next GPU line.

Decision (2026-10-21): keep torch 2.13.0+cu129, which gives Blackwell support and the exact RAPIDS
CUDA version, and drop Maxwell, Pascal and Volta, which the CUDA 13 line drops as well.

Found and fixed while verifying:
- `import triton` after `import tensorflow` crashes the process (segfault with triton 3.7.1 and
  3.8.0, so the earlier build had it too). bitsandbytes and `torch.compile` import triton, so a
  TensorFlow model followed by a quantised or compiled torch model in one DSDL process crashed.
  The gpu venv now imports triton just before TensorFlow (`docker/finalize-venv.sh`, smoke check
  `tensorflow_then_triton`).
- `finalize-venv` kept all 16 bitsandbytes native libraries when no exact CUDA match existed. It
  now keeps the one bitsandbytes loads (cuda128 for CUDA 12.9).
- The PyTorch cu129 index publishes torchvision 0.28.0 without a hash. It is now a direct-URL
  source, so `uv.lock` records its sha256; no lock has an unhashed artifact.

## Residual HIGH findings

- **OS packages without a released fix** (Red Hat status "affected"): expat (2), libuuid (2) and
  python3.12 (2) in every image. They are picked up by the monthly
  rebuild (`tools/update-digests.sh`) once Red Hat ships fixes.
- **pip's vendored copies** of urllib3 (2), msgpack and setuptools: 4 HIGH in every image, fixed
  when pip ships a release with newer vendored libraries. A CRITICAL published on 2026-10-20 in
  pip 26.2.1's vendored certifi (GHSA-6q48-6c7w-9gw2, compromised emSign intermediate CAs) is fixed
  by pip 26.2.2, which the locks now pin; images built before that need a rebuild.
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
  a Spark SQL/FPGrowth/ALS job; the GPU stack: one CUDA 12.9 wheel set without CUDA 13 packages in
  `gpu` and `rapids`, PyTorch architectures sm_75 to sm_120, ptxas and libdevice 12.9 for XLA,
  TensorFlow followed by bitsandbytes/triton in one process, the bitsandbytes library it loads, a
  random-weight TimesFM checkpoint load and forecast; cuDF import; the agentic application;
- an unknown host name fails as "not known" (NSS modules present), and every shipped notebook is
  valid nbformat (JupyterLab's post-save export needs that);
- PROD start (HTTPS API on 5000) as the default user, and as an arbitrary UID with a read-only root
  filesystem on a volume whose TLS files belong to another UID; DEV start with JupyterLab (8888),
  MLflow UI (6000) and TensorBoard (6006), a notebook kernel started through JupyterLab (login,
  websocket) running a cell, and the MLflow API serving its store; the same DEV start on a volume as
  5.2.x left it (MLflow file store with a run, which is migrated to SQLite, and an existing
  `~/.ipython`); no DEV service may exit.
- Jupyter kernels of amd64 images hang under Rosetta (Docker Desktop on Apple silicon): Rosetta
  deadlocks in its own lock inside the kernel (also with the 5.2.x images). On Apple silicon use the
  arm64 images; the smoke test reports the kernel check as SKIP there.

The branch was also reviewed in five areas (Dockerfile/rootfs, entrypoint vs. the 5.2.x bootstrap
contract, build tooling, dependency profiles, documentation/compatibility) with an adversarial
verifier per area; the confirmed findings are fixed in commit 8984d9a and later.

## Not verified here

- **No NVIDIA GPU** on the build host: the GPU and RAPIDS images were import- and CPU-tested only.
  Before release they need a GPU run: the driver matrix (R535 and R580 on Turing to Hopper, R570+
  on Blackwell), TensorFlow on Hopper with R535 (PTX JIT of its compute_90 code), `torch.compile`
  on Blackwell, TensorFlow and PyTorch sharing a GPU in one process, the bitsandbytes cuda128
  library on CUDA 12.9, and RAPIDS kernels.
- **arm64** was built and smoke-tested for `cpu`, and on 2026-10-21 for `llm` and `agentic` (all checks
  passed, including the JupyterLab kernel; Trivy then flagged the pip certifi CRITICAL above). The
  `spark` lock resolves for arm64, but no arm64 `spark` image was built.
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
