# Image Build Report (branch `v5.2.5-dev`, release 5.2.5)

**5.2.5 test images** below are the `splunk/mltk-container-*:5.2.5` images pushed to Docker Hub in
September 2026 from the original v5.2.5 branch (10 image references built the 5.2.4 way; the internal
`agentic-ai-debian` image was built but not pushed), for testing only and never released with the
DSDL app. The images of this build replace them under the same tags.

## Result

What the 11 images of the original v5.2.5 branch contained (the 10 5.2.5 test images and the internal
`agentic-ai-debian`) is now built from **one multi-stage Dockerfile**
(`docker/Dockerfile`) as **5 build targets**, published as five images, one name and one `images.conf`
stanza each: Golden Image CPU (`golden-cpu`), LLM RAG (`llm-rag`), Agentic AI (`agentic-ai`), Golden
Image GPU (`golden-gpu`) and Rapids 26.10 (`golden-gpu-rapids`). Images that only differed in their
library mix were merged: `golden-cpu-arm`, `escu-cpu`, `golden-cpu-transformers` and `spark` into Golden
Image CPU, `golden-gpu-transformers` into Golden Image GPU; those names are no longer published. The
internal `agentic-ai-debian` test image is retired.

| | 5.2.5 test images (10 published references) | 5.2.5 (5 published images) | Change |
|---|---:|---:|---:|
| Trivy findings | 32,357 | 1,038 | −96.8 % |
| CRITICAL | 157 | 0 | −100 % |
| HIGH + CRITICAL | 3,383 | 89 | −97.4 % |
| Distinct images and size | 10 images, 38.3 GB | 5 images (cpu amd64 + arm64, llm, gpu, rapids), 19.0 GB | −50 % |
| Operating systems | 6 (UBI 9, Debian 11 and 13, DHI Debian 13, Ubuntu 20.04 and 22.04) | 1 (UBI 9.8) | |
| Python versions | 4 (3.8, 3.9, 3.10, 3.11) | 1 (3.12) | |

The numbers of the 5.2.5 test images understate that baseline: Trivy reports no OS findings at all for the two
Ubuntu 20.04 images (end of life), so `golden-gpu` and `golden-gpu-transformers` show only
their Python findings there.

## Images

The 5.2.5 release images, all built on 2026-10-28 from commit ea31934 with `PLATFORM=all ./build.sh
<target> splunk/ 5.2.5`, one target at a time. Every image has the same payload: the notebook layout
of 8a2a7a0 (`notebooks/deprecated/`, `notebooks/README.md`), the `tabpfn_model` notebook and the
`multivariate_lstm` fix for numpy 2.5; `cpu` includes Spark, tabpfn 9.2.0 and huggingface-hub 1.33.0.
`cpu`, `llm` and `agentic` are built for linux/amd64 and linux/arm64, `gpu` and `rapids` for
linux/amd64. Every image was smoke-tested on each platform with `tools/smoke-test.sh` (arm64 natively,
amd64 under emulation on Apple silicon, where the Jupyter kernel check is skipped) and scanned per
platform with Trivy 0.64.1 (`tools/scan.sh --gate`, vulnerability DB of 2026-10-28). Sizes from
`docker image inspect` (same method
as the report of the 5.2.5 test images); arm64: `llm`/`agentic` 1.42 GB, with the same findings as amd64.

| Target | Published as (title, stanza) | Size | Modules importing (test images union) | Findings | CRIT | HIGH |
|---|---|---:|---:|---:|---:|---:|
| `cpu` amd64 | Golden Image CPU (`golden-cpu`) | 2.13 GB | 51 (44) | 296 | 0 | 39 |
| `cpu` arm64 | Golden Image CPU (`golden-cpu`), arm64 | 1.78 GB | 51 (44) | 296 | 0 | 39 |
| `llm` | LLM RAG (`llm-rag`) | 1.77 GB | 61 (46) | 184 | 0 | 15 |
| `agentic` | Agentic AI (`agentic-ai`) | 1.77 GB | 61 (46) | 184 | 0 | 15 |
| `gpu` | Golden Image GPU (`golden-gpu`) | 5.75 GB | 51 (48) | 184 | 0 | 11 |
| `rapids` | Rapids 26.10 (`golden-gpu-rapids`) | 7.57 GB | 36 (22) | 190 | 0 | 9 |

"Modules importing" counts the `app/model` modules that import in the image; the number in
brackets is the union of the modules that imported in the 5.2.5 test images the target replaces. The
smoke test's lists (`tests/expected/<target>.txt`) hold that union plus `transformers_classification`
and, where they import, `transformers_summarization`, which failed in the 5.2.5 test images and import now.
No module regressed in any image.

### Per test image reference

| 5.2.5 test image | test image base | test image findings / H+C | replaced by (5.2.5) | findings / H+C |
|---|---|---:|---|---:|
| `golden-cpu:<v>` | UBI 9 | 881 / 127 | Golden Image CPU | 296 / 39 |
| `golden-cpu:<v>-arm64` | UBI 9 | 990 / 188 | Golden Image CPU (arm64) | 296 / 39 |
| `escu-cpu:<v>` | Debian 11 (end of life) | 6,912 / 720 | Golden Image CPU | 296 / 39 |
| `golden-cpu-transformers:<v>` | Debian 13 | 4,239 / 379 | Golden Image CPU | 296 / 39 |
| `ubi-llm-rag:<v>` | UBI 9 | 1,105 / 224 | LLM RAG | 184 / 15 |
| `ubi-llm-rag:agentic-ai-<v>` | UBI 9 | 7,079 / 253 | Agentic AI | 184 / 15 |
| `golden-gpu:<v>` | Ubuntu 20.04 (end of life, OS not scanned) | 357 / 152 | Golden Image GPU | 184 / 11 |
| `golden-gpu-transformers:<v>` | Ubuntu 20.04 (end of life, OS not scanned) | 289 / 138 | Golden Image GPU | 184 / 11 |
| `golden-gpu-rapids:<v>` | Ubuntu 22.04 + conda | 4,664 / 508 | Rapids 26.10 | 190 / 9 |
| `spark:<v>` | Ubuntu 22.04 + conda | 5,841 / 694 | Golden Image CPU | 296 / 39 |

The 5.2.5 totals above count each of the five published images once (Golden Image CPU on one
platform, as the test images were counted).

## Where the reduction comes from

1. **No build tools in runtime images.** Compilers, `-devel` packages and kernel headers stay in
   the builder stage. In the 5.2.5 test images they were about 75 % of all findings (kernel headers
   alone 67.6 %).
2. **A minimal operating system.** The runtime is an exact list of 71 UBI 9 RPMs in a ubi-micro
   root (`docker/rootfs/allow.txt`), without package manager, PAM, su/sudo, ssh, Node.js, setuid
   files or a second Python. The `cpu` image adds 10 RPMs for Java 21, for Spark (`allow-jre.txt`).
3. **One Python and fresh, locked libraries.** Python 3.12 everywhere, four `uv.lock` profiles with
   hashes (`deps/`), shared packages at the same versions; legacy pins removed (pymc3, TF 2.8 to
   2.10, spaCy 2, xgboost 0.90, ...).
4. **Spark:** no jupyter/conda base, unused Spark components removed, bundled netty, Jackson and
   lz4-java replaced by hash-pinned patch releases.
5. **GPU:** CUDA from the NVIDIA wheels only (the 5.2.5 test images shipped a second, unused CUDA in
   the OS layer); `gpu` and `rapids` share one CUDA 12.9.1 user space, the CUDA of RAPIDS 26.10
   (5.2.5 test images: three images on CUDA 12.4, 11.3 and 12.2).

## GPU images on the RAPIDS CUDA version

In the 5.2.5 test images the GPU images were built on three CUDA versions: `golden-gpu` on nvidia/cuda 12.4.1 (Ubuntu 20.04,
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

Versus the 5.2.5 test images, PyTorch gains Blackwell for `golden-gpu` and Hopper and Blackwell for
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

- **OS packages without a released fix** (Red Hat status "affected"): expat (1) and libuuid (2) in
  every image, java-21-openjdk-headless (2) in `cpu`. They are picked up by the monthly rebuild
  (`tools/update-digests.sh`) once Red Hat ships fixes.
- **mlflow 3.17.1:** one HIGH (CVE-2026-53995) in every image, and **pypdf 6.21.1:** three HIGH in the
  llm images, both fixed in releases of 2026-10-21 (mlflow 3.17.2, pypdf 6.21.2) after these images
  were locked. The 5.2.5 images ship as built; the next lock refresh (`tools/lock.sh
  --upgrade-package=mlflow`, `--upgrade-package=pypdf llm`) takes the fixes.
- **pip's vendored copies** of urllib3 (2), msgpack (2) and setuptools: 5 HIGH in every image, fixed
  when pip ships a release with newer vendored libraries. A CRITICAL published on 2026-10-20 in
  pip 26.2.1's vendored certifi (GHSA-6q48-6c7w-9gw2, compromised emSign intermediate CAs) is fixed
  by pip 26.2.2, which the locks now pin; images built before that need a rebuild.
- **nltk 3.10.3:** two HIGH (cpu, llm, gpu): CVE-2026-81726 without a fix, and GHSA-j456-xh4h-cpf2
  (published 2026-10-27), fixed in nltk 3.10.4 after the images were locked; the next lock refresh
  (`tools/lock.sh --upgrade-package=nltk`) takes it.
- **llama-index-core 0.12:** one HIGH in the llm images, fixed only in 0.13, which needs a port of
  the LLM/RAG modules (they use APIs removed in 0.13).
- **Spark (in `cpu`):** 26 HIGH in its bundled jars. 23 are Jackson, Jetty, jline and Guava copies that
  Hadoop, Parquet and Spark shade into their own artefacts (hadoop-client-runtime, parquet-jackson,
  spark-network-common, ...), fixed only by Spark/Hadoop releases. 3 are in the standalone
  guava-33.6.0-jre.jar (fixed in 33.7.2) and snappy-java-1.1.10.8.jar (fixed in 1.1.10.9), published
  after the hash-pinned overrides in `docker/spark-jar-overrides.txt` were chosen.

No image has a CRITICAL finding, a secret, or a finding on an end-of-life OS.

## Verification

For every image (`tools/smoke-test.sh`):
- security invariants: no setuid/setgid files, compilers, headers, package manager, Node.js, ssh,
  su/sudo, curl/wget/openssl/perl; one Python; read-only `/etc/passwd` and image content; no
  private key; no world-writable files; time zone database present; every ELF dependency resolves;
- every `app/model` module imported; none of the modules of the 5.2.5 test images regressed;
- functional checks: torch and TensorFlow compute; the ESCU pretrained models (the 3 torch models
  and the DGA notebook end to end, which failed in every 5.2.5 test image); spaCy 3 with its model;
  a Spark SQL/FPGrowth/ALS job; the GPU stack: one CUDA 12.9 wheel set without CUDA 13 packages in
  `gpu` and `rapids`, PyTorch architectures sm_75 to sm_120, ptxas and libdevice 12.9 for XLA,
  TensorFlow followed by bitsandbytes/triton in one process, the bitsandbytes library it loads, a
  random-weight TimesFM checkpoint load and forecast; cuDF import; the agentic application;
  transformers in `cpu` and `gpu`: BERT and T5 with their tokenizers loaded from a local directory
  (as the assistants load their base models), MeCab with ipadic, Janome and ROUGE;
- an unknown host name fails as "not known" (NSS modules present), and every shipped notebook is
  valid nbformat (JupyterLab's post-save export needs that);
- PROD start (HTTPS API on 5000) as the default user, and as an arbitrary UID with a read-only root
  filesystem on a volume whose TLS files belong to another UID; DEV start with JupyterLab (8888),
  MLflow UI (6000) and TensorBoard (6006), a notebook kernel started through JupyterLab (login,
  websocket) running a cell, and the MLflow API serving its store; the same DEV start on a volume as
  5.2.x left it (MLflow file store with a run, which is migrated to SQLite, and an existing
  `~/.ipython`); no DEV service may exit.
- `agentic` only (`tools/agentic-security-test.sh`, against a fake OpenAI-compatible LLM, a fake
  Splunk MCP server and a blackhole that accepts connections and never answers): `/mcp/status`
  returns no client configuration, and only the configured LLM is available (`is_configured`
  "true" is, "false" and 0 are not); a wrong token gets 401 on all chat routes, and no token gets
  401 with `DSDL_AGENTIC_REQUIRE_TOKEN=true`; a two-turn chat runs the MCP tool once with the
  configured MCP token; a header that only contains the api token is refused (`validate_token`
  compares the whole header in every image); the replies "what about yesterday?", "yes, but only for index=main" and "y"
  do not approve it, a "yes" after switching the LLM option does not run the other option's pending
  call, and `/logReview` drops a pending approval, without leaving a checkpoint behind; a failing
  tool call is answered and the session goes on, and so is a tool call during which the MCP server
  dies (`DSDL_AGENTIC_TOOL_TIMEOUT`); a session answers one message at a time, and a turn over
  `DSDL_AGENTIC_TURN_TIMEOUT` ends with a short answer; user and session names cannot escape `/srv/logs`; `/logReview` reads at most
  5000 events, and `/logReview` and `/chatbox` bodies over their limit, also chunked, get a short
  answer; no CORS headers; one start-up probe; a probe without answer ends after its timeout and
  keeps tool calling; an evicted session (`MAX_SESSIONS`) loses its pending approval and graph
  checkpoint; while MCP and LLM endpoints never answer, the DSDL API and `/mcp/status` answer at
  once and a chat after about 20 s; while the LLM probe hangs, the DSDL API and `/mcp/status`
  answer at once; without `llm_config` the chat says so; an MCP server that is down at start-up is
  connected once it is up, and reconnected after it restarts and after it stopped answering for a
  while (paused: two failed pings, reported disconnected meanwhile), with the tool working each time
  and an approval from before the reconnect not resumed; SIGTERM shuts the API down within seconds and
  without errors with an open MCP session (closing it), during the MCP connect, during the LLM
  probe and while the start-up waits. In mutation runs, removing any of 14 protections in turn
  (token, approval, eviction, body limits, supervisor, timeouts, ...) makes the suite fail.
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
- **amd64 Jupyter kernels:** amd64 was smoke-tested under emulation on Apple silicon, where Jupyter
  kernels hang (Rosetta deadlock, also with the 5.2.x images), so the kernel check reports SKIP
  there; on arm64 it passed for every image. Run the kernel check on an amd64 host before release.
- **Playwright UI tests** (`testing/`, against a Splunk instance) and calls to real LLM providers
  were not run. Images were not pushed or signed.
- **Known application issues** unchanged by this branch: `ServiceContext` (llama-index 0.12) fails
  at fit time in `RAG_Encoding`, `RAG_Inferencing` and `llm_rag_script_saas_llm`;
  `timesfm_multi_resolution` imports a missing name. The agentic chat routes require the api token
  only with `DSDL_AGENTIC_REQUIRE_TOKEN=true`, which the DSDL app sets from 5.2.5 on (it also sends the
  token); started by an older DSDL app, the chat accepts requests without a token. Chat sessions
  are keyed by the client's session id only, not bound to a user. The agentic image runs one API
  worker, and `/fit` and `/apply` run on its event loop, so a long fit on that container holds the
  chat until it finishes.

## Reproduce

```
PLATFORM=all ./build.sh all splunk/ 5.2.5          # the release images, every declared platform
DSDL_TEST_PLATFORM=linux/amd64 tools/smoke-test.sh splunk/mltk-container-golden-cpu:5.2.5 cpu
tools/scan.sh --gate --platform linux/amd64,linux/arm64 splunk/mltk-container-golden-cpu:5.2.5
./build.sh --push-only --dry-run all splunk/ 5.2.5  # check the tested local images for a push
./build.sh --push-only all splunk/ 5.2.5            # push them as tested (no rebuild, no attestations)
./build.sh --push-rebuild all splunk/ 5.2.5         # or: rebuild and push with SBOM and provenance
./build.sh golden-cpu local/ 5.2.5                  # one target, one platform
tools/smoke-test.sh local/mltk-container-golden-cpu:5.2.5 cpu
tools/scan.sh --gate local/mltk-container-golden-cpu:5.2.5
tools/lock.sh --check                               # locks and cpu-constraint blocks are current
```

Build the images one at a time: a GPU or RAPIDS build needs about 30 to 40 GB of Docker disk,
including the build cache (`docker builder prune` frees it afterwards).
