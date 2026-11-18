#!/usr/bin/env bash
# Smoke-test a built DSDL image without a Splunk instance.
#
# Usage: tools/smoke-test.sh <image-ref> <target> [--quick]
#
#   1. security invariants inside the image (tests/smoke/image_checks.sh)
#   2. every app/model module imports; nothing in tests/expected/<target>.txt regresses
#   3. PROD start: HTTPS API on 5000 answers, as the default user and as an arbitrary
#      UID with a read-only root filesystem (OpenShift style)
#   4. functional checks per target (tests/smoke/functional.py): compute with torch/TF,
#      ESCU models load, local Spark job, CUDA build of the GPU stack, ...
#   5. DEV start: JupyterLab (8888), MLflow UI (6000) and TensorBoard (6006) answer, a
#      notebook kernel started through JupyterLab runs a cell (tests/smoke/jupyter_kernel.py)
#      and the MLflow API serves its store; on a fresh volume and on a volume as 5.2.x left
#      it (MLflow file store in /srv/mlruns, ~/.ipython from earlier kernels)
#   6. agentic only: chat routes against fake LLM and MCP services (tools/agentic-security-test.sh)
# --quick runs only 1 and 3. DSDL_TEST_PLATFORM (e.g. linux/amd64) tests that platform of a
# multi-platform local image (PLATFORM=all builds); default: the platform docker picks.
set -Eeuo pipefail

cd "$(dirname "$0")/.."
image="${1:?image reference}"
target="${2:?target name (cpu, llm, agentic, gpu, rapids)}"
quick="${3:-}"
if [ ! -f "tests/expected/$target.txt" ]; then
  echo "unknown target '$target' (cpu, llm, agentic, gpu, rapids)" >&2
  exit 2
fi
platform="${DSDL_TEST_PLATFORM:-$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$image")}"
engine="linux/$(docker version --format '{{.Server.Arch}}' 2>/dev/null)"
emulated=0
[ "$platform" != "$engine" ] && emulated=1  # e.g. amd64 under Rosetta or QEMU on Apple silicon
echo "Testing $image ($target) on $platform$([ $emulated = 1 ] && echo " (emulated on $engine)")"
run=(docker run --rm -i --platform "$platform" --network none)
failed=0
step() { echo; echo "== $*"; }

step "security invariants"
"${run[@]}" --entrypoint /bin/bash "$image" -s < tests/smoke/image_checks.sh || failed=1

if [ "$quick" != "--quick" ]; then
  step "import matrix (expected: tests/expected/$target.txt)"
  "${run[@]}" -e EXPECTED="$(cat "tests/expected/$target.txt")" --entrypoint /opt/dltk/venv/bin/python \
    "$image" - < tests/smoke/import_models.py || failed=1
fi

if [ "$quick" != "--quick" ]; then
  step "functional checks ($target)"
  "${run[@]}" --entrypoint /opt/dltk/venv/bin/python "$image" - "$target" < tests/smoke/functional.py || failed=1
fi

wait_for() { # container port scheme
  local container="$1" port="$2" scheme="$3"
  for _ in $(seq 1 90); do
    if docker exec "$container" python -c "
import ssl, sys, urllib.request
ctx = ssl._create_unverified_context()
try:
    urllib.request.urlopen('$scheme://127.0.0.1:$port/', context=ctx, timeout=5)
except urllib.error.HTTPError:
    pass  # any HTTP answer means the service is up
" >/dev/null 2>&1; then return 0; fi
    if [ "$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null)" != "true" ]; then return 1; fi
    sleep 4
  done
  return 1
}

start_and_probe() { # name description docker-args... -- port:scheme...
  local name="$1" description="$2"; shift 2
  local args=() probes=()
  while [ "$1" != "--" ]; do args+=("$1"); shift; done; shift
  probes=("$@")
  step "$description"
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker volume rm "$name-srv" >/dev/null 2>&1 || true
  if [ -n "${SEED:-}" ]; then # prepare the volume as an earlier release left it
    docker run --rm --platform "$platform" --network none -v "$name-srv:/srv" \
      --entrypoint /opt/dltk/venv/bin/python "$image" -c "$SEED" >/dev/null 2>&1 \
      || { echo "FAIL preparing the volume of $name"; failed=1; }
  fi
  docker run -d --platform "$platform" --name "$name" -v "$name-srv:/srv" "${args[@]}" "$image" >/dev/null
  local ok=1 probe
  for probe in "${probes[@]}"; do
    if wait_for "$name" "${probe%%:*}" "${probe##*:}"; then echo "PASS port ${probe%%:*} answers (${probe##*:})"
    else echo "FAIL port ${probe%%:*} did not answer (${probe##*:})"; ok=0; fi
  done
  if docker exec "$name" python -c "
import sys
site = next(i for i, p in enumerate(sys.path) if p.endswith('site-packages') and p.startswith('/opt/dltk/venv'))
user = sys.path.index('/srv/.dsdl/site-packages')
sys.exit(0 if user > site else 1)" 2>/dev/null; then echo "PASS user package dir is searched after the image site-packages"
  else echo "FAIL user package dir missing from sys.path or ahead of the image site-packages"; ok=0; fi
  if [ -n "${KERNEL_CHECK:-}" ]; then
    docker exec -i -e SMOKE_JUPYTER_PASSWORD="$jupyter_password" -e SMOKE_EMULATED="$emulated" "$name" /opt/dltk/venv/bin/python - \
      < tests/smoke/jupyter_kernel.py || ok=0
  fi
  if [ -n "${MLFLOW_EXPERIMENT:-}" ]; then
    if docker exec "$name" python -c "
import json, urllib.request
url = 'http://127.0.0.1:6000/api/2.0/mlflow/experiments/get-by-name?experiment_name=$MLFLOW_EXPERIMENT'
print(json.load(urllib.request.urlopen(url, timeout=30))['experiment']['name'])" >/dev/null 2>&1
    then echo "PASS MLflow serves experiment '$MLFLOW_EXPERIMENT'"
    else echo "FAIL MLflow does not serve experiment '$MLFLOW_EXPERIMENT'"; ok=0; fi
  fi
  # read the log once: with pipefail, `docker logs | grep -q` fails when grep stops early on a match
  local container_log
  container_log="$(docker logs "$name" 2>&1)"
  if [ -n "${EXPECT_LOG:-}" ]; then # one expected line per line of EXPECT_LOG
    local expected
    while IFS= read -r expected; do
      if grep -qF "$expected" <<< "$container_log"; then echo "PASS log shows: $expected"
      else echo "FAIL log does not show: $expected"; ok=0; fi
    done <<< "$EXPECT_LOG"
  fi
  if grep -qE '^\[dsdl\] warning: .* exited with status' <<< "$container_log"; then
    echo "FAIL a service exited: $(grep -E '^\[dsdl\] warning: .* exited with status' <<< "$container_log" \
      | head -3 | tr '\n' ' ')"; ok=0
  fi
  if [ "$ok" = 0 ]; then docker logs "$name" 2>&1 | tail -30; failed=1; fi
  docker rm -f "$name" >/dev/null
  docker volume rm "$name-srv" >/dev/null
}

# DSDL passes JUPYTER_PASSWD as a password hash (sha512:salt:digest); users log in with the plain password
jupyter_password=smoke-test-password
jupyter_hash="$("${run[@]}" --entrypoint /opt/dltk/venv/bin/python "$image" \
  -c "from jupyter_server.auth.security import passwd; print(passwd('$jupyter_password', 'sha512'))")"
start_and_probe dsdl-smoke-prod "PROD start (default user)" -e api_token=smoke-token -- 5000:https
# the volume was used by another UID before: its TLS directory and key are not readable
other_uid_tls='
import os
os.umask(0o002)  # as the entrypoint; earlier images then created the TLS directory with 0700
os.makedirs("/srv/.dsdl/site-packages")
os.makedirs("/srv/.dsdl/tls")
for name in ("dltk.pem", "dltk.key"):
    fd = os.open("/srv/.dsdl/tls/" + name, os.O_WRONLY | os.O_CREAT, 0o600)
    os.write(fd, b"written by another UID\n")
    os.close(fd)
os.chmod("/srv/.dsdl/tls", 0o700)
'
SEED="$other_uid_tls" EXPECT_LOG="using a certificate for this container only" start_and_probe dsdl-smoke-uid \
  "PROD start (arbitrary UID, read-only root filesystem, TLS files of another UID)" \
  --user 1000660000:0 --read-only --tmpfs /tmp -e api_token=smoke-token -- 5000:https
# a volume as 5.2.x left it: experiments in the MLflow file store /srv/mlruns, ~/.ipython from
# earlier kernels (jupyter_core then tries its one-time config migration), and version.md with
# this image's own version, as the 5.2.5 test images wrote "5.2.5", plus a shipped notebook
# edited in place: it must be backed up although the number matches
legacy_volume='
import os, shutil
os.makedirs("/srv/notebooks", exist_ok=True)
shutil.copyfile("/dltk/notebooks/barebone_template.ipynb", "/srv/notebooks/barebone_template.ipynb")
with open("/srv/notebooks/barebone_template.ipynb", "a") as notebook:
    notebook.write(" ")
with open("/srv/version.md", "w") as version:
    version.write(os.environ["DSDL_VERSION"] + "\n")
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import mlflow
mlflow.set_tracking_uri("file:///srv/mlruns")
mlflow.set_experiment("dsdl-smoke-legacy")
with mlflow.start_run():
    mlflow.log_metric("m", 1.0)
os.makedirs("/srv/.dsdl-home/.ipython/profile_default", exist_ok=True)
'
if [ "$quick" != "--quick" ]; then
  KERNEL_CHECK=1 MLFLOW_EXPERIMENT=Default start_and_probe dsdl-smoke-dev \
    "DEV start (JupyterLab, MLflow UI, TensorBoard)" \
    -e MODE_DEV_PROD=DEV -e JUPYTER_PASSWD="$jupyter_hash" -e api_token=smoke-token \
    -- 5000:https 8888:https 6000:http 6006:http
  SEED="$legacy_volume" KERNEL_CHECK=1 MLFLOW_EXPERIMENT=dsdl-smoke-legacy \
    EXPECT_LOG="MLflow: migrated the 5.2.x file store
Backed up 1 notebook files from an earlier" start_and_probe dsdl-smoke-dev-legacy \
    "DEV start on a 5.2.x volume (MLflow file store, existing ~/.ipython)" \
    -e MODE_DEV_PROD=DEV -e JUPYTER_PASSWD="$jupyter_hash" -e api_token=smoke-token \
    -- 5000:https 8888:https 6000:http 6006:http
fi

# agentic image: chat routes against fake LLM and MCP services (token check, no config leak, no
# path traversal, no CORS, the DSDL API not blocked by unreachable endpoints)
if [ "$target" = agentic ] && [ "$quick" != "--quick" ]; then
  step "agentic security checks (tools/agentic-security-test.sh)"
  if security_log="$(tools/agentic-security-test.sh "$image" 2>&1)"; then
    grep -E '^(== |PASS|FAIL)' <<< "$security_log"
  else
    echo "$security_log"; failed=1  # with the API log tail the test prints on a failure
  fi
fi

echo
if [ "$failed" = 0 ]; then echo "SMOKE TEST PASSED: $image ($target)"; else echo "SMOKE TEST FAILED: $image ($target)"; fi
exit "$failed"
