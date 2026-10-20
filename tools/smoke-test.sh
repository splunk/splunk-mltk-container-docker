#!/usr/bin/env bash
# Smoke-test a built DSDL image without a Splunk instance.
#
# Usage: tools/smoke-test.sh <image-ref> <target> [--quick]
#
#   1. security invariants inside the image (tests/smoke/image_checks.sh)
#   2. every app/model module imports; nothing in tests/expected/<target>.txt regresses
#   3. PROD start: HTTPS API on 5000 answers, as the default user and as an arbitrary
#      UID with a read-only root filesystem (OpenShift style)
#   4. DEV start: JupyterLab (8888), MLflow UI (6000) and TensorBoard (6006) answer
# --quick runs only 1 and 3.
set -Eeuo pipefail

cd "$(dirname "$0")/.."
image="${1:?image reference}"
target="${2:?target name (cpu, llm, agentic, gpu, rapids, spark)}"
quick="${3:-}"
platform="$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$image")"
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
  docker run -d --platform "$platform" --name "$name" -v "$name-srv:/srv" "${args[@]}" "$image" >/dev/null
  local ok=1 probe
  for probe in "${probes[@]}"; do
    if wait_for "$name" "${probe%%:*}" "${probe##*:}"; then echo "PASS port ${probe%%:*} answers (${probe##*:})"
    else echo "FAIL port ${probe%%:*} did not answer (${probe##*:})"; ok=0; fi
  done
  if [ "$ok" = 0 ]; then docker logs "$name" 2>&1 | tail -30; failed=1; fi
  docker rm -f "$name" >/dev/null
  docker volume rm "$name-srv" >/dev/null
}

start_and_probe dsdl-smoke-prod "PROD start (default user)" -e api_token=smoke-token -- 5000:https
start_and_probe dsdl-smoke-uid "PROD start (arbitrary UID, read-only root filesystem)" \
  --user 1000660000:0 --read-only --tmpfs /tmp -e api_token=smoke-token -- 5000:https
if [ "$quick" != "--quick" ]; then
  start_and_probe dsdl-smoke-dev "DEV start (JupyterLab, MLflow UI, TensorBoard)" \
    -e MODE_DEV_PROD=DEV -e JUPYTER_PASSWD=smoke-test-password -e api_token=smoke-token \
    -- 5000:https 8888:https 6000:http 6006:http
fi

echo
if [ "$failed" = 0 ]; then echo "SMOKE TEST PASSED: $image ($target)"; else echo "SMOKE TEST FAILED: $image ($target)"; fi
exit "$failed"
