#!/bin/bash
# Run the Playwright UI tests for every published stanza in images.toml.
# Usage: ./bulk_test_containers.sh [version]
cd "$(dirname "$0")"
log_file="./testing/output/$(date +%s)-bulktest.log"
echo "Running a bulk test for all stanzas in images.toml. See results in ${log_file}"
python=./testing/venv/bin/python
[ -x "$python" ] || python=python3
stanzas="$("$python" tools/render.py --stanzas)" || { echo "tools/render.py needs Python 3.11+ or tomli (run ./test_container.sh once to create testing/venv)" >&2; exit 1; }
rc=0
for stanza in $stanzas; do
  echo "Testing: ${stanza} ----------------------------------------------" 2>&1 | tee -a "$log_file"
  ./test_container.sh "$stanza" "$1" 2>&1 | tee -a "$log_file"
  [ "${PIPESTATUS[0]}" = 0 ] || rc=1
done
exit "$rc"
