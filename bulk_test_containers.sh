#!/bin/bash
# Run the Playwright UI tests for every published stanza in images.toml.
# Usage: ./bulk_test_containers.sh [version]
cd "$(dirname "$0")"
log_file="./testing/output/$(date +%s)-bulktest.log"
echo "Running a bulk test for all stanzas in images.toml. See results in ${log_file}"
for stanza in $(python3 tools/render.py --stanzas); do
  echo "Testing: ${stanza} ----------------------------------------------" 2>&1 | tee -a "$log_file"
  ./test_container.sh "$stanza" "$1" 2>&1 | tee -a "$log_file"
done
