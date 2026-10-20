#!/usr/bin/env bash
# Scan DSDL images with Trivy: full JSON report for tracking plus a release gate.
#
# Usage: tools/scan.sh [--gate] <image-ref>...
#
# Reports go to scan_logs/<image>_<timestamp>.json with a one-line summary per image.
# --gate fails (exit 1) when an image
#   * runs on an end-of-life OS (scanners report no OS findings for those),
#   * has a CRITICAL finding with a fixed version available, or
#   * contains a secret or private key.
# HIGH findings with a fix are reported for the monthly rebuild/lock refresh.
#
# Secret findings in known sample data are allowed in tools/trivy-secret.yaml.
# TRIVY_IMAGE overrides the scanner image (pin it by digest in CI).
set -Eeuo pipefail

cd "$(dirname "$0")/.."
trivy_image="${TRIVY_IMAGE:-aquasec/trivy:0.69.3}"
gate=false
if [ "${1:-}" = "--gate" ]; then gate=true; shift; fi
if [ "$#" -eq 0 ]; then sed -n '2,13p' "$0"; exit 2; fi

mkdir -p scan_logs
failed=0
for image in "$@"; do
  report="scan_logs/$(echo "$image" | tr '/:' '__')_$(date +%Y%m%d%H%M%S).json"
  docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v dsdl-trivy-cache:/root/.cache/ \
    -v "$PWD/tools/trivy-secret.yaml:/trivy-secret.yaml:ro" \
    "$trivy_image" image --quiet --format json --scanners vuln,secret --secret-config /trivy-secret.yaml \
    --timeout 45m "$image" > "$report"
  if ! python3 - "$report" "$image" "$gate" <<'EOF'
import collections, json, sys
report, image, gate = sys.argv[1], sys.argv[2], sys.argv[3] == "true"
data = json.load(open(report))
os_info = data.get("Metadata", {}).get("OS", {})
vulns, secrets = [], []
for result in data.get("Results") or []:
    vulns += result.get("Vulnerabilities") or []
    secrets += result.get("Secrets") or []
severity = collections.Counter(v.get("Severity", "UNKNOWN") for v in vulns)
fixable = [v for v in vulns if v.get("FixedVersion")]
fixable_critical = [v for v in fixable if v.get("Severity") == "CRITICAL"]
fixable_high = [v for v in fixable if v.get("Severity") == "HIGH"]
print(f"{image}: {os_info.get('Family')} {os_info.get('Name')} | findings {len(vulns)} "
      f"(C {severity['CRITICAL']}, H {severity['HIGH']}, M {severity['MEDIUM']}, L {severity['LOW']}) | "
      f"fixable C {len(fixable_critical)}, H {len(fixable_high)} | secrets {len(secrets)} | {report}")
problems = []
if os_info.get("EOSL"):
    problems.append("end-of-life OS")
if fixable_critical:
    problems.append("fixable CRITICAL: " + ", ".join(sorted({f"{v['PkgName']} {v['InstalledVersion']}" for v in fixable_critical})))
if secrets:
    problems.append("secrets: " + ", ".join(sorted({s.get("RuleID", "?") for s in secrets})))
for problem in problems:
    print(f"  GATE: {problem}")
sys.exit(1 if gate and problems else 0)
EOF
  then
    failed=1
  fi
done
exit "$failed"
