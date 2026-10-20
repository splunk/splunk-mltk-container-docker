#!/bin/bash
# Security invariants of every DSDL runtime image (runs inside the container).
set -uo pipefail

fail=0
check() {
  local description="$1"; shift
  if "$@"; then echo "PASS $description"; else echo "FAIL $description"; fail=1; fi
}
absent() { ! command -v "$1" >/dev/null 2>&1; }
none_found() { [ -z "$("$@" 2>/dev/null | head -1)" ]; }

check "no setuid/setgid files" none_found find / -xdev -type f -perm /6000
for tool in gcc cc g++ c++ ld as make cpp gfortran node npm dnf microdnf yum rpm sudo su ssh scp openssl curl wget perl; do
  check "no $tool" absent "$tool"
done
# python3.12-libs ships pyconfig-64.h for sysconfig; any other header means a -devel package
check "no C headers" none_found find /usr/include -name '*.h' ! -name 'pyconfig*.h'
check "no kernel headers" test ! -d /usr/include/linux
check "exactly one python interpreter" test "$(ls /usr/bin/python3* /usr/local/bin/python3* 2>/dev/null | grep -Ec 'python3(\.[0-9]+)?$')" -le 2
check "/etc/passwd is not writable by group or others" test "$(stat -c %a /etc/passwd)" = 644
# key material files (library sources that merely contain the PEM marker are not keys)
check "no private key in the image" none_found grep -rlI --include='*.key' --include='*.pem' --include='*.p12' \
  --include='*.der' -e 'BEGIN [A-Z ]*PRIVATE KEY' /dltk /opt /etc /srv /usr/share
check "image content is read-only for the runtime user" bash -c '! touch /opt/dltk/venv/.w 2>/dev/null && ! touch /dltk/app/.w 2>/dev/null'
check "runs as uid 1001 / gid 0 by default" test "$(id -u):$(id -g)" = "1001:0"
check "time zone database present" test -f /usr/share/zoneinfo/America/New_York
check "no world-writable files outside /tmp" none_found find / -xdev ! -type l -perm -o+w ! -path '/tmp*' ! -path '/var/tmp*' ! -path '/proc/*' ! -path '/dev/*'
check "tar available (kubectl cp)" command -v tar

# every ELF dependency of the venv and the OS must resolve inside the image. OPTIONAL: libraries
# that are injected at run time (libcuda), multi-node fabrics, alternative accelerator backends,
# CuPy's optional cuTENSOR, and ALSA for Java Sound (libjsound; removed with the JRE trim)
python3 - <<'EOF' || fail=1
import os, struct, sys
OPTIONAL = ("libcuda.so", "libnvidia", "libibverbs", "librdmacm", "libfabric", "libucp", "libucs", "libuct",
            "libmpi", "liboshmem", "libpmix", "libcutensor", "libasound", "libnccl-net", "libamdhip", "libhip", "librocm", "libsycl", "libze_", "libOpenCL",
            "libcudart.so.11", "libcudart.so.13", "libcublas.so.11", "libcublas.so.13", "libcusparse.so.13",
            "libnvJitLink.so.13", "libnvrtc.so.13", "libtbb", "libjvm", "libmlx")
def needed(path):
    try:
        with open(path, "rb") as f:
            h = f.read(64)
            if h[:4] != b"\x7fELF" or h[4] != 2:
                return None
            le = "<" if h[5] == 1 else ">"
            shoff = struct.unpack_from(le + "Q", h, 0x28)[0]
            shentsize, shnum = struct.unpack_from(le + "HH", h, 0x3A)
            f.seek(shoff)
            sh = f.read(shentsize * shnum)
            secs = [struct.unpack_from(le + "IIQQQQIIQQ", sh, i * shentsize) for i in range(shnum)]
            dyn = [s for s in secs if s[1] == 6]
            if not dyn:
                return []
            strtab = secs[dyn[0][6]]
            f.seek(strtab[4]); st = f.read(strtab[5])
            f.seek(dyn[0][4]); data = f.read(dyn[0][5])
            out = []
            for i in range(0, len(data), 16):
                tag, val = struct.unpack_from(le + "qQ", data, i)
                if tag == 0:
                    break
                if tag == 1:
                    out.append(st[val:st.index(b"\0", val)].decode())
            return out
    except (OSError, ValueError, struct.error, IndexError):
        return None
roots = ["/opt/dltk/venv", "/usr/lib64", "/usr/bin", "/usr/libexec", "/usr/lib/jvm"]
present, missing = set(), {}
files = []
for root in roots:
    for dirpath, _, names in os.walk(root):
        for name in names:
            path = os.path.join(dirpath, name)
            present.add(name)
            if not os.path.islink(path) and (".so" in name or os.access(path, os.X_OK)):
                files.append(path)
for path in files:
    for lib in needed(path) or []:
        if lib not in present and not lib.startswith(OPTIONAL) and not lib.startswith("ld-linux"):
            missing.setdefault(lib, path)
for lib, path in sorted(missing.items()):
    print(f"  unresolved {lib} (needed by {path})")
print(("PASS" if not missing else "FAIL") + f" ELF dependencies resolve ({len(files)} files)")
sys.exit(1 if missing else 0)
EOF
exit "$fail"
