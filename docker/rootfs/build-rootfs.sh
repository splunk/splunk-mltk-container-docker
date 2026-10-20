#!/bin/bash
# Assemble the DSDL runtime operating system into /rootfs (runs in the builder stage).
#
#   build-rootfs.sh base <python-version>   install packages.txt, apply trim.txt, check allow.txt
#   build-rootfs.sh jre                     add the JRE for the Spark image, write the delta to /delta
#
# The result has no package manager, no PAM/su/sudo, no ssh client, no compilers or
# headers, no setuid/setgid files and a read-only /etc/passwd.
set -Eeuo pipefail

policy=/policy
root=/rootfs

lines() { sed -e "s/@PY@/${PY:-}/g" -e 's/#.*//' -e '/^[[:space:]]*$/d' "$1"; }

dnf_root() {
  dnf -y --installroot="$root" --releasever=9 \
    --setopt=reposdir=/etc/yum.repos.d --setopt=install_weak_deps=False --setopt=tsflags=nodocs "$@"
}
dnf_install() { dnf_root install "$@"; }

# Every file an installed RPM declares must exist (documentation and licenses excepted):
# catches packages that ubi-micro registers but ships stripped.
payload_check() {
  local missing
  # rpm -V exits non-zero whenever it reports anything; only "missing" lines matter here
  # (/var/cache and /var/log are emptied on purpose by finish())
  missing="$({ rpm --root="$root" -Va --nodeps --nodigest --nosignature --noscripts 2>/dev/null || true; } \
    | awk '$1 == "missing" && $2 != "d" && $2 != "l" && $NF !~ /^\/var\/(cache|log)\// {print $NF}')"
  if [ -n "$missing" ]; then
    echo "Installed RPMs are missing files:" >&2
    echo "$missing" | head -20 >&2
    exit 1
  fi
}

# Remove packages with --nodeps; each entry in a trim list documents why it is safe.
trim() {
  local installed=()
  local name
  while read -r name; do
    if rpm --root="$root" -q "$name" >/dev/null 2>&1; then installed+=("$name"); fi
  done < <(lines "$1")
  if [ "${#installed[@]}" -gt 0 ]; then
    rpm --root="$root" -e --nodeps --noscripts "${installed[@]}"
  fi
}

finish() {
  find "$root" -xdev -type f -perm /6000 -exec chmod ug-s {} +
  dnf --installroot="$root" clean all >/dev/null
  rm -rf "$root"/var/cache/* "$root"/var/log/* "$root"/var/lib/dnf
}

case "${1:-}" in
  base)
    PY="${2:?python version}"
    # ubi-micro registers tzdata in its RPM database but ships it without /usr/share/zoneinfo;
    # drop the stub so packages.txt installs the full time zone database
    rpm --root="$root" -e --nodeps --noscripts tzdata
    # the RPMs ubi-micro pre-installs (glibc, bash, ...) get the current repository fixes too
    dnf_root upgrade
    dnf_install $(lines "$policy/packages.txt")
    trim "$policy/trim.txt"
    # one static account; arbitrary UIDs get an entry through nss_wrapper at start
    printf 'dltk:x:1001:0:dltk user:/srv/.dsdl-home:/bin/bash\n' >> "$root/etc/passwd"
    chmod 0644 "$root/etc/passwd" "$root/etc/group"
    # the runtime contains AppStream RPMs: declare those content sets so that
    # scanners match AppStream advisories (ubi-micro only declares BaseOS)
    for dir in usr/share/buildinfo root/buildinfo; do
      cp /usr/share/buildinfo/content-sets.json "$root/$dir/content-sets.json"
    done
    finish
    payload_check
    rpm --root="$root" -qa --qf '%{NAME}\n' | LC_ALL=C sort > /rootfs-packages.txt
    if [ "${ROOTFS_ALLOWLIST_CHECK:-1}" = "1" ] && ! diff -u "$policy/allow.txt" /rootfs-packages.txt; then
      echo "The runtime RPM set differs from docker/rootfs/allow.txt. Review the diff above;" >&2
      echo "if it is intended, update allow.txt (tools/update-rootfs-allowlist.sh)." >&2
      exit 1
    fi
    ;;
  jre)
    rpm --root="$root" -qa --qf '%{NAME}\n' | LC_ALL=C sort > /tmp/.jre-before
    touch /tmp/.jre-marker
    sleep 1
    dnf_install java-21-openjdk-headless
    trim "$policy/trim-jre.txt"
    finish
    payload_check
    rpm --root="$root" -qa --qf '%{NAME}\n' | LC_ALL=C sort > /tmp/.jre-after
    LC_ALL=C comm -13 /tmp/.jre-before /tmp/.jre-after > /rootfs-jre-packages.txt
    if [ "${ROOTFS_ALLOWLIST_CHECK:-1}" = "1" ] && ! diff -u "$policy/allow-jre.txt" /rootfs-jre-packages.txt; then
      echo "The JRE RPM set differs from docker/rootfs/allow-jre.txt. Review the diff above;" >&2
      echo "if it is intended, update allow-jre.txt (tools/update-rootfs-allowlist.sh)." >&2
      exit 1
    fi
    # The delta = files of the newly installed RPMs (they keep their packaged mtimes) plus
    # everything the scriptlets wrote (alternatives links, ldconfig cache, rpm database).
    {
      LC_ALL=C comm -13 /tmp/.jre-before /tmp/.jre-after | xargs rpm --root="$root" -ql | grep '^/'
      (cd "$root" && find . -xdev -newer /tmp/.jre-marker \( -type f -o -type l \) | sed 's#^\.##')
    } | LC_ALL=C sort -u | while read -r path; do
      if [ -e "$root$path" ] || [ -L "$root$path" ]; then printf '.%s\0' "$path"; fi
    done > /tmp/.jre-files
    mkdir -p /delta
    (cd "$root" && tar --null --no-recursion -T /tmp/.jre-files -cf - | tar -xf - -C /delta)
    echo "JRE delta: $(tr -cd '\0' < /tmp/.jre-files | wc -c) paths, $(du -sh /delta | cut -f1)"
    ;;
  *)
    sed -n '2,8p' "$0" >&2
    exit 2
    ;;
esac
