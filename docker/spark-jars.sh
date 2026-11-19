#!/bin/bash
# Trim and patch the Spark jars bundled with pyspark (runs in the builder stage).
#   spark-jars-trim.txt       components DSDL's local-mode Spark never loads
#   spark-jar-overrides.txt   hash-pinned patch releases of bundled libraries
set -Eeuo pipefail
venv=/opt/dltk/venv
jars="$(echo "$venv"/lib/python3.*/site-packages/pyspark/jars)"
policy="${1:-/src/spark}"
cd "$jars"

sed -e 's/#.*//' -e '/^[[:space:]]*$/d' "$policy/spark-jars-trim.txt" | while read -r glob; do
  case "$glob" in
    bin/*) rm -f ../$glob "$venv"/$glob ;;   # launcher: pyspark/bin and the venv bin/
    *) rm -rf $glob ;;
  esac
done

mirrors=(https://repo1.maven.org/maven2 https://maven-central.storage-download.googleapis.com/maven2)

# Maven Central answers 429 to shared CI runners for minutes at a time; the sha256 pin makes any mirror safe.
fetch() {
  local path="$1" sha256="$2" out="$3" attempt mirror delay=10
  for attempt in 1 2 3 4 5 6 7 8; do
    for mirror in "${mirrors[@]}"; do
      if curl -fsSL --retry 2 --retry-delay 5 -o "$out" "$mirror/$path" \
        && echo "$sha256  $out" | sha256sum -c --quiet -; then
        return 0
      fi
      rm -f "$out"
      echo "spark-jar-overrides: $mirror/$path failed (attempt $attempt)" >&2
    done
    sleep "$delay"
    delay=$(( delay * 2 > 120 ? 120 : delay * 2 ))
  done
  return 1
}

sed -e 's/#.*//' -e '/^[[:space:]]*$/d' "$policy/spark-jar-overrides.txt" | while read -r glob path sha256; do
  old=( $glob )
  if [ ! -e "${old[0]}" ]; then
    echo "spark-jar-overrides: no jar matches $glob (update the list for this Spark release)" >&2
    exit 1
  fi
  new="$(basename "$path")"
  fetch "$path" "$sha256" "$new.part"
  rm -f "${old[@]}"
  mv "$new.part" "$new"
  chmod 0644 "$new"
  echo "replaced ${old[*]} -> $new"
done
