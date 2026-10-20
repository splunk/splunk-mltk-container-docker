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

sed -e 's/#.*//' -e '/^[[:space:]]*$/d' "$policy/spark-jar-overrides.txt" | while read -r glob path sha256; do
  old=( $glob )
  if [ ! -e "${old[0]}" ]; then
    echo "spark-jar-overrides: no jar matches $glob (update the list for this Spark release)" >&2
    exit 1
  fi
  new="$(basename "$path")"
  curl -fsSL --retry 3 -o "$new.part" "https://repo1.maven.org/maven2/$path"
  echo "$sha256  $new.part" | sha256sum -c --quiet -
  rm -f "${old[@]}"
  mv "$new.part" "$new"
  chmod 0644 "$new"
  echo "replaced ${old[*]} -> $new"
done
