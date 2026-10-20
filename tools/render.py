#!/usr/bin/env python3
"""Render build and app configuration from images.toml.

  tools/render.py                    write docker-bake.json
  tools/render.py --check            fail if docker-bake.json is out of date
  tools/render.py --images-conf      write images_conf_files/images.conf (uses --repo/--version);
                                     --only T,S limits it to the given targets and stanzas
  tools/render.py --resolve NAME     print the build target for a target or legacy stanza name
  tools/render.py --targets          print all build target names
  tools/render.py --platforms NAME   print the platforms to build for a target (all of its
                                     platforms) or a 5.2.x stanza name (its published platform)
  tools/render.py --stanzas          print all published stanza ids
  tools/render.py --references       print every published image reference (uses --repo/--version)
  tools/render.py --title STANZA     print the images.conf title of a stanza (uses --version)
  tools/render.py --alias-ref NAME   print the reference of a single-platform 5.2.x name
                                     (e.g. golden-cpu-arm), nothing for other names
  tools/render.py --aliases          print "<target> <platform> <alias> <source>" lines for the
                                     single-platform aliases created after a push

Needs Python 3.11+ (tomllib) or the tomli package; build.sh runs it inside the
builder image otherwise.
"""

import argparse
import json
import pathlib
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11: pip install tomli
    import tomli as tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "images.toml"
BAKE = ROOT / "docker-bake.json"
IMAGES_CONF = ROOT / "images_conf_files" / "images.conf"


def load():
    with MANIFEST.open("rb") as handle:
        return tomllib.load(handle)


def references(manifest, repo, version):
    """Yield (target, entry, full reference) for every published name."""
    for name, target in manifest["targets"].items():
        for entry in target["publish"]:
            tag = entry["tag"].format(version=version)
            yield name, entry, f"{repo}{entry['image']}:{tag}"


def bake(manifest):
    targets = {}
    for name, target in manifest["targets"].items():
        targets[name] = {
            "context": ".",
            "dockerfile": "docker/Dockerfile",
            "target": name,
            "platforms": target["platforms"],
            "args": {
                "PYTHON_VERSION": "${DSDL_PYTHON_VERSION}",
                "DSDL_VERSION": "${VERSION}",
                "VCS_REF": "${VCS_REF}",
            },
            "labels": {"org.opencontainers.image.description": target["description"]},
            # multi-platform names only; single-platform aliases (e.g. -arm64) are
            # created from the pushed index by tools/release.sh
            "tags": sorted({
                "${REPO}" + entry["image"] + ":" + entry["tag"].replace("{version}", "${VERSION}")
                for entry in target["publish"] if "platform" not in entry
            }),
        }
    return {
        "variable": {
            "REPO": {"default": manifest["repo"]},
            "VERSION": {"default": manifest["version"]},
            "DSDL_PYTHON_VERSION": {"default": manifest["python"]},
            "VCS_REF": {"default": "unknown"},
        },
        "group": {"default": {"targets": list(manifest["targets"])}},
        "target": targets,
    }


def images_conf(manifest, repo, version, only=None):
    lines = []
    for target, entry, _ in references(manifest, repo, version):
        if only and target not in only and entry["stanza"] not in only:
            continue
        lines += [
            f"[{entry['stanza']}]",
            f"title = {entry['title'].format(version=version)}",
            f"image = {entry['image']}:{entry['tag'].format(version=version)}",
            f"repo = {repo}",
            f"runtime = {entry['runtime']}",
            "",
        ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--images-conf", action="store_true")
    parser.add_argument("--resolve")
    parser.add_argument("--targets", action="store_true")
    parser.add_argument("--platforms")
    parser.add_argument("--stanzas", action="store_true")
    parser.add_argument("--alias-ref")
    parser.add_argument("--only", help="comma-separated targets and/or stanzas for --images-conf")
    parser.add_argument("--references", action="store_true")
    parser.add_argument("--title")
    parser.add_argument("--aliases", action="store_true")
    parser.add_argument("--repo")
    parser.add_argument("--version")
    args = parser.parse_args()

    manifest = load()
    repo = args.repo if args.repo is not None else manifest["repo"]
    version = args.version or manifest["version"]

    if args.targets:
        print("\n".join(manifest["targets"]))
        return 0
    if args.references:
        print("\n".join(ref for _, _, ref in references(manifest, repo, version)))
        return 0
    if args.alias_ref:
        for _, entry, ref in references(manifest, repo, version):
            if args.alias_ref == entry["stanza"] and "platform" in entry:
                print(ref)
        return 0
    if args.stanzas:
        print("\n".join(entry["stanza"] for _, entry, _ in references(manifest, repo, version)))
        return 0
    if args.title:
        for _, entry, _ in references(manifest, repo, version):
            if entry["stanza"] == args.title:
                print(entry["title"].format(version=version))
                return 0
        print(f"unknown stanza: {args.title}", file=sys.stderr)
        return 2
    if args.platforms:
        if args.platforms in manifest["targets"]:
            print(",".join(manifest["targets"][args.platforms]["platforms"]))
            return 0
        for _, entry, _ in references(manifest, repo, version):
            if args.platforms in (entry["stanza"], entry["image"].removeprefix("mltk-container-")):
                print(entry.get("platform", "linux/amd64"))
                return 0
        print(f"unknown target or stanza: {args.platforms}", file=sys.stderr)
        return 2
    if args.resolve:
        for name, entry, _ in references(manifest, repo, version):
            if args.resolve in (name, entry["stanza"], entry["image"].removeprefix("mltk-container-")):
                print(name)
                return 0
        print(f"unknown target or stanza: {args.resolve}", file=sys.stderr)
        return 2
    if args.aliases:
        refs = list(references(manifest, repo, version))
        for name, entry, ref in refs:
            if "platform" in entry:
                # the multi-platform reference of the same image repository is the source
                source = next(r for n, e, r in refs if n == name and "platform" not in e and e["image"] == entry["image"])
                print(name, entry["platform"], ref, source)
        return 0
    if args.images_conf:
        IMAGES_CONF.parent.mkdir(exist_ok=True)
        only = {item for item in (args.only or "").split(",") if item}
        IMAGES_CONF.write_text(images_conf(manifest, repo, version, only))
        print(f"wrote {IMAGES_CONF.relative_to(ROOT)}")
        return 0

    rendered = json.dumps(bake(manifest), indent=2) + "\n"
    if args.check:
        if not BAKE.exists() or BAKE.read_text() != rendered:
            print("docker-bake.json is out of date; run tools/render.py", file=sys.stderr)
            return 1
        return 0
    BAKE.write_text(rendered)
    print(f"wrote {BAKE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
