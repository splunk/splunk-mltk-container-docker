# Releasing the DSDL container images

A release builds the five images for a version, pushes them to Docker Hub (`splunk/`), signs every pushed image and checks the signatures. It runs in the GitLab mirror of this repository and is started by a version tag. Nobody builds, pushes or signs by hand.

For what a build produces and how to build locally, see `BUILD.md` and `./build.sh`.

## What a release publishes

For version `X.Y.Z`, the images listed in `images.toml`:

| Target | Image and tag | Platforms |
| --- | --- | --- |
| `cpu` | `splunk/mltk-container-golden-cpu:X.Y.Z` | amd64, arm64 |
| `llm` | `splunk/mltk-container-ubi-llm-rag:X.Y.Z` | amd64, arm64 |
| `agentic` | `splunk/mltk-container-ubi-llm-rag:agentic-ai-X.Y.Z` | amd64, arm64 |
| `gpu` | `splunk/mltk-container-golden-gpu:X.Y.Z` | amd64 |
| `rapids` | `splunk/mltk-container-golden-gpu-rapids:X.Y.Z` | amd64 |

Pushing a version that already exists on Docker Hub replaces that tag. Each image gets a new digest and the pipeline signs the new digest.

## Trigger a release

1. Merge the release branch (for example `v5.2.5-dev`) into `master` on GitHub. The tagged commit must contain `.gitlab-ci.yml`, `images.toml`, `build.sh` and `docker/`.
2. Tag the commit on `master` with `vX.Y.Z` and push the tag to GitHub. The name must match `vX.Y.Z` or `vX.Y.Z.W`. The version of the images is the tag without the `v`.
3. The GitLab mirror pulls the tag. The tag is protected there (`v*`, Maintainers), and a protected tag starts the pipeline. If no pipeline appears after the mirror has synced, run the pipeline manually on the tag from the project's CI/CD page. No variables are needed.
4. Watch the three jobs. The build is the long one (up to 3 hours, two platforms emulated with QEMU).
5. When all three pass, the tags are on Docker Hub and signed.

Only a protected tag publishes to `splunk/`. The repository and the version are taken from the tag, and any `IMAGE_REPO` or `RELEASE_VERSION` that is passed in is ignored.

## The jobs

| Job | What it does |
| --- | --- |
| `build-and-push` | Logs in to Docker Hub and runs `./build.sh --push-rebuild all splunk/ X.Y.Z`. It writes the pushed references and their digests to `digests.txt`. |
| `sign` | Signs each reference with `artifact-ci sign docker <ref> -r docker.io --upload-to-tlog`. |
| `validate` | Checks that each tag still points to the digest that was built, then runs `artifact-ci verify docker <ref>`. |

Jobs run in this order, and a failed job stops the later ones. `resource_group: mltk-release` keeps two releases from running at the same time.

## Re-run after a failure

Retry only the failed job from the pipeline page. `sign` and `validate` read `digests.txt` from the `build-and-push` job, so a retry of either does not rebuild anything.

| Symptom | Cause | What to do |
| --- | --- | --- |
| `build-and-push` fails with HTTP 429 from `repo1.maven.org` | Rate limiting on the shared runner while `docker/spark-jars.sh` downloads jars | Retry the job. `spark-jars.sh` waits and retries before it gives up. |
| Checkout fails on Git LFS objects | The mirror does not hold every LFS object. The `basemodels` files are not used by the build | Already handled: the pipeline sets `GIT_LFS_SKIP_SMUDGE: "1"`. |
| `docker login` fails | `DOCKERHUB_USER` or `DOCKERHUB_TOKEN` is wrong, expired or not available to the ref | Check the project variables. They are protected, so they only reach protected refs. |
| `sign` is denied | The ref is not protected, or the project's signing grant is missing | Check that the tag is protected and that the pipeline is running on it. |
| `validate` reports a changed digest | A tag was pushed to while the pipeline was running | Find who pushed. Then re-run the whole pipeline. |

## Run it manually without publishing to `splunk/`

Use this to test pipeline changes. From the project's CI/CD page choose **Run pipeline** on a protected branch and set:

- `RELEASE_VERSION`, for example `0.0.0-test`
- `IMAGE_REPO`, a Docker Hub `<namespace>/<prefix>` such as `myorg/test-`

Manual runs have no default repository and refuse `splunk` in any spelling. They create repositories in the namespace you give, and you remove them afterwards. The Docker Hub token must be able to write there.

## Configuration

Set in the project's CI/CD variables (protected):

| Variable | Used for |
| --- | --- |
| `BUILD_IMAGE` | Job image that has `docker`, `buildx` and `artifact-ci` |
| `DOCKERHUB_USER`, `DOCKERHUB_TOKEN` | Docker Hub login (the token is masked) |
| `SIGNING_SERVICE_ADDR` | Audience of the signing-service token |

`CICD_VAULT_ADDR` comes from the GitLab instance. Signing uses a Vault role granted to the project's protected refs.

## Verify a signature

The signature of a digest is stored as a tag in the same repository: `sha256-<digest>.sig`. Tags that share a digest share a signature.

With `artifact-ci`: `artifact-ci verify docker splunk/mltk-container-golden-cpu:X.Y.Z`.

Without it, using `curl` and `openssl`:

1. Get the digest of the tag, and a pull token from `auth.docker.io` for the repository.
2. Fetch the manifest of the tag `sha256-<hex digest>.sig` from `registry-1.docker.io`.
3. Take the `dev.cosignproject.cosign/signature` annotation of its layer and decode it from base64.
4. Download the layer blob. That is the signed payload, a JSON document that names the image digest.
5. Run `openssl dgst -sha256 -verify <public key> -signature <signature file> <payload file>`. It must print `Verified OK`, and the digest in the payload must equal the digest of the tag.

The public key is published in the Splunk Observability documentation (`gdi/opentelemetry/install-the-collector.rst` in `splunk/public-o11y-docs`, section "collector-verify-docker"). Use the key whose SHA-256 over the DER form starts with `e95c18590701d618`.
