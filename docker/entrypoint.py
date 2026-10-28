"""Start-up logic for every Splunk DSDL container image.

Replaces bootstrap_fast.sh, bootstrap_transformers.sh, bootstrap_rapids.sh,
bootstrap_spark.sh and bootstrap_backup.sh. Behaviour kept from 5.2.4 and the 5.2.5
test images (the splunk/mltk-container-*:5.2.5 test builds of September 2026):

* /srv is the persistent volume: on every start the image's /dltk/app is
  overlaid onto /srv/app (stale __pycache__ removed first) and /dltk/notebooks
  is copied into /srv/notebooks. When the image version differs from the one
  recorded in /srv/version.md, the shipped notebooks that this copy would
  overwrite and that differ from the image (e.g. edited in place) are backed up
  first.
* MODE_DEV_PROD=PROD (default) starts only the DSDL API (uvicorn on 5000).
  MODE_DEV_PROD=DEV also starts JupyterLab (8888), TensorBoard (6006) and the
  MLflow UI (6000).
* ENABLE_HTTPS (default true), api_workers and API_SSL_CERT / API_SSL_KEY are
  honoured as before.

New in the 5.2.5 release:

* No TLS private key is baked into the image. The certificate is taken from
  API_SSL_CERT / API_SSL_KEY, else from /dltk/.jupyter/dltk.{pem,key} if a
  derived image provides them, else a per-installation self-signed pair is
  generated into /srv/.dsdl/tls and renewed before it expires.
* /etc/passwd is read-only. Arbitrary UIDs (OpenShift) get a passwd entry, and a
  container hostname missing from /etc/hosts gets the loopback address, via
  nss_wrapper instead of editing /etc at runtime.
* DSDL_APP_MODULE selects the FastAPI application (app.main:app by default,
  app.main_agentic:app for the agentic-ai tags).
* The image content is read-only. Processes started here `pip install` into
  /srv/.dsdl/site-packages (persistent, PIP_TARGET); that path is searched after
  the image's site-packages, so it cannot shadow image packages.
* This script stays PID 1 in both modes: it forwards signals and reaps orphans.
* The MLflow UI gets an explicit store, /srv/mlflow.db (SQLite): a 5.2.x file
  store /srv/mlruns with data is migrated into it once (and kept as it was);
  MLflow 3 refuses an implicit file store. One worker, no job runner; MLflow's
  own host and CORS protection stays on (full access from localhost only;
  MLFLOW_SERVER_ALLOWED_HOSTS and MLFLOW_SERVER_CORS_ALLOWED_ORIGINS open it to
  the Docker host's name or address).
* /srv/version.md records the image version that last used the volume; a change
  in either direction triggers the backup above, into a new folder each time. So
  does the same version written by a 5.2.x bootstrap (the 5.2.5 test images also
  wrote "5.2.5"): only this script also writes /srv/.dsdl/image-version.
"""

import datetime
import filecmp
import os
import pwd
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

SRV = "/srv"
DLTK = "/dltk"
TLS_DIR = os.path.join(SRV, ".dsdl", "tls")
LEGACY_CERT = os.path.join(DLTK, ".jupyter", "dltk.pem")
LEGACY_KEY = os.path.join(DLTK, ".jupyter", "dltk.key")
NSS_WRAPPER_LIB = "/usr/lib64/libnss_wrapper.so"
# DEV-mode `pip install` target; searched after the image packages (zz-dsdl-user-packages.pth)
USER_PACKAGES = os.path.join(SRV, ".dsdl", "site-packages")
CERT_VALIDITY_DAYS = 365
CERT_RENEW_DAYS = 30


def log(message):
    print(f"[dsdl] {message}", flush=True)


def write_atomically(path, data, mode=0o664):
    """Write through a unique temporary file and a rename: readers never see a partial file,
    and containers that start at the same time on one volume do not share a temp file
    (the entrypoint is PID 1 in every container, so the PID is no unique suffix)."""
    fd, temporary = tempfile.mkstemp(dir=os.path.dirname(path), prefix=os.path.basename(path) + ".")
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def version_tuple(version):
    """Numeric comparison key: 5.2.10 > 5.2.5 (5.2.x compared strings)."""
    return tuple(int(part) for part in re.findall(r"\d+", version or ""))


def backup_notebooks(image_version):
    version_file = os.path.join(SRV, "version.md")
    # written only by this entrypoint: a version.md without it comes from a 5.2.x bootstrap
    marker = os.path.join(SRV, ".dsdl", "image-version")
    notebooks_dir = os.path.join(SRV, "notebooks")
    recorded = ""
    try:
        if os.path.isfile(version_file):
            with open(version_file, encoding="utf-8") as handle:
                recorded = handle.read().strip()
        earlier = recorded or "an earlier version"
        if recorded == image_version and os.path.isfile(marker):
            log(f"Notebook backup not needed (volume at {recorded}).")
            return
        if recorded == image_version:
            # the same number from a 5.2.x bootstrap: the 5.2.5 test images, a different image
            earlier = f"an earlier {recorded} build"
            log(f"The volume was last used by {earlier} (5.2.x start-up scripts).")
        if version_tuple(recorded) > version_tuple(image_version):
            log(f"warning: the volume was last used by DSDL {recorded}, newer than this image ({image_version})")
        if os.path.isdir(notebooks_dir) and os.listdir(notebooks_dir):
            # a new folder every time: an earlier (possibly partial) backup is never reused
            base = backup_dir = os.path.join(SRV, f"notebooks_backup_{image_version}")
            number = 1
            while os.path.lexists(backup_dir):
                number += 1
                backup_dir = f"{base}.{number}"
            try:
                shutil.copytree(notebooks_dir, backup_dir, symlinks=True, copy_function=shutil.copyfile,
                                ignore=kept_by_refresh(notebooks_dir, os.path.join(DLTK, "notebooks")))
                copied = remove_empty_dirs(backup_dir)
                if copied:
                    log(f"Backed up {copied} notebook files from {earlier} that this "
                        f"image replaces to {backup_dir}.")
            except shutil.Error as error:
                # recorded anyway, so a permanently unreadable file does not add a copy per start
                log(f"warning: notebook backup {backup_dir} is incomplete: {len(error.args[0])} files not copied")
        write_atomically(version_file, (image_version + "\n").encode())
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        write_atomically(marker, (image_version + "\n").encode())
    except OSError as error:
        log(f"warning: notebook backup failed: {error}")


def kept_by_refresh(root, image_root):
    """copytree ignore function: skip what refresh_srv() leaves alone (files and folders that the
    image does not ship) and shipped files that still equal the image copy, so a backup holds only
    the files the refresh is about to overwrite with different content (e.g. edited in place).
    User files, data and TensorBoard logs are never touched by the refresh and need no copy."""
    def ignore(directory, names):
        image_dir = os.path.join(image_root, os.path.relpath(directory, root))
        skipped = []
        for name in names:
            source, shipped = os.path.join(directory, name), os.path.join(image_dir, name)
            if os.path.isdir(source) and not os.path.islink(source):
                if not os.path.isdir(shipped):
                    skipped.append(name)
            elif not os.path.isfile(shipped) or filecmp.cmp(source, shipped, shallow=False):
                skipped.append(name)
        return skipped
    return ignore


def remove_empty_dirs(path):
    """Remove empty folders below path (and path itself if empty); return the number of files left."""
    files = 0
    for directory, subdirs, names in os.walk(path, topdown=False):
        files += len(names)
        if not names and not any(os.path.lexists(os.path.join(directory, sub)) for sub in subdirs):
            os.rmdir(directory)
    return files


def overlay(src, dst):
    """Copy src over dst like 5.2.x `cp -RT`: file contents only, no chmod/utime of
    existing directories (they may belong to another UID of a shared volume). New
    directories get 0775 (umask 002) and inherit the setgid group of /srv. Entries
    that cannot be written are reported and skipped, never fatal."""
    for root, dirs, files in os.walk(src):
        target_root = os.path.join(dst, os.path.relpath(root, src))
        try:
            os.makedirs(target_root, exist_ok=True)
        except OSError as error:
            log(f"warning: could not create {target_root}: {error}")
            dirs[:] = []
            continue
        for name in files + [d for d in dirs if os.path.islink(os.path.join(root, d))]:
            source, target = os.path.join(root, name), os.path.join(target_root, name)
            try:
                if os.path.islink(source):
                    if os.path.lexists(target) and not os.path.isdir(target):
                        os.unlink(target)
                    if not os.path.lexists(target):
                        os.symlink(os.readlink(source), target)
                else:
                    shutil.copyfile(source, target)
            except OSError as error:
                log(f"warning: could not refresh {target}: {error}")


def refresh_srv():
    app_dir = os.path.join(SRV, "app")
    for root, dirs, _ in os.walk(app_dir):
        if "__pycache__" in dirs:
            shutil.rmtree(os.path.join(root, "__pycache__"), ignore_errors=True)
            dirs.remove("__pycache__")
    overlay(os.path.join(DLTK, "app"), app_dir)
    overlay(os.path.join(DLTK, "notebooks"), os.path.join(SRV, "notebooks"))


def ensure_home():
    home = os.environ.setdefault("HOME", os.path.join(SRV, ".dsdl-home"))
    for path in (home,
                 USER_PACKAGES,
                 os.environ.get("JUPYTERLAB_SETTINGS_DIR", ""),
                 os.environ.get("JUPYTERLAB_WORKSPACES_DIR", "")):
        if path:
            try:
                os.makedirs(path, exist_ok=True)
            except OSError as error:
                log(f"warning: could not create {path}: {error}")


def hostname_resolves():
    try:
        socket.getaddrinfo(socket.gethostname(), None)
        return True
    except OSError:
        return False


def ensure_identity(env):
    """Give arbitrary UIDs a passwd entry, and the container hostname an address, without
    writing to /etc (read-only). Both go through nss_wrapper for the child processes.

    The hostname matters for Spark (InetAddress.getLocalHost) and other libraries; Docker
    and Kubernetes normally list it in /etc/hosts, but not with every network mode."""
    uid, gid = os.getuid(), os.getgid()
    try:
        pwd.getpwuid(uid)
        need_passwd = False
    except KeyError:
        need_passwd = True
    need_hosts = not hostname_resolves()
    if not (need_passwd or need_hosts):
        return
    if not os.path.exists(NSS_WRAPPER_LIB):
        log("warning: nss_wrapper is missing; cannot register the UID or the hostname.")
        return
    nss_dir = os.path.join("/tmp", "dsdl-nss")
    os.makedirs(nss_dir, exist_ok=True)
    if need_passwd:
        passwd_file = os.path.join(nss_dir, "passwd")
        group_file = os.path.join(nss_dir, "group")
        with open("/etc/passwd", encoding="utf-8") as src, open(passwd_file, "w", encoding="utf-8") as dst:
            dst.write(src.read())
            dst.write(f"dltk:x:{uid}:{gid}:dltk user:{env['HOME']}:/bin/bash\n")
        shutil.copyfile("/etc/group", group_file)
        env["NSS_WRAPPER_PASSWD"] = passwd_file
        env["NSS_WRAPPER_GROUP"] = group_file
        log(f"Registered UID {uid} through nss_wrapper.")
    if need_hosts:
        hostname = socket.gethostname()
        hosts_file = os.path.join(nss_dir, "hosts")
        with open("/etc/hosts", encoding="utf-8") as src, open(hosts_file, "w", encoding="utf-8") as dst:
            dst.write(src.read())
            dst.write(f"\n127.0.0.1 {hostname}\n::1 {hostname}\n")
        env["NSS_WRAPPER_HOSTS"] = hosts_file
        log(f"Mapped hostname {hostname} to the loopback address through nss_wrapper.")
    env["LD_PRELOAD"] = (NSS_WRAPPER_LIB + " " + env.get("LD_PRELOAD", "")).strip()


def certificate_valid(cert_path):
    from cryptography import x509

    try:
        with open(cert_path, "rb") as handle:
            cert = x509.load_pem_x509_certificate(handle.read())
    except (OSError, ValueError):
        return False
    remaining = cert.not_valid_after_utc - datetime.datetime.now(datetime.timezone.utc)
    return remaining > datetime.timedelta(days=CERT_RENEW_DAYS)


def generate_certificate(cert_path, key_path):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "dsdl")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=CERT_VALIDITY_DAYS))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("dsdl"), x509.DNSName("localhost")]), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    # group 0 (every container UID) may read and replace the pair, so a volume that moves to
    # another UID (OpenShift, Kubernetes runAsUser) keeps working; written via rename
    os.makedirs(os.path.dirname(cert_path), mode=0o770, exist_ok=True)
    write_atomically(key_path, key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                 serialization.NoEncryption()), 0o640)
    write_atomically(cert_path, cert.public_bytes(serialization.Encoding.PEM), 0o644)


def key_matches(cert_path, key_path):
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    try:
        with open(cert_path, "rb") as handle:
            cert = x509.load_pem_x509_certificate(handle.read())
        with open(key_path, "rb") as handle:
            key = serialization.load_pem_private_key(handle.read(), password=None)
    except (OSError, ValueError, TypeError):
        return False
    return cert.public_key().public_numbers() == key.public_key().public_numbers()


def resolve_tls(env):
    """Return (cert, key) and export them for Jupyter, or None when HTTPS is off.

    The generated certificate is checked (expiry, matching key) at every container
    start; a long-running container keeps its certificate until it is restarted."""
    if env.get("ENABLE_HTTPS", "true").lower() == "false":
        return None
    cert, key = env.get("API_SSL_CERT"), env.get("API_SSL_KEY")
    configured = bool(cert or key)
    if cert and key and os.path.isfile(cert) and os.path.isfile(key):
        source = "API_SSL_CERT/API_SSL_KEY"
        configured = False
    elif os.path.isfile(LEGACY_CERT) and os.path.isfile(LEGACY_KEY):
        cert, key, source = LEGACY_CERT, LEGACY_KEY, "/dltk/.jupyter (derived image)"
    else:
        cert, key = os.path.join(TLS_DIR, "dltk.pem"), os.path.join(TLS_DIR, "dltk.key")
        source = "generated per installation"
        if not (certificate_valid(cert) and key_matches(cert, key)):
            try:
                generate_certificate(cert, key)
                log(f"Generated a self-signed TLS certificate valid for {CERT_VALIDITY_DAYS} days.")
            except OSError as error:
                # e.g. /srv/.dsdl/tls created by another UID with an older image
                cert, key = "/tmp/dsdl-tls/dltk.pem", "/tmp/dsdl-tls/dltk.key"
                generate_certificate(cert, key)
                source = "generated for this container only"
                log(f"warning: cannot write {TLS_DIR} ({error}); using a certificate for this container only")
    if configured:
        log(f"WARNING: API_SSL_CERT/API_SSL_KEY ({env.get('API_SSL_CERT')}, {env.get('API_SSL_KEY')}) "
            f"do not both exist; using the {source} certificate instead")
    env["API_SSL_CERT"], env["API_SSL_KEY"] = cert, key
    log(f"ENABLE_HTTPS=true, certificate: {source}")
    return cert, key


def uvicorn_command(env, tls):
    # 5.2.x passed --reload-dir app in DEV mode without --reload, which never reloaded and only
    # made uvicorn warn; the API reloads a model module itself on /fit
    workers = env.get("api_workers") or "1"
    module = env.get("DSDL_APP_MODULE", "app.main:app")
    if module.startswith("app.main_agentic:") and workers != "1":
        # chat sessions and tool approvals live in the worker process; DSDL sets one
        # api_workers value for all containers
        log(f"api_workers={workers} ignored: the agentic API runs one worker")
        workers = "1"
    command = ["uvicorn", module, "--host", "0.0.0.0", "--port", "5000", "--workers", workers]
    if tls:
        command += ["--ssl-certfile", tls[0], "--ssl-keyfile", tls[1]]
    return command


def file_store_has_data(path):
    """True when a 5.2.x MLflow file store holds runs, experiments or registered models
    (5.2.x `mlflow ui` created an empty Default experiment 0 on its first start)."""
    try:
        for entry in os.scandir(path):
            if entry.name == "models" and entry.is_dir() and any(os.scandir(entry.path)):
                return True
            if entry.is_dir() and entry.name.isdigit():
                if entry.name != "0" or any(run.is_dir() for run in os.scandir(entry.path)):
                    return True
    except OSError:
        return True  # unreadable: leave it as it is
    return False


MLFLOW_MIGRATION_FAILED = os.path.join(SRV, ".dsdl", "mlflow-migration-failed")


def migrate_file_store(legacy, database, env):
    """Copy a 5.2.x file store into SQLite once; the file store itself stays unchanged. It runs
    before the services start; after a failure a marker stops it from blocking every later start."""
    if os.path.exists(MLFLOW_MIGRATION_FAILED):
        log(f"MLflow: not migrating {legacy} again after an earlier failure "
            f"(remove {MLFLOW_MIGRATION_FAILED} to retry)")
        return False
    target = database + ".migrating"
    try:
        for leftover in (target, target + "-wal", target + "-shm"):
            if os.path.exists(leftover):
                os.unlink(leftover)
        result = subprocess.run(["mlflow", "migrate-filestore", "--source", legacy, "--target", "sqlite:///" + target,
                                 "--no-progress"], env=dict(env, MLFLOW_ALLOW_FILE_STORE="true"), cwd=SRV,
                                capture_output=True, text=True, timeout=900)
        if result.returncode != 0:
            output = (result.stderr or result.stdout or "").strip().splitlines()
            # the exception line, not SQLAlchemy's trailing "(Background on this error at: ...)"
            errors = [line for line in output if re.match(r"^[\w.]+(Error|Exception)\b", line)]
            raise RuntimeError(((errors or output or ["unknown error"])[-1])[:300])
        # the migration writes in WAL mode: fold the WAL into the file before renaming it, and
        # use the same rollback journal as a store that MLflow creates itself
        import sqlite3
        connection = sqlite3.connect(target)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.close()
        os.chmod(target, 0o664)
        os.replace(target, database)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        log(f"warning: migrating the MLflow file store {legacy} failed: {error}")
        try:
            write_atomically(MLFLOW_MIGRATION_FAILED, f"{datetime.datetime.now().isoformat()} {error}\n".encode())
        except OSError:
            pass
        return False
    skipped = [line for line in (result.stderr or "").splitlines() if " WARNING " in line]
    log(f"MLflow: migrated the 5.2.x file store {legacy} into {database}; {legacy} is kept unchanged."
        + (f" {len(skipped)} warning(s), first: {skipped[0][:200]}" if skipped else ""))
    return True


def mlflow_command(env):
    """MLflow UI on 6000. MLflow 3 needs an explicit store (it refuses the implicit ./mlruns of a
    5.2.x volume); one worker and no job runner (4 workers plus the job runner use about 2.3 GB).
    MLflow's protection against DNS rebinding and cross-site requests stays on: a browser on
    localhost has full access, other hosts and origins are refused. MLFLOW_SERVER_ALLOWED_HOSTS and
    MLFLOW_SERVER_CORS_ALLOWED_ORIGINS (exact origins, never '*' patterns: MLflow treats those
    as "any origin") open it to the Docker host's name."""
    settings = ["MLFLOW_SERVER_ENABLE_JOB_EXECUTION=" + env.get("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "false")]
    command = ["mlflow", "ui", "-p", "6000", "-h", "0.0.0.0", "--workers", env.get("MLFLOW_WORKERS") or "1",
               "--artifacts-destination", os.path.join(SRV, "mlartifacts")]
    if not env.get("MLFLOW_SERVER_CORS_ALLOWED_ORIGINS"):
        log("MLflow UI: full access from a browser on localhost only; to open it through the Docker "
            "host's name or address set MLFLOW_SERVER_ALLOWED_HOSTS and MLFLOW_SERVER_CORS_ALLOWED_ORIGINS "
            "(exact values, e.g. http://<host>:<port>)")
    legacy, database = os.path.join(SRV, "mlruns"), os.path.join(SRV, "mlflow.db")
    if (os.path.isdir(legacy) and not os.path.exists(database) and file_store_has_data(legacy)
            and not migrate_file_store(legacy, database, env)):
        log(f"MLflow UI: using the file store {legacy}; migrate it with `mlflow migrate-filestore "
            f"--source {legacy} --target sqlite:///{database}`")
        return ["env", "MLFLOW_ALLOW_FILE_STORE=true"] + settings + command + ["--backend-store-uri", "file://" + legacy]
    if not os.path.exists(database):
        # created group-writable, so a volume that moves to another UID keeps a writable store
        # (an empty file is a valid SQLite database; its journal files take the same mode)
        try:
            os.close(os.open(database, os.O_WRONLY | os.O_CREAT, 0o664))
        except OSError as error:
            log(f"warning: could not create {database}: {error}")
    return ["env"] + settings + command + ["--backend-store-uri", "sqlite:///" + database]


def supervise(commands, env):
    """Run the services as children of this PID 1, forward signals and reap orphans.
    The container lives as long as the DSDL API (the last command) does."""
    processes = [subprocess.Popen(command, env=env, cwd=SRV) for command in commands]
    api = processes[-1]
    # service name for the log: the first argument that is neither `env` nor a VAR=value
    names = {process.pid: next(arg for arg in command if arg != "env" and "=" not in arg)
             for process, command in zip(processes, commands)}

    def forward(signum, _frame):
        for process in processes:
            if process.poll() is None:
                process.send_signal(signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    while True:
        try:
            pid, status = os.wait()  # also reaps orphans inherited by PID 1
        except ChildProcessError:
            break
        if pid == api.pid:
            api.returncode = os.waitstatus_to_exitcode(status)
            break
        if pid in names:
            log(f"warning: {names[pid]} exited with status {os.waitstatus_to_exitcode(status)}; "
                "the DSDL API keeps running")
    forward(signal.SIGTERM, None)
    deadline = time.monotonic() + 10
    for process in processes[:-1]:
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
    code = api.returncode or 0
    return 128 - code if code < 0 else code


def main():
    os.umask(0o002)
    os.chdir(SRV)
    env = dict(os.environ)
    image_version = env.get("DSDL_VERSION", "0")

    ensure_home()
    env["HOME"] = os.environ["HOME"]
    backup_notebooks(image_version)
    refresh_srv()
    try:  # TensorBoard logdir of the shipped notebooks
        os.makedirs(os.path.join(SRV, "notebooks", "logs"), exist_ok=True)
    except OSError as error:
        log(f"warning: could not create /srv/notebooks/logs: {error}")
    ensure_identity(env)
    tls = resolve_tls(env)

    # like 5.2.x `${MODE_DEV_PROD:-PROD}`: unset or empty means PROD
    mode = (env.get("MODE_DEV_PROD") or "PROD").upper()
    dev_mode = mode != "PROD"
    # DEV `pip install` lands on the persistent volume; its console scripts are on PATH.
    # Set for the child processes only, so `RUN pip install` in derived images still
    # installs into the image venv.
    env.setdefault("PIP_TARGET", USER_PACKAGES)
    env["PATH"] = env.get("PATH", "") + ":" + os.path.join(USER_PACKAGES, "bin")
    workers = env.get("api_workers") or "1"
    log(f"Starting DSDL {image_version} in mode = {'DEV' if dev_mode else 'PROD'} with {workers} api workers")

    api = uvicorn_command(env, tls)
    if not dev_mode:
        sys.exit(supervise([api], env))

    services = [
        ["jupyter", "lab", "--no-browser", "--ip=0.0.0.0", "--port=8888"],
        # --load_fast=false: the Rust data server binary (tensorboard-data-server 0.7.2, no newer
        # release) bundles its own TLS roots, including the distrusted emSign ECC Root CA - G3
        ["tensorboard", "--bind_all", "--load_fast=false", "--logdir", os.path.join(SRV, "notebooks", "logs")],
        mlflow_command(env),
    ]
    sys.exit(supervise(services + [api], env))


if __name__ == "__main__":
    main()
