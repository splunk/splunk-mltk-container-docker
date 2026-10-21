"""Start-up logic for every Splunk DSDL container image.

Replaces bootstrap_fast.sh, bootstrap_transformers.sh, bootstrap_rapids.sh,
bootstrap_spark.sh and bootstrap_backup.sh. Behaviour kept from 5.2.x:

* /srv is the persistent volume: on every start the image's /dltk/app is
  overlaid onto /srv/app (stale __pycache__ removed first) and /dltk/notebooks
  is copied into /srv/notebooks. When the image version is newer than the one
  recorded in /srv/version.md, /srv/notebooks is backed up first.
* MODE_DEV_PROD=PROD (default) starts only the DSDL API (uvicorn on 5000).
  MODE_DEV_PROD=DEV also starts JupyterLab (8888), TensorBoard (6006) and the
  MLflow UI (6000).
* ENABLE_HTTPS (default true), api_workers and API_SSL_CERT / API_SSL_KEY are
  honoured as before.

New in 5.3.0:

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
* The MLflow UI gets an explicit store: /srv/mlflow.db (SQLite), or the 5.2.x
  file store /srv/mlruns when a volume has one and no mlflow.db yet (MLflow 3
  refuses an implicit file store).
"""

import datetime
import os
import pwd
import re
import shutil
import signal
import socket
import subprocess
import sys
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


def version_tuple(version):
    """Numeric comparison key: 5.2.10 > 5.2.5 (5.2.x compared strings)."""
    return tuple(int(part) for part in re.findall(r"\d+", version or ""))


def backup_notebooks(image_version):
    version_file = os.path.join(SRV, "version.md")
    notebooks_dir = os.path.join(SRV, "notebooks")
    recorded = ""
    try:
        if os.path.isfile(version_file):
            with open(version_file, encoding="utf-8") as handle:
                recorded = handle.read().strip()
        if version_tuple(recorded) >= version_tuple(image_version):
            log(f"Notebook backup not needed (volume at {recorded or 'unknown'}).")
            return
        if os.path.isdir(notebooks_dir):
            backup_dir = os.path.join(SRV, f"notebooks_backup_{image_version}")
            if not os.path.exists(backup_dir):
                shutil.copytree(notebooks_dir, backup_dir, symlinks=True, copy_function=shutil.copyfile)
                log(f"Backed up notebooks from {recorded or 'an earlier version'} to {backup_dir}.")
        with open(version_file, "w", encoding="utf-8") as handle:
            handle.write(image_version + "\n")
    except OSError as error:
        log(f"warning: notebook backup failed: {error}")


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
                 os.environ.get("JUPYTERLAB_WORKSPACES_DIR", ""),
                 os.path.join(SRV, "notebooks", "logs")):
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
    os.makedirs(os.path.dirname(cert_path), mode=0o700, exist_ok=True)
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                       serialization.NoEncryption()))
    with open(cert_path, "wb") as handle:
        handle.write(cert.public_bytes(serialization.Encoding.PEM))


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
            generate_certificate(cert, key)
            log(f"Generated a self-signed TLS certificate valid for {CERT_VALIDITY_DAYS} days.")
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
    command = ["uvicorn", env.get("DSDL_APP_MODULE", "app.main:app"),
               "--host", "0.0.0.0", "--port", "5000", "--workers", workers]
    if tls:
        command += ["--ssl-certfile", tls[0], "--ssl-keyfile", tls[1]]
    return command


def mlflow_command():
    """MLflow UI on 6000 with an explicit backend store. Without one, MLflow 3 picks up the
    ./mlruns of a 5.2.x volume and refuses to start (file store in maintenance mode)."""
    command = ["mlflow", "ui", "-p", "6000", "-h", "0.0.0.0"]
    legacy = os.path.join(SRV, "mlruns")
    database = os.path.join(SRV, "mlflow.db")
    if os.path.isdir(legacy) and not os.path.exists(database):
        log("MLflow UI: keeping the 5.2.x file store /srv/mlruns; migrate it with "
            "`mlflow migrate-filestore` (MLflow 3 no longer develops the file store).")
        return ["env", "MLFLOW_ALLOW_FILE_STORE=true"] + command + ["--backend-store-uri", "file://" + legacy]
    return command + ["--backend-store-uri", "sqlite:///" + database]


def supervise(commands, env):
    """Run the services as children of this PID 1, forward signals and reap orphans.
    The container lives as long as the DSDL API (the last command) does."""
    processes = [subprocess.Popen(command, env=env, cwd=SRV) for command in commands]
    api = processes[-1]

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
    forward(signal.SIGTERM, None)
    deadline = time.monotonic() + 10
    for process in processes[:-1]:
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
    return api.returncode or 0


def main():
    os.umask(0o002)
    os.chdir(SRV)
    env = dict(os.environ)
    image_version = env.get("DSDL_VERSION", "0")

    ensure_home()
    env["HOME"] = os.environ["HOME"]
    backup_notebooks(image_version)
    refresh_srv()
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
        ["tensorboard", "--bind_all", "--logdir", os.path.join(SRV, "notebooks", "logs")],
        mlflow_command(),
    ]
    sys.exit(supervise(services + [api], env))


if __name__ == "__main__":
    main()
