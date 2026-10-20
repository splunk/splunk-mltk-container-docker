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
* /etc/passwd is read-only. Arbitrary UIDs (OpenShift) get a passwd entry via
  nss_wrapper instead of editing /etc/passwd at runtime.
* DSDL_APP_MODULE selects the FastAPI application (app.main:app by default,
  app.main_agentic:app for the agentic-ai tags).
* The image content is read-only. In DEV mode `pip install` writes to
  PIP_TARGET (/srv/.dsdl/site-packages, persistent); that path is searched after
  the image's site-packages, so it cannot shadow image packages.
"""

import datetime
import os
import pwd
import re
import shutil
import signal
import subprocess
import sys
import time

SRV = "/srv"
DLTK = "/dltk"
TLS_DIR = os.path.join(SRV, ".dsdl", "tls")
LEGACY_CERT = os.path.join(DLTK, ".jupyter", "dltk.pem")
LEGACY_KEY = os.path.join(DLTK, ".jupyter", "dltk.key")
NSS_WRAPPER_LIB = "/usr/lib64/libnss_wrapper.so"
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
    if os.path.isfile(version_file):
        with open(version_file, encoding="utf-8") as handle:
            recorded = handle.read().strip()
    if version_tuple(recorded) >= version_tuple(image_version):
        log(f"Notebook backup not needed (volume at {recorded or 'unknown'}).")
        return
    if os.path.isdir(notebooks_dir):
        backup_dir = os.path.join(SRV, f"notebooks_backup_{image_version}")
        if not os.path.exists(backup_dir):
            shutil.copytree(notebooks_dir, backup_dir, symlinks=True)
            log(f"Backed up notebooks from {recorded or 'an earlier version'} to {backup_dir}.")
    with open(version_file, "w", encoding="utf-8") as handle:
        handle.write(image_version + "\n")


def refresh_srv():
    app_dir = os.path.join(SRV, "app")
    os.makedirs(app_dir, exist_ok=True)
    for root, dirs, _ in os.walk(app_dir):
        if "__pycache__" in dirs:
            shutil.rmtree(os.path.join(root, "__pycache__"), ignore_errors=True)
            dirs.remove("__pycache__")
    # copyfile instead of copy2: image files are read-only, the copies must stay
    # editable by the runtime user (equivalent to 5.2.x `cp -RT` / `cp -R`).
    shutil.copytree(os.path.join(DLTK, "app"), app_dir, dirs_exist_ok=True, copy_function=shutil.copyfile)
    shutil.copytree(os.path.join(DLTK, "notebooks"), os.path.join(SRV, "notebooks"),
                    dirs_exist_ok=True, copy_function=shutil.copyfile)


def ensure_home():
    home = os.environ.setdefault("HOME", os.path.join(SRV, ".dsdl-home"))
    for path in (home,
                 os.environ.get("PIP_TARGET", ""),
                 os.environ.get("JUPYTERLAB_SETTINGS_DIR", ""),
                 os.environ.get("JUPYTERLAB_WORKSPACES_DIR", ""),
                 os.path.join(SRV, "notebooks", "logs")):
        if path:
            os.makedirs(path, exist_ok=True)


def ensure_user(env):
    """Give arbitrary UIDs a passwd entry without a writable /etc/passwd."""
    uid, gid = os.getuid(), os.getgid()
    try:
        pwd.getpwuid(uid)
        return
    except KeyError:
        pass
    if not os.path.exists(NSS_WRAPPER_LIB):
        log(f"UID {uid} has no passwd entry and nss_wrapper is missing; continuing without one.")
        return
    nss_dir = os.path.join("/tmp", "dsdl-nss")
    os.makedirs(nss_dir, exist_ok=True)
    passwd_file = os.path.join(nss_dir, "passwd")
    group_file = os.path.join(nss_dir, "group")
    with open("/etc/passwd", encoding="utf-8") as src, open(passwd_file, "w", encoding="utf-8") as dst:
        dst.write(src.read())
        dst.write(f"dltk:x:{uid}:{gid}:dltk user:{env['HOME']}:/bin/bash\n")
    shutil.copyfile("/etc/group", group_file)
    env["LD_PRELOAD"] = (NSS_WRAPPER_LIB + " " + env.get("LD_PRELOAD", "")).strip()
    env["NSS_WRAPPER_PASSWD"] = passwd_file
    env["NSS_WRAPPER_GROUP"] = group_file
    log(f"Registered UID {uid} through nss_wrapper.")


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


def resolve_tls(env):
    """Return (cert, key) and export them for Jupyter, or None when HTTPS is off."""
    if env.get("ENABLE_HTTPS", "true").lower() == "false":
        return None
    cert, key = env.get("API_SSL_CERT"), env.get("API_SSL_KEY")
    if cert and key and os.path.isfile(cert) and os.path.isfile(key):
        source = "API_SSL_CERT/API_SSL_KEY"
    elif os.path.isfile(LEGACY_CERT) and os.path.isfile(LEGACY_KEY):
        cert, key, source = LEGACY_CERT, LEGACY_KEY, "/dltk/.jupyter (derived image)"
    else:
        cert, key = os.path.join(TLS_DIR, "dltk.pem"), os.path.join(TLS_DIR, "dltk.key")
        source = "generated per installation"
        if not (os.path.isfile(key) and certificate_valid(cert)):
            generate_certificate(cert, key)
            log(f"Generated a self-signed TLS certificate valid for {CERT_VALIDITY_DAYS} days.")
    env["API_SSL_CERT"], env["API_SSL_KEY"] = cert, key
    log(f"ENABLE_HTTPS=true, certificate: {source}")
    return cert, key


def uvicorn_command(env, tls, dev_mode):
    workers = env.get("api_workers") or "1"
    command = ["uvicorn", env.get("DSDL_APP_MODULE", "app.main:app"),
               "--host", "0.0.0.0", "--port", "5000", "--workers", workers]
    if dev_mode:
        command += ["--reload-dir", "app"]
    if tls:
        command += ["--ssl-certfile", tls[0], "--ssl-keyfile", tls[1]]
    return command


def supervise(commands, env):
    """Run DEV services; the container lives as long as the DSDL API does."""
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
    ensure_user(env)
    tls = resolve_tls(env)

    mode = env.get("MODE_DEV_PROD", "PROD").upper()
    dev_mode = mode != "PROD"
    workers = env.get("api_workers") or "1"
    log(f"Starting DSDL {image_version} in mode = {'DEV' if dev_mode else 'PROD'} with {workers} api workers")

    api = uvicorn_command(env, tls, dev_mode)
    if not dev_mode:
        os.execvpe(api[0], api, env)

    services = [
        ["jupyter", "lab", "--no-browser", "--ip=0.0.0.0", "--port=8888"],
        ["tensorboard", "--bind_all", "--logdir", os.path.join(SRV, "notebooks", "logs")],
        ["mlflow", "ui", "-p", "6000", "-h", "0.0.0.0"],
    ]
    sys.exit(supervise(services + [api], env))


if __name__ == "__main__":
    main()
