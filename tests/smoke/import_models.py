"""Import every app/model module inside an image; compare with tests/expected/<target>.txt.

Runs inside the container (python - < import_models.py) with the repository's
expected list on stdin's environment: EXPECTED holds the newline-separated module
names that must import. Prints one line per module and a JSON summary; exits 1
when an expected module fails to import.
"""

import contextlib
import importlib
import io
import json
import os
import shutil
import signal
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# same layout the entrypoint creates: /srv/app is a copy of /dltk/app
work = "/srv"
shutil.copytree("/dltk/app", os.path.join(work, "app"), dirs_exist_ok=True)
sys.path.insert(0, work)
os.chdir(work)


class Timeout(Exception):
    pass


def on_alarm(signum, frame):
    raise Timeout("timeout")


signal.signal(signal.SIGALRM, on_alarm)
expected = {line.strip() for line in os.environ.get("EXPECTED", "").splitlines()
            if line.strip() and not line.startswith("#")}
modules = sorted(name[:-3] for name in os.listdir(os.path.join(work, "app", "model"))
                 if name.endswith(".py") and name != "__init__.py")
results = {}
for module in modules + ["__app.main"]:
    target = "app.main" if module == "__app.main" else "app.model." + module
    started = time.time()
    signal.alarm(int(os.environ.get("PER_MOD_TIMEOUT", "300")))
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            importlib.import_module(target)
        results[module] = {"ok": True}
    except BaseException as error:  # noqa: BLE001 - report every failure kind
        results[module] = {"ok": False, "err": traceback.format_exception_only(type(error), error)[-1].strip()[:300]}
    finally:
        signal.alarm(0)
    status = "OK  " if results[module]["ok"] else "FAIL"
    marker = "*" if module in expected else " "
    print(f"{status} {marker} {module} ({time.time() - started:.1f}s) {results[module].get('err', '')}", flush=True)

regressions = sorted(m for m in expected if not results.get(m, {}).get("ok"))
gained = sorted(m for m, r in results.items() if r["ok"] and m not in expected and not m.startswith("__"))
summary = {"imported": sum(r["ok"] for r in results.values()), "total": len(results),
           "expected": len(expected), "regressions": regressions, "gained": gained,
           "app_main": results["__app.main"]["ok"]}
print("SUMMARY " + json.dumps(summary))
sys.exit(1 if regressions or not summary["app_main"] else 0)
