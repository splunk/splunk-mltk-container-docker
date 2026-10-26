"""DEV mode: log in to JupyterLab, start a python3 kernel and run a cell over the websocket.

Runs inside a container started with MODE_DEV_PROD=DEV and JUPYTER_PASSWD set to a password hash, as
DSDL does (python - < jupyter_kernel.py, plain password in SMOKE_JUPYTER_PASSWORD), the way a user's
notebook talks to the kernel. Prints PASS, FAIL or (emulated platform, SMOKE_EMULATED=1) SKIP; exits 1 on FAIL.
"""

import asyncio
import http.cookiejar
import json
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request
import uuid

from tornado.httpclient import HTTPRequest
from tornado.websocket import websocket_connect

BASE = "https://127.0.0.1:8888"
CONTEXT = ssl._create_unverified_context()
jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=CONTEXT),
                                     urllib.request.HTTPCookieProcessor(jar))


def cookie(name):
    return next((c.value for c in jar if c.name == name), "")


def request(path, data=None, method=None):
    headers = {"X-XSRFToken": cookie("_xsrf")}
    if isinstance(data, dict):
        data, headers["Content-Type"] = json.dumps(data).encode(), "application/json"
    with opener.open(urllib.request.Request(BASE + path, data=data, headers=headers, method=method), timeout=30) as r:
        return json.loads(r.read() or b"null")


async def execute(kernel_id, code, timeout):
    headers = {"Cookie": "; ".join(f"{c.name}={c.value}" for c in jar), "X-XSRFToken": cookie("_xsrf")}
    url = BASE.replace("https", "wss") + f"/api/kernels/{kernel_id}/channels"
    # the server holds the websocket upgrade until the kernel answers kernel_info (up to 60 s)
    ws = await websocket_connect(HTTPRequest(url, headers=headers, ssl_options=CONTEXT, connect_timeout=90,
                                             request_timeout=90))
    msg_id = uuid.uuid4().hex
    ws.write_message(json.dumps({
        "header": {"msg_id": msg_id, "msg_type": "execute_request", "username": "smoke", "session": uuid.uuid4().hex,
                   "version": "5.3"},
        "parent_header": {}, "metadata": {}, "channel": "shell",
        "content": {"code": code, "silent": False, "store_history": False, "user_expressions": {},
                    "allow_stdin": False, "stop_on_error": True},
    }))
    result, status, deadline = None, None, time.monotonic() + timeout
    while status is None or result is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        message = await asyncio.wait_for(ws.read_message(), remaining)
        if message is None:
            break
        message = json.loads(message)
        if message.get("parent_header", {}).get("msg_id") != msg_id:
            continue
        if message["msg_type"] == "execute_result":
            result = message["content"]["data"].get("text/plain")
        elif message["msg_type"] == "execute_reply":
            status = message["content"]["status"]
            if status != "ok":
                break
    ws.close()
    return status, result


def main():
    started = time.monotonic()
    opener.open(BASE + "/login", timeout=30).read()
    form = urllib.parse.urlencode({"password": os.environ["SMOKE_JUPYTER_PASSWORD"], "_xsrf": cookie("_xsrf")}).encode()
    opener.open(urllib.request.Request(BASE + "/login?next=%2Flab", data=form), timeout=30).read()
    kernel = request("/api/kernels", {"name": "python3"})
    try:
        # connect right away, as JupyterLab does: the server opens the shell channel and asks the
        # kernel for kernel_info only when a websocket connects (a hung kernel never answers it and
        # JupyterLab shows "Connecting")
        status, result = asyncio.run(execute(kernel["id"], "1 + 1", timeout=90))
        state = request(f"/api/kernels/{kernel['id']}")["execution_state"]
    finally:
        request(f"/api/kernels/{kernel['id']}", method="DELETE")
    if status == "ok" and result == "2":
        print(f"PASS kernel starts and runs a cell in JupyterLab ({time.monotonic() - started:.1f}s, state {state})")
        return 0
    return failed(f"reply {status}, result {result}, state {state} (after {time.monotonic() - started:.1f}s)")


def failed(detail):
    # tools/smoke-test.sh sets SMOKE_EMULATED=1 when the image platform differs from the Docker
    # engine's (amd64 on Apple silicon): Rosetta deadlocks in the kernel (futex LOCK_PI in Rosetta
    # itself, also with the 5.2.x images), and /proc/self/exe shows the guest binary, so the
    # emulator cannot be detected from inside. Not a defect of the image: do not fail the run.
    if os.environ.get("SMOKE_EMULATED") == "1":
        print(f"SKIP kernel check under emulation (amd64 on Apple silicon hangs Jupyter kernels; verify on "
              f"an amd64 host): {detail}")
        return 0
    print(f"FAIL kernel did not run a cell: {detail}")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # noqa: BLE001 - every failure is a FAIL (or SKIP) line
        sys.exit(failed(f"{type(error).__name__}: {error}"))
