"""No coroutine may make a blocking call.

A synchronous call made from an `async def` body does not slow one request down,
it stops the process: nothing else on that event loop runs until it returns. The
symptom is indistinguishable from an overloaded upstream, which is why it cost a
whole afternoon once -- a 5,000-document OCR run rented 127 pods and completed
nothing for ten minutes, while the pods sat idle and healthy and the real fault
was `placement.choose()` (a pool acquire, up to 1200 s, able to rent a pod
inline) called straight from a coroutine. The front had stopped answering
/v1/runpod/status too, which was the only honest signal in the whole system.

So it is checked here instead of being found again. The rule that makes the
check precise is that it does NOT look inside a nested `def` or `lambda`:

    await asyncio.to_thread(lambda: list(iter_pdf_page_pdfs(data)))   # fine
    await run_on_pod_pool(placement.choose, model_id)                 # fine
    placement.choose(model_id)                                        # NOT

The first hides the call in a function object and the second passes a reference
without calling it, so only a call reached without crossing a function boundary
is actually on the loop. Awaited calls and coroutines handed to
`ensure_future`/`create_task`/`gather` are coroutines by construction.

`codai/api/loop_watch.py` is the runtime half of this: it measures how long the
loop went without running, so a blocking call this check cannot see -- one
inside a library, or behind a layer of indirection -- still reports itself.
"""

import ast
import os

import pytest

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "codai")

#: Dotted call names that block the calling thread, and why.
BLOCKING = {
    "time.sleep": "sleeps the thread",
    "requests.get": "sync HTTP", "requests.post": "sync HTTP",
    "requests.put": "sync HTTP", "requests.delete": "sync HTTP",
    "requests.head": "sync HTTP", "requests.patch": "sync HTTP",
    "requests.request": "sync HTTP",
    "urllib.request.urlopen": "sync HTTP",
    "subprocess.run": "waits for a process",
    "subprocess.call": "waits for a process",
    "subprocess.check_call": "waits for a process",
    "subprocess.check_output": "waits for a process",
    "os.system": "waits for a process",
    "placement.choose": "ends in a pool acquire: up to 1200 s, may rent inline",
    "pod_http.post": "sync HTTP — the thread twin of apost",
    "pod_http.get": "sync HTTP — the thread twin of aget",
}

#: Method names that block when the receiver is not an asyncio primitive.
BLOCKING_METHODS = {
    "acquire": "a pool or threading lock acquire waits",
    "communicate": "waits for a process",
    "join": "waits for a thread or process",
    "result": "waits for a future",
}

#: Receiver name fragments whose `.acquire()`/`.wait()` are asyncio-native.
SAFE_RECEIVER = ("asyncio", "sem", "lock", "cv", "event", "cond", "gate")

#: `join` is also `str.join` and `os.path.join`.
JOINABLE = ("thread", "proc", "worker", "task", "child")

#: Calls that hand a coroutine or a callable to someone else to run.
HANDOFF = ("ensure_future", "create_task", "gather", "wait_for", "shield",
           "to_thread", "run_in_executor", "run_on_pod_pool", "run_coroutine_threadsafe")

#: (file suffix, line, call) deliberately left blocking, each with a reason.
#: Empty, and it should stay that way: an entry here is a request that cannot be
#: served while that call runs.
ALLOWED: dict = {}


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _classify(call):
    name = _dotted(call.func)
    if not name:
        return None
    for suffix, why in BLOCKING.items():
        if name == suffix or name.endswith("." + suffix):
            return name, why
    if isinstance(call.func, ast.Attribute):
        why = BLOCKING_METHODS.get(call.func.attr)
        if not why:
            return None
        recv = _dotted(call.func.value).lower()
        if not recv or any(h in recv for h in SAFE_RECEIVER):
            return None
        if call.func.attr == "join":
            if "path" in recv or recv.endswith("os"):
                return None
            tail = recv.rsplit(".", 1)[-1]
            if not any(j in tail for j in JOINABLE):
                return None
        return name, why
    return None


def _offenders_in(path: str):
    """Blocking calls reached from an async body without crossing a frame."""
    try:
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    except SyntaxError as exc:                      # a real problem, say so
        pytest.fail(f"{path} does not parse: {exc}")
    found = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.AsyncFunctionDef):
            continue
        handed_off, stack = set(), list(fn.body)
        while stack:
            n = stack.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue                            # another frame's problem
            if isinstance(n, ast.Await) and isinstance(n.value, ast.Call):
                handed_off.add(id(n.value))
            if isinstance(n, ast.Call):
                if _dotted(n.func).rsplit(".", 1)[-1] in HANDOFF:
                    for a in list(n.args) + [k.value for k in n.keywords]:
                        if isinstance(a, ast.Call):
                            handed_off.add(id(a))
                if id(n) not in handed_off:
                    hit = _classify(n)
                    if hit:
                        found.append((n.lineno, fn.name, hit[0], hit[1]))
            stack.extend(ast.iter_child_nodes(n))
    return found


def _sources():
    for base, _dirs, files in os.walk(ROOT):
        if "__pycache__" in base:
            continue
        for f in sorted(files):
            if f.endswith(".py"):
                yield os.path.join(base, f)


def test_no_coroutine_makes_a_blocking_call():
    bad = []
    for path in _sources():
        rel = os.path.relpath(path, os.path.dirname(ROOT))
        for lineno, fnname, name, why in _offenders_in(path):
            if ALLOWED.get((rel, lineno)) == name:
                continue
            bad.append(f"{rel}:{lineno} async {fnname}() calls {name}() — {why}")
    assert not bad, (
        "these calls block the event loop, so NO request of any kind is served "
        "while they run. Move each one off the loop (await asyncio.to_thread(...), "
        "or run_on_pod_pool for a pod lease) or use the async twin:\n  "
        + "\n  ".join(sorted(bad)))


# --------------------------------------------- the checker checks what it says #
# A checker nobody has seen fail is a checker that might match nothing at all.

def _hits(src: str):
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        name = fh.name
    try:
        return _offenders_in(name)
    finally:
        os.unlink(name)


def test_it_catches_a_blocking_call_in_a_coroutine():
    assert _hits("import time\nasync def f():\n    time.sleep(1)\n")


def test_it_catches_the_pool_lease_that_caused_this():
    assert _hits("async def f():\n    placement.choose('m')\n")


def test_it_allows_the_same_call_in_a_thread():
    assert not _hits("import asyncio, time\n"
                     "async def f():\n"
                     "    await asyncio.to_thread(time.sleep, 1)\n")


def test_it_allows_a_reference_passed_to_a_pool():
    assert not _hits("async def f():\n"
                     "    await run_on_pod_pool(placement.choose, 'm')\n")


def test_it_allows_a_call_hidden_in_a_lambda():
    assert not _hits("import asyncio\n"
                     "async def f():\n"
                     "    await asyncio.to_thread(lambda: placement.choose('m'))\n")


def test_it_ignores_a_blocking_call_in_a_plain_def():
    """A sync function is free to block: its caller decided where it runs."""
    assert not _hits("import time\ndef f():\n    time.sleep(1)\n")


def test_it_does_not_flag_os_path_join_or_str_join():
    assert not _hits("import os\n"
                     "async def f():\n"
                     "    p = os.path.join('a', 'b')\n"
                     "    return ','.join(['x', 'y']) + p\n")


def test_it_does_not_flag_an_asyncio_primitive():
    assert not _hits("async def f(sem, lock):\n"
                     "    await sem.acquire()\n"
                     "    await lock.acquire()\n")


def test_it_does_not_flag_a_coroutine_handed_to_the_loop():
    assert not _hits("import asyncio\n"
                     "async def f(q):\n"
                     "    t = asyncio.ensure_future(q.acquire('k'))\n"
                     "    return await t\n")


def test_it_catches_a_thread_join_but_not_a_path_join():
    assert _hits("async def f(thread):\n    thread.join()\n")
