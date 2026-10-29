# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Four things a production day asked for.

Measured on one host over 21h30m, 2026-10-29:

* a 3.5B music model held 7.9 GB of a 3090 and 2.1 GB of host RAM for hours
  after its last request, because the on-demand path keeps a model resident
  whenever there is room ("Ondemand mode - keeping 'audio_gen:ACE-Step' in VRAM
  alongside new model");
* 78% of the log (426,682 of 548,687 lines) was per-request debug from one
  engine, filling the 2,000-line view in about four minutes and leaving no
  record of the loads and evictions behind a slow render;
* those loads and evictions could only be inferred from that log;
* ACE-Step was asked for a song with no lyrics instead of an instrumental.
"""

import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------ idle unload ---

class _FakeManager:
    """Only what unload_idle_models touches."""

    from codai.models.manager import MultiModelManager as _MM
    unload_idle_models = _MM.unload_idle_models
    record_history = _MM.record_history
    get_history = _MM.get_history

    def __init__(self):
        from collections import deque
        self.models = {}
        self._last_used = {}
        self._keep_resident = set()
        self._history = deque(maxlen=500)
        self.evicted = []
        self.busy = set()

    def _lru_order(self):
        return sorted(self.models, key=lambda k: self._last_used.get(k, 0.0))

    def _is_key_busy(self, key):
        return key in self.busy

    def _evict_one(self, key, reason=""):
        self.evicted.append((key, reason))
        self.models.pop(key, None)
        self._last_used.pop(key, None)


def _mgr(**ages):
    m = _FakeManager()
    now = time.monotonic()
    for key, age in ages.items():
        m.models[key] = object()
        m._last_used[key] = now - age
    return m


def test_an_idle_model_is_unloaded():
    m = _mgr(**{"audio_gen:ACE-Step": 3600.0})
    freed = m.unload_idle_models(1800.0)
    assert freed == ["audio_gen:ACE-Step"]
    assert m.evicted and "idle" in m.evicted[0][1]


def test_a_recently_used_model_is_kept():
    m = _mgr(**{"video:X": 60.0})
    assert m.unload_idle_models(1800.0) == []
    assert m.models


def test_a_busy_model_is_never_pulled():
    m = _mgr(**{"video:X": 9999.0})
    m.busy.add("video:X")
    assert m.unload_idle_models(1800.0) == []


def test_keep_resident_is_respected():
    m = _mgr(**{"tts:Kokoro": 9999.0})
    m._keep_resident.add("tts:Kokoro")
    assert m.unload_idle_models(1800.0) == []


def test_an_unstamped_model_is_left_alone():
    """A key with no _last_used is indistinguishable from one loaded a second
    ago by a path that forgot to stamp it, so unloading it would be the
    opposite of what is wanted."""
    m = _FakeManager()
    m.models["video:X"] = object()          # never stamped
    assert m.unload_idle_models(1800.0) == []


def test_zero_disables_it():
    m = _mgr(**{"video:X": 99999.0})
    assert m.unload_idle_models(0) == []
    assert m.unload_idle_models(None) == []


def test_the_setting_exists_and_is_plumbed():
    from codai.config import OffloadConfig
    assert hasattr(OffloadConfig(), "idle_unload_seconds")
    main = io.open(os.path.join(_ROOT, "codai/main.py"), encoding="utf-8").read()
    assert "global_args.idle_unload_seconds" in main, (
        "the setting never reaches the running server")
    rm = io.open(os.path.join(_ROOT, "codai/models/ram_monitor.py"), encoding="utf-8").read()
    assert "unload_idle_models" in rm, "nothing ever runs the idle sweep"


# ---------------------------------------------------------------- history ---

def test_history_records_a_load_with_what_it_cost():
    m = _FakeManager()
    m.record_history("load", "video:X", seconds=44.2, vram_gb=15.68,
                     strategy="resident")
    row = m.get_history()[0]
    assert row["event"] == "load" and row["model"] == "video:X"
    assert row["seconds"] == 44.2 and row["vram_gb"] == 15.68
    assert row["time"] > 0


def test_history_records_why_a_model_was_evicted():
    m = _FakeManager()
    m.record_history("evict", "video:X", reason="cosite: a co-located engine "
                                                "needed the GPU", freed_gb=11.1)
    row = m.get_history()[0]
    assert "co-located" in row["reason"]
    assert row["freed_gb"] == 11.1


def test_history_is_newest_first_and_filterable():
    m = _FakeManager()
    m.record_history("load", "a")
    m.record_history("evict", "b")
    assert [r["model"] for r in m.get_history()] == ["b", "a"]
    assert [r["model"] for r in m.get_history(event="load")] == ["a"]


def test_history_is_bounded():
    """It must not become the leak it is there to help diagnose."""
    m = _FakeManager()
    for i in range(1200):
        m.record_history("load", f"m{i}")
    assert len(m._history) == 500


def test_eviction_reasons_are_passed_at_the_call_sites():
    src = io.open(os.path.join(_ROOT, "codai/models/manager.py"), encoding="utf-8").read()
    for marker in ("reason=f\"vram:", "reason=\"cosite:", "reason=f\"idle for"):
        assert marker in src, f"no reason given at the {marker!r} call site"


def test_the_endpoints_exist():
    eng = io.open(os.path.join(_ROOT, "codai/api/app.py"), encoding="utf-8").read()
    assert '"/v1/models/history"' in eng
    front = io.open(os.path.join(_ROOT, "codai/frontproxy/app.py"), encoding="utf-8").read()
    assert '"/v1/models/history"' in front, (
        "the front does not aggregate it, so a caller sees one engine's half of "
        "a cross-engine swap")
    # It must be registered BEFORE the catch-all, or it is proxied to one engine.
    assert front.index('"/v1/models/history"') < front.index("/admin/api/engines")


# ------------------------------------------------------------- log volume ---

def test_embeddings_are_summarised_not_dumped():
    from codai.api import log as L
    assert "/v1/embeddings" in L._SUMMARY_ONLY_PATHS
    assert "/v1/embeddings" in L._TRACKED_PATHS, (
        "summarising must not stop the admin page counting them")


def test_the_summary_says_what_was_asked_without_the_content():
    from codai.api import log as L
    secret = "a client's private property listing " * 20
    line = L._summarise({"model": "Qwen3-Embedding-4B", "input": [secret, secret],
                         "dimensions": 1024})
    assert "model=Qwen3-Embedding-4B" in line
    assert "inputs=2" in line
    assert "dimensions=1024" in line
    assert "property listing" not in line, "the caller's content is still in the log"


def test_a_prompt_is_left_intact():
    """Deliberately NOT truncated: a prompt is the most useful thing in the log
    -- a 4,000-character script prompt was what identified a music defect --
    and the volume problem was one per-item endpoint, not prompts."""
    from codai.api import log as L
    # Real prose, not a run of base64-legal characters: a long unbroken token
    # IS treated as a blob, and correctly so.
    prompt = ("Write a 81-second video script about the rally. " * 100)
    out = L._redact_blobs({"prompt": prompt})
    assert out["prompt"] == prompt, "a script prompt is being truncated"
    # A base64 payload is still redacted.
    blob = L._redact_blobs({"init_image": "A" * 5000})["init_image"]
    assert "truncated" in blob


def test_a_short_value_is_untouched():
    from codai.api import log as L
    assert L._redact_blobs({"model": "Krea-2-Turbo"}) == {"model": "Krea-2-Turbo"}


# ------------------------------------------------------- acestep lyrics ----

def test_an_instrumental_request_says_so():
    """ACE-Step is a SONG model. Empty lyrics leave it conditioning on a single
    padding token (lyric_token_idx = [0], lyric_mask = [0]) and it still tries
    to sing. Its own UI documents the way to ask: "Use [instrumental] or [inst]
    to generate instrumental music"."""
    src = io.open(os.path.join(_ROOT, "codai/api/audio_gen.py"), encoding="utf-8").read()
    block = src.split("elif model_type == 'ace_step':", 1)[1][:2000]
    assert "[instrumental]" in block, (
        "an empty-lyrics request is still sent as a song with no words")
    assert "lyrics=_lyrics" in block
    assert "instrumental" in block.split("_prompt")[1][:400], (
        "the tag prompt is not told it is instrumental either")
