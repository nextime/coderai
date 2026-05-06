# Manual Multimodal Test Client Sample Sourcing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic auto-download of missing default sample media for the manual multimodal test client while preserving explicit file overrides and existing mode behavior.

**Architecture:** Extend `tools/manual_multimodal_test_client.py` with a small internal sample-source manifest plus helpers that resolve default `samples/` assets before request construction. Keep network access isolated in one helper so tests in `tests/test_manual_multimodal_test_client.py` can mock downloads without changing request-building or response-parsing responsibilities.

**Tech Stack:** Python 3, `pathlib`, `subprocess` for `wget`, `pytest`, `monkeypatch`

---

## File Structure

- `tools/manual_multimodal_test_client.py`
  - Keep mode defaults in `MODE_DEFAULTS`
  - Add a fixed sample manifest for default sample media URLs
  - Add helpers that detect whether a configured file path is a default managed sample
  - Add one download helper that shells out to `wget` and stores files in `samples/`
  - Update file resolution so default paths auto-download when missing but explicit overrides still fail clearly
- `tests/test_manual_multimodal_test_client.py`
  - Add unit tests for sample-manifest resolution and `wget` invocation
  - Add request-builder tests proving missing default samples trigger managed download
  - Keep network and shell interactions mocked

### Task 1: Add failing tests for managed default sample resolution

**Files:**
- Modify: `tests/test_manual_multimodal_test_client.py`
- Modify: `tools/manual_multimodal_test_client.py`
- Test: `tests/test_manual_multimodal_test_client.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from tools.manual_multimodal_test_client import ensure_sample_file


def test_ensure_sample_file_returns_existing_path_without_download(tmp_path, monkeypatch):
    sample_path = tmp_path / "samples" / "transcription.wav"
    sample_path.parent.mkdir(parents=True)
    sample_path.write_bytes(b"wav-bytes")
    calls = []

    monkeypatch.setattr(
        "tools.manual_multimodal_test_client.download_default_sample",
        lambda path: calls.append(path),
    )

    result = ensure_sample_file(str(sample_path), "--audio-file")

    assert result == sample_path
    assert calls == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_manual_multimodal_test_client.py::test_ensure_sample_file_returns_existing_path_without_download -v`
Expected: FAIL with `ImportError` or `AttributeError` because `ensure_sample_file` and `download_default_sample` do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
SAMPLE_URLS = {
    "samples/transcription.wav": "https://example.invalid/transcription.wav",
    "samples/question-video.mp4": "https://example.invalid/question-video.mp4",
    "samples/question-audio.wav": "https://example.invalid/question-audio.wav",
}


def download_default_sample(path: Path) -> Path:
    raise NotImplementedError


def ensure_sample_file(path_value: str | None, flag_name: str) -> Path:
    if not path_value:
        raise FileNotFoundError(f"Missing required file. Supply {flag_name}.")
    path = Path(path_value)
    if path.exists():
        return path
    raise FileNotFoundError(f"File not found: {path}. Supply {flag_name}.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_manual_multimodal_test_client.py::test_ensure_sample_file_returns_existing_path_without_download -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_manual_multimodal_test_client.py tools/manual_multimodal_test_client.py
git commit -m "test: cover existing managed sample resolution"
```

### Task 2: Add failing tests for auto-download of missing default samples

**Files:**
- Modify: `tests/test_manual_multimodal_test_client.py`
- Modify: `tools/manual_multimodal_test_client.py`
- Test: `tests/test_manual_multimodal_test_client.py`

- [ ] **Step 1: Write the failing test**

```python
def test_ensure_sample_file_downloads_missing_managed_default(tmp_path, monkeypatch):
    managed_path = tmp_path / "samples" / "question-audio.wav"
    downloaded = []

    def fake_download(path):
        downloaded.append(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"audio")
        return path

    monkeypatch.setattr(
        "tools.manual_multimodal_test_client.SAMPLE_URLS",
        {managed_path.as_posix(): "https://example.invalid/question-audio.wav"},
    )
    monkeypatch.setattr(
        "tools.manual_multimodal_test_client.download_default_sample",
        fake_download,
    )

    result = ensure_sample_file(str(managed_path), "--audio-file")

    assert result == managed_path
    assert downloaded == [managed_path]
    assert managed_path.read_bytes() == b"audio"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_manual_multimodal_test_client.py::test_ensure_sample_file_downloads_missing_managed_default -v`
Expected: FAIL because `ensure_sample_file` does not yet recognize managed defaults or invoke `download_default_sample`.

- [ ] **Step 3: Write minimal implementation**

```python
def _managed_sample_key(path: Path) -> str:
    return path.as_posix()


def ensure_sample_file(path_value: str | None, flag_name: str) -> Path:
    if not path_value:
        raise FileNotFoundError(f"Missing required file. Supply {flag_name}.")
    path = Path(path_value)
    if path.exists():
        return path
    if _managed_sample_key(path) in SAMPLE_URLS:
        return download_default_sample(path)
    raise FileNotFoundError(f"File not found: {path}. Supply {flag_name}.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_manual_multimodal_test_client.py::test_ensure_sample_file_downloads_missing_managed_default -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_manual_multimodal_test_client.py tools/manual_multimodal_test_client.py
git commit -m "feat: auto-resolve missing managed sample files"
```

### Task 3: Add failing tests for explicit overrides and download failures

**Files:**
- Modify: `tests/test_manual_multimodal_test_client.py`
- Modify: `tools/manual_multimodal_test_client.py`
- Test: `tests/test_manual_multimodal_test_client.py`

- [ ] **Step 1: Write the failing tests**

```python
import subprocess


def test_ensure_sample_file_rejects_missing_explicit_override(tmp_path):
    missing_path = tmp_path / "custom.wav"

    with pytest.raises(FileNotFoundError, match=rf"File not found: {missing_path}\. Supply --audio-file\."):
        ensure_sample_file(str(missing_path), "--audio-file")


def test_download_default_sample_runs_wget_and_returns_path(tmp_path, monkeypatch):
    sample_path = tmp_path / "samples" / "transcription.wav"
    recorded = {}

    def fake_run(command, check):
        recorded["command"] = command
        recorded["check"] = check
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_bytes(b"downloaded-wav")

    monkeypatch.setattr(
        "tools.manual_multimodal_test_client.SAMPLE_URLS",
        {sample_path.as_posix(): "https://example.invalid/transcription.wav"},
    )
    monkeypatch.setattr("tools.manual_multimodal_test_client.subprocess.run", fake_run)

    result = download_default_sample(sample_path)

    assert result == sample_path
    assert recorded["command"] == [
        "wget",
        "-O",
        str(sample_path),
        "https://example.invalid/transcription.wav",
    ]
    assert recorded["check"] is True
    assert sample_path.read_bytes() == b"downloaded-wav"


def test_download_default_sample_wraps_wget_failure(tmp_path, monkeypatch):
    sample_path = tmp_path / "samples" / "question-video.mp4"

    def fake_run(command, check):
        raise subprocess.CalledProcessError(returncode=1, cmd=command)

    monkeypatch.setattr(
        "tools.manual_multimodal_test_client.SAMPLE_URLS",
        {sample_path.as_posix(): "https://example.invalid/question-video.mp4"},
    )
    monkeypatch.setattr("tools.manual_multimodal_test_client.subprocess.run", fake_run)

    with pytest.raises(FileNotFoundError, match=r"Unable to download required default file"):
        download_default_sample(sample_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_manual_multimodal_test_client.py -k "rejects_missing_explicit_override or runs_wget_and_returns_path or wraps_wget_failure" -v`
Expected: FAIL because `download_default_sample` does not yet run `wget` or convert failure into a clear file-resolution error.

- [ ] **Step 3: Write minimal implementation**

```python
import subprocess


def download_default_sample(path: Path) -> Path:
    key = _managed_sample_key(path)
    url = SAMPLE_URLS.get(key)
    if url is None:
        raise FileNotFoundError(f"No managed sample source configured for {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["wget", "-O", str(path), url], check=True)
    except subprocess.CalledProcessError as exc:
        raise FileNotFoundError(f"Unable to download required default file: {path}") from exc
    if not path.exists():
        raise FileNotFoundError(f"Unable to download required default file: {path}")
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_manual_multimodal_test_client.py -k "rejects_missing_explicit_override or runs_wget_and_returns_path or wraps_wget_failure" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_manual_multimodal_test_client.py tools/manual_multimodal_test_client.py
git commit -m "feat: add wget-based managed sample downloads"
```

### Task 4: Route request builders through managed sample resolution

**Files:**
- Modify: `tests/test_manual_multimodal_test_client.py`
- Modify: `tools/manual_multimodal_test_client.py`
- Test: `tests/test_manual_multimodal_test_client.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_build_request_spec_for_transcription_downloads_missing_default_audio(tmp_path, monkeypatch):
    config = {
        "mode": "transcription",
        "url": "http://127.0.0.1:6745",
        "model": "audio:test",
        "prompt": "Transcribe carefully",
        "output_dir": tmp_path,
        "token": None,
        "audio_file": "samples/transcription.wav",
        "video_file": None,
        "response_format": None,
    }
    downloaded_path = tmp_path / "samples" / "transcription.wav"

    monkeypatch.setattr(
        "tools.manual_multimodal_test_client.ensure_sample_file",
        lambda path_value, flag_name: downloaded_path,
    )
    downloaded_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded_path.write_bytes(b"wav-bytes")

    spec = build_request_spec(config)

    uploaded_name, uploaded_file = spec["files"]["file"]
    assert uploaded_name == "transcription.wav"
    assert uploaded_file.read() == b"wav-bytes"
    uploaded_file.close()


def test_build_request_spec_for_video_doubt_downloads_missing_default_video(tmp_path, monkeypatch):
    downloaded_path = tmp_path / "samples" / "question-video.mp4"
    downloaded_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded_path.write_bytes(b"video-bytes")
    config = {
        "mode": "video-doubt",
        "url": "http://127.0.0.1:6745",
        "model": "vision:test",
        "prompt": "What happens in this clip?",
        "output_dir": tmp_path,
        "token": None,
        "audio_file": None,
        "video_file": "samples/question-video.mp4",
        "response_format": None,
    }

    monkeypatch.setattr(
        "tools.manual_multimodal_test_client.ensure_sample_file",
        lambda path_value, flag_name: downloaded_path,
    )

    spec = build_request_spec(config)

    assert str(downloaded_path) in spec["json"]["messages"][0]["content"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_manual_multimodal_test_client.py -k "downloads_missing_default_audio or downloads_missing_default_video" -v`
Expected: FAIL because `build_request_spec` still calls `_require_file` directly.

- [ ] **Step 3: Write minimal implementation**

```python
def build_request_spec(config: dict) -> dict:
    mode = config["mode"]
    headers = {"Accept": "application/json"}
    if config.get("token"):
        headers["Authorization"] = f"Bearer {config['token']}"

    if mode == "transcription":
        audio_path = ensure_sample_file(config.get("audio_file"), "--audio-file")
        ...

    if mode == "video-doubt":
        video_path = ensure_sample_file(config.get("video_file"), "--video-file")
        ...

    if mode == "music-audio-doubt":
        audio_path = ensure_sample_file(config.get("audio_file"), "--audio-file")
        ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_manual_multimodal_test_client.py -k "downloads_missing_default_audio or downloads_missing_default_video" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_manual_multimodal_test_client.py tools/manual_multimodal_test_client.py
git commit -m "feat: use managed sample resolution in request builders"
```

### Task 5: Final verification for the client and tests

**Files:**
- Modify: `tests/test_manual_multimodal_test_client.py`
- Modify: `tools/manual_multimodal_test_client.py`
- Test: `tests/test_manual_multimodal_test_client.py`

- [ ] **Step 1: Run the full focused test file**

Run: `"/storage/coderai/venv_all/bin/python" -m pytest tests/test_manual_multimodal_test_client.py -v`
Expected: PASS with all tests green.

- [ ] **Step 2: Review the final diff for scope**

Run: `git diff -- tools/manual_multimodal_test_client.py tests/test_manual_multimodal_test_client.py`
Expected: Diff shows only managed sample download logic, tests, and any required imports/constants.

- [ ] **Step 3: Commit the completed feature**

```bash
git add tools/manual_multimodal_test_client.py tests/test_manual_multimodal_test_client.py
git commit -m "feat: auto-download managed multimodal sample media"
```
