"""Step 7: context-parallel multi-GPU, and the pod image that can serve LongCat.

Two things make this more than a flag:

1. Context parallelism is N PROCESSES under torchrun, not N threads, and every rank must
   enter the same pipeline call or the collectives deadlock. So rank 0 owns the socket and
   broadcasts each job; the other ranks sit in a loop waiting for one. A rank-0-only
   server that simply called generate() would hang on the first collective.
2. The generic RunPod capability image is python:3.12 on the cu128 torch index. LongCat
   needs 3.10 on cu124, so the 'video' image cannot serve it at all — it would boot, pass
   its health check, accept a request and have nothing to run it with.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codai.api import longcat_worker as W
from codai.api.runpod_worker import (PUBLISHED_CAPABILITY_IMAGES,
                                     default_capability_image, model_capability)

SERVICE = (ROOT / "tools/longcat_service.py").read_text()
WORKER = (ROOT / "codai/api/longcat_worker.py").read_text()
PODFILE = (ROOT / "packaging/runpod/Dockerfile.capability-video-longcat").read_text()


# ---------------------------------------------------------------- context parallel
def test_the_service_accepts_a_context_parallel_size():
    assert "--context-parallel-size" in SERVICE


def test_nccl_and_context_parallel_are_initialised_before_the_model_loads():
    """init_context_parallel has to run before the DiT is built — it is what makes the
    DiT split its spatial dims — so load_pipeline must not be called first."""
    main = SERVICE[SERVICE.index("def main(argv=None):"):]
    assert main.index("cp_init(cp_size)") < main.index("load_pipeline(")
    assert "init_context_parallel" in SERVICE
    assert 'backend="nccl"' in SERVICE


def test_only_rank_zero_binds_the_socket():
    """Every rank binding the port would collide; every rank serving would mean N
    different answers to one request."""
    main = SERVICE[SERVICE.index("def main(argv=None):"):]
    assert "if rank != 0:" in main
    assert main.index("if rank != 0:") < main.index("ThreadingHTTPServer")


def test_other_ranks_wait_for_a_broadcast_job():
    assert "def cp_worker_loop" in SERVICE
    assert "broadcast_object_list" in SERVICE


def test_rank_zero_broadcasts_before_generating():
    """This is the deadlock: rank 0 must publish the job BEFORE entering the pipeline,
    because the other ranks have to be in the same call."""
    handler = SERVICE[SERVICE.index('if self.path.startswith("/generate")'):]
    handler = handler[:handler.index("elif")]
    assert handler.index("cp_broadcast(job)") < handler.index("generate(job)")


def test_the_group_is_told_to_shut_down():
    """Otherwise the non-zero ranks outlive the server and the process never exits."""
    assert "_CP_SHUTDOWN" in SERVICE
    main = SERVICE[SERVICE.index("def main(argv=None):"):]
    assert "cp_broadcast(dict(_CP_SHUTDOWN))" in main


def test_a_single_gpu_does_not_initialise_distributed():
    """cp_size 1 is the default path and must not need NCCL at all."""
    main = SERVICE[SERVICE.index("def main(argv=None):"):]
    assert "if cp_size > 1:" in main
    cp = SERVICE[SERVICE.index("def cp_init"):SERVICE.index("def cp_broadcast")]
    assert "if cp_size <= 1:" in cp and "return rank, local_rank" in cp


def test_the_worker_launches_under_torchrun_only_when_asked():
    launch = WORKER[WORKER.index("cp = 0"):WORKER.index('"--model", model_path')]
    assert "torch.distributed.run" in launch
    assert "nproc_per_node" in launch
    assert "if cp > 1:" in launch and "launcher = [str(py)]" in launch


def test_asking_for_more_ranks_than_gpus_is_refused_early():
    """NCCL's own failure for this names nothing useful."""
    launch = WORKER[WORKER.index("cp = 0"):WORKER.index('"--model", model_path')]
    assert "_visible_gpu_count" in launch
    assert "lower it" in launch


@pytest.mark.parametrize("env,expected", [
    ({"CUDA_VISIBLE_DEVICES": "0,1,2"}, 3),
    ({"CUDA_VISIBLE_DEVICES": "0"}, 1),
    ({"CUDA_VISIBLE_DEVICES": ""}, None),      # falls back to a real query
])
def test_the_visible_gpu_count_follows_the_selector(env, expected):
    got = W._visible_gpu_count(env)
    if expected is None:
        assert got >= 0
    else:
        assert got == expected


def test_cp_size_comes_from_the_model_entry():
    """It is a per-model setting with a UI control, saved and whitelisted in step 5."""
    html = (ROOT / "codai/admin/templates/models.html").read_text()
    assert "cfg-longcat-cp" in html
    routes = (ROOT / "codai/admin/routes.py").read_text()
    block = routes[routes.index('for key in ("alias", "config_name", "backend"'):]
    assert "cp_size" in block[:block.index("):")]


# ---------------------------------------------------------------- the pod image
def test_the_image_is_registered():
    """A capability absent from this tuple has no image and demands an explicit one."""
    assert "video-longcat" in PUBLISHED_CAPABILITY_IMAGES
    assert default_capability_image("video-longcat").endswith(
        "coderai-video-longcat:latest")


def test_a_longcat_model_is_not_placed_on_the_plain_video_image():
    """The 'video' image is python:3.12/cu128 with diffusers — it cannot run LongCat at
    all. Placing it there gives a pod that boots, passes health and then has nothing to
    generate with."""
    assert model_capability({"model_type": "video_models",
                             "backend": "longcat"}) == "video-longcat"
    assert model_capability({"model_type": "video_models",
                             "backend": "auto"}) == "video"


def test_an_explicit_capability_still_wins():
    """The operator's override must stay authoritative."""
    assert model_capability({"model_type": "video_models", "backend": "longcat",
                             "capability": "video"}) == "video"


def test_the_profile_uses_the_custom_dockerfile_escape_hatch():
    base = ROOT / "packaging/runpod/profiles"
    assert (base / "video-longcat.txt").is_file()
    assert (base / "video-longcat.light").is_file()       # light core: no torch in main
    assert (base / "video-longcat.dockerfile").read_text().strip() == \
        "Dockerfile.capability-video-longcat"


def test_the_image_builds_the_310_venv_from_the_same_requirements_file():
    """One source of truth: the pod and a local install must not drift."""
    assert "requirements-longcat.txt" in PODFILE
    assert "python3.10 -m venv" in PODFILE


def test_the_image_verifies_the_venv_rather_than_hoping():
    """A pip install that resolved differently would otherwise surface on a rented pod."""
    assert "(3,10)" in PODFILE and "'2.6'" in PODFILE and "'4.41'" in PODFILE


def test_the_image_clones_the_pipeline_source_and_checks_it_imports():
    """`longcat_video` is in the repo, not on PyPI."""
    assert "meituan-longcat/LongCat-Video" in PODFILE
    assert "import longcat_video.pipeline_longcat_video" in PODFILE
    assert "source_problems" in PODFILE


def test_the_image_bakes_no_weights():
    """~83 GB across variants, and the rule is that released images carry no models."""
    assert "--depth 1" in PODFILE
    assert "rm -rf /opt/coderai/LongCat-Video/.git" in PODFILE
    assert not re.search(r"huggingface-cli download|hf_hub_download", PODFILE)


def test_the_image_points_the_worker_at_what_it_baked():
    assert "CODERAI_LONGCAT_VENV=/opt/coderai/longcat_venv" in PODFILE
    assert "CODERAI_LONGCAT_SRC=/opt/coderai/LongCat-Video" in PODFILE


def test_those_env_vars_are_the_ones_the_worker_reads():
    """Seeded and read must agree, or the pod silently builds its own venv."""
    assert "CODERAI_LONGCAT_VENV" in WORKER
    assert "CODERAI_LONGCAT_SRC" in WORKER


def test_the_image_answers_a_health_check():
    assert "HEALTHCHECK" in PODFILE and "/healthz" in PODFILE
