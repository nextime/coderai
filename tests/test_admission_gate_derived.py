"""An OCR install must not have to know server.max_parallel_requests exists.

The shipped default is 2 in flight / 6 queued, which is right for one local
card answering chat and is the throughput ceiling for an OCR install
dispatching pages to a rented fleet. Measured on one A40: the pod sat at
`Running: 1, Waiting: 0` -- asking for work and not getting it -- and 219 pages
took 1210s, the same as reading them one at a time, while ocr.max_concurrency
said 4096 and the pod was handed 64 of its own.

So when OCR is enabled and the operator has not written the key, the gate is
derived from the page budget they did ask for. The rule these tests pin, in
order of how much damage getting it wrong would do:

  1. an EXPLICIT value is obeyed, however small -- deriving over a deliberate
     choice would be worse than the bug being fixed
  2. an env override is obeyed for the same reason
  3. a chat-only install keeps the default of 2
  4. the derived value is bounded, because a slot costs memory
"""
import pytest


def derive_gate(ocr_enabled, page_budget, explicit_keys, env_override,
                current=2):
    """Mirror of the rule in codai/main.py."""
    if env_override:
        return current
    if not ocr_enabled or not page_budget or page_budget <= 0:
        return current
    if "max_parallel_requests" in explicit_keys:
        return current
    derived = max(16, min(256, page_budget // 8))
    return derived if derived > current else current


def derive_queue(derived, explicit_keys, env_override, current=6):
    if env_override or "queue_max_size" in explicit_keys:
        return current
    return min(2048, derived * 8)


# --- 1. an explicit value wins, always ---

def test_an_explicit_small_gate_is_obeyed():
    """An operator who wrote 2 meant 2."""
    assert derive_gate(True, 4096, {"max_parallel_requests"}, None) == 2


def test_an_explicit_gate_is_obeyed_even_when_absurdly_low():
    assert derive_gate(True, 4096, {"max_parallel_requests"}, None, current=1) == 1


def test_other_explicit_server_keys_do_not_block_the_derivation():
    """Writing `host` must not be read as an opinion about concurrency."""
    assert derive_gate(True, 4096, {"host", "port"}, None) == 256


# --- 2. env overrides win (a pod is configured only through env) ---

def test_an_env_override_is_obeyed():
    assert derive_gate(True, 4096, set(), "32", current=32) == 32


# --- 3. a chat-only install is unchanged ---

def test_ocr_disabled_keeps_the_shipped_default():
    assert derive_gate(False, 4096, set(), None) == 2


def test_no_page_budget_keeps_the_shipped_default():
    assert derive_gate(True, 0, set(), None) == 2
    assert derive_gate(True, None, set(), None) == 2


# --- 4. the derived value is bounded (a slot holds its whole upload) ---

@pytest.mark.parametrize("budget,expect", [
    (4096, 256),      # the observed production case, capped
    (99999, 256),     # never unbounded, whatever the page budget says
    (128, 16),        # a small budget still gets a usable floor
    (8, 16),          # the floor applies rather than deriving 1
    (2048, 256),
    (1024, 128),
])
def test_derived_gate_is_bounded_both_ends(budget, expect):
    assert derive_gate(True, budget, set(), None) == expect


def test_the_derivation_never_lowers_an_existing_higher_value():
    assert derive_gate(True, 128, set(), None, current=200) == 200


# --- the queue follows the gate, unless it too was set ---

def test_queue_follows_the_derived_gate():
    assert derive_queue(256, set(), None) == 2048


def test_an_explicit_queue_is_obeyed():
    assert derive_queue(256, {"queue_max_size"}, None, current=6) == 6


def test_queue_is_capped():
    assert derive_queue(256, set(), None) <= 2048


# --- the rule above must be the rule main.py applies ---

def test_main_applies_the_same_derivation():
    src = open("codai/main.py").read()
    assert 'max(16, min(256, _page_budget // 8))' in src
    assert '"max_parallel_requests" not in _explicit' in src
    assert 'not _mp' in src, "env override must still win"
    block = src[src.index("An OCR install must not have to know"):]
    block = block[:block.index("# Say so when this gate is the ceiling")]
    assert "except Exception:" in block, "must never stop a server booting"


def test_config_manager_records_explicit_server_keys():
    """The derivation is only safe if 'was it written' is actually known."""
    src = open("codai/config.py").read()
    assert "self.explicit_server_keys" in src
    # and it must be set on BOTH paths, or a missing config.json leaves a stale set
    assert src.count("self.explicit_server_keys") >= 2
