"""A spark2_5 base must have its ATTENTION adapted, not only its MLP.

Spark-X2.5 names its projections ``q_k_v_proj`` (fused QKV), ``out_proj`` and
``g_proj`` (the head-wise attention gate) -- none of which are the Llama/Qwen
names the target selector knew. The failure this guards is silent: with those
names absent from the suffix set, ``select_target_modules`` returns only the
MLP's ``gate/up/down_proj``, the LoRA trains attention not at all, and the run
still reports success with a weaker adapter nobody was told about.

The vision check is the other half: extending the suffix set must not start
adapting a vision tower's ``out_proj`` (a name a VLM's attention also uses).
"""

from codai.api.text_lora_trainer import select_target_modules, LANGUAGE_SUFFIXES


SPARK_LINEARS = [
    "model.layers.0.self_attn.q_k_v_proj",
    "model.layers.0.self_attn.out_proj",
    "model.layers.0.self_attn.g_proj",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.0.mlp.down_proj",
    "model.embed_tokens",     # never adapted
    "lm_head",                # never adapted
]


def test_the_fused_qkv_output_and_gate_are_all_selected():
    picked = set(select_target_modules(SPARK_LINEARS))
    assert "model.layers.0.self_attn.q_k_v_proj" in picked
    assert "model.layers.0.self_attn.out_proj" in picked
    assert "model.layers.0.self_attn.g_proj" in picked
    # and the MLP, as before
    assert "model.layers.0.mlp.gate_proj" in picked
    assert "model.layers.0.mlp.down_proj" in picked


def test_attention_is_actually_in_the_set_not_just_the_mlp():
    """The regression: MLP-only is the wrong-but-passing outcome."""
    picked = select_target_modules(SPARK_LINEARS)
    attn = [n for n in picked if "self_attn" in n]
    assert len(attn) == 3, f"expected all 3 attention projections, got {attn}"


def test_the_head_and_embeddings_are_left_alone():
    picked = set(select_target_modules(SPARK_LINEARS))
    assert "lm_head" not in picked
    assert "model.embed_tokens" not in picked


def test_a_vision_towers_out_proj_is_not_adapted():
    """`out_proj` is also a vision attention name; the vision filter is by
    dotted-path marker, so it must still exclude it."""
    linears = SPARK_LINEARS + [
        "visual.blocks.0.attn.out_proj",
        "vision_tower.encoder.layers.0.self_attn.out_proj",
    ]
    picked = set(select_target_modules(linears))
    assert "visual.blocks.0.attn.out_proj" not in picked
    assert "vision_tower.encoder.layers.0.self_attn.out_proj" not in picked
    # the language out_proj is still in
    assert "model.layers.0.self_attn.out_proj" in picked


def test_an_explicit_suffix_override_pins_exactly_that_set():
    """`target_modules` on the request becomes the suffix set, so an operator
    can reproduce a published adapter's targets."""
    only_qkv = select_target_modules(SPARK_LINEARS, suffixes=("q_k_v_proj",))
    assert only_qkv == ["model.layers.0.self_attn.q_k_v_proj"]


def test_the_new_names_are_declared_once():
    for name in ("q_k_v_proj", "out_proj", "g_proj"):
        assert name in LANGUAGE_SUFFIXES
