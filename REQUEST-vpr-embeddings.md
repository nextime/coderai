# Request: serve a visual place recognition (VPR) model

**From:** HomeHunter (property search, `192.168.42.44`)
**To:** whoever maintains CoderAI (`192.168.42.3:8000`)
**Status:** request — the HomeHunter side is already built and waiting on the model
**Follows:** `REQUEST-geoclip-embeddings.md` (delivered — thank you, it works)

## What we need in one line

One model id on the existing `POST /v1/embeddings` endpoint — **`mixvpr`** (or
SALAD/EigenPlaces, your choice) — that turns an image into a single vector
trained so that **two photos of the same place land close together**.

Same request shape as `dinov2-large` already uses. Nothing else changes.

## Why, and why not the models you already serve

We are trying to find a listing's **actual street address** by matching its
exterior photo against Google Street View panoramas: the matching panorama's own
coordinate *is* the address. The matching code, the panorama cache and the spend
caps are already built and deployed. The only missing piece is a model that can
reliably answer "is this the same building?"

Nothing currently on the gateway answers that question:

| Model | What it actually does | Why it fails here |
|---|---|---|
| **dinov2-large** | generic self-supervised features | Rates *similar-looking* houses as matches. We measured this: nearest-neighbour retrieval over DINOv2 vectors placed one listing **288 km** from truth. |
| **gme-Qwen2-VL** | text↔image semantic retrieval | Matches content categories ("a house with a pool"), not building identity. |
| **geoclip** (just added) | image → global GPS prior | Separates continents, not streets. Measured: a Cape Town photo scores Cape Town **0.190**, London **-0.064**, but a point **16 km away in the same metro still 0.185**. Median error 1.98 km against a 1.17 km baseline — worse than doing nothing. Now disabled. |

VPR models are trained specifically for the discrimination the others lack:
same place under different viewpoint, lighting, weather and season.

## The contract

```http
POST /v1/embeddings
Authorization: Bearer <token>
Content-Type: application/json

{"model": "mixvpr", "input": "data:image/jpeg;base64,/9j/4AAQ...", "dimensions": 4096}
```

```json
{"data": [{"embedding": [0.021, -0.008, ...], "index": 0}], "model": "mixvpr"}
```

Exactly the shape `dinov2-large` already answers in. We send a JPEG data URI
(downscaled to 1024 px on our side) and expect one vector back.

## Requirements that matter

1. **L2-normalise the output.** We compare with a plain dot product.
2. **Tell us the true width** and keep it stable — MixVPR is typically 4096,
   SALAD 8448, EigenPlaces 2048. We set it in config and validate the response
   length, so a mismatch is rejected rather than silently stored.
3. **Deterministic.** Same image, same vector. We cache panorama embeddings by
   location and reuse them across listings.
4. **One vector per image** (a pooled/aggregated descriptor, not patch tokens).

Any of these are fine, in rough order of preference:

| Model | Dim | Note |
|---|---|---|
| **MixVPR** | 4096 | Strong, small, single forward pass. Good default. |
| **SALAD** (DINOv2-SALAD) | 8448 | Currently near state of the art; heavier. |
| **EigenPlaces / CosPlace** | 512–2048 | Lighter, still far better than raw DINOv2. |
| AnyLoc | — | Only if patch tokens are easy for you: it is DINOv2 features + VLAD, so it could reuse the DINOv2 you already serve. |

All are open weights on GitHub/HuggingFace and much smaller than GeoCLIP.

## How we will verify it

The decisive test is **ordering**, not absolute scores. Given a listing's
exterior photo:

```
similarity(listing photo, Street View AT that address)
    >  similarity(listing photo, Street View 200 m down the same street)
    >> similarity(listing photo, Street View in a different suburb)
```

DINOv2 fails this — it ranks any pleasant suburban facade about equally. If the
model you pick reproduces that ordering, it is the right one.

Then the real test, which is ours to run: Street View matching is measured
against ground truth (listings whose source published a coordinate, re-located
with that coordinate hidden). The bar is the **suburb-centroid baseline, 1.17 km
median**. GeoCLIP was wired in, measured, and left disabled for failing it. VPR
gets the same treatment — no model ships on plausibility.

## What we do not need

- No new endpoint, auth, or streaming.
- No changes to any existing model.
- Not a VLM or a captioner — we are not asking it to describe anything.

## What HomeHunter is already configured for

The `vpr` provider is set up and pointing at the **same gateway DINOv2 uses**
(`http://192.168.42.3:8000/v1`, plain model ids like `dinov2-large-Q8_0.gguf` and
`geoclip`), currently expecting:

    model_id: salad        dimensions: 8448     <- active
    model_id: eigenplaces  dimensions: 2048     <- alternative, swap both fields

Serving either id is enough; the model id and width are editable in the admin, so
any other place-recognition model works too as long as the pair matches. If the
model is missing the matcher logs it and falls back to DINOv2 rather than
disabling Street View matching, so configuring ahead of you is harmless.

## Contact

HomeHunter side: `/working/homehunter`. The matcher is
`app/services/streetview.py` (already switches to a `vpr` provider the moment one
is configured, falling back to DINOv2 otherwise), the design rationale is
`docs/geolocation-visual-matching-design.md`, and accuracy is measured by
`scripts/eval_locations.py`.
