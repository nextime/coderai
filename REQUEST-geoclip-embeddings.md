# Request: serve GeoCLIP through the OpenAI embeddings API

**From:** HomeHunter (property search, `192.168.42.44`)
**To:** whoever maintains CoderAI (`192.168.42.3:8000`)
**Status:** request — nothing is built on the HomeHunter side yet that depends on it

## What we need in one line

Two new model ids on the existing `POST /v1/embeddings` endpoint — `geoclip` for
images and `geoclip-location` for coordinates — both returning 512-dimension
L2-normalised vectors in the **same** space.

No new endpoint, no custom protocol. If you serve those two ids, HomeHunter can
use them with the client it already has.

## Why

HomeHunter locates listings whose source published no coordinate. Today it asks a
vision model to *describe* each photo, then geocodes the words. That loses almost
all the signal: only photos containing something nameable (a street sign, a
famous landmark) produce anything, and a name geocodes to a coarse or ambiguous
point. Measured against ground truth, the text pipeline places **1.7%** of
listings.

GeoCLIP removes the text hop. It embeds images and GPS coordinates into one
shared space, so a photo can be scored *directly* against candidate coordinates —
the match itself is the answer. We would build a grid of candidate points over
the listing's suburb, embed the photo once, and take the highest-scoring cell.

We already tried the cheap alternative and it failed: nearest-neighbour retrieval
over DINOv2 vectors fired on 2.5% of listings and missed by up to 288 km, because
a *similarity* model is not a *location* model. That is why the ask is
specifically GeoCLIP and not "some image embedding".

## The contract

### 1. Image model — `geoclip`

Identical in shape to how `dinov2-large` is already served.

```http
POST /v1/embeddings
Authorization: Bearer <token>
Content-Type: application/json

{"model": "geoclip", "input": "data:image/jpeg;base64,/9j/4AAQ...", "dimensions": 512}
```

```json
{"data": [{"embedding": [0.013, -0.044, ...], "index": 0}], "model": "geoclip"}
```

### 2. Location model — `geoclip-location`

The only unusual part: the "text" being embedded is a coordinate pair. The wire
format stays completely standard.

```http
POST /v1/embeddings

{"model": "geoclip-location",
 "input": ["-33.9249,18.4241", "-33.9270,18.4300", "-33.9310,18.4185"],
 "dimensions": 512}
```

```json
{"data": [{"embedding": [...], "index": 0},
          {"embedding": [...], "index": 1},
          {"embedding": [...], "index": 2}], "model": "geoclip-location"}
```

Input format: `"<latitude>,<longitude>"` in decimal degrees, WGS84. Please accept
optional surrounding whitespace. A single string (not an array) should also work.

## Requirements that actually matter

1. **Same space.** The two models must be the image and location encoders of the
   *same* GeoCLIP checkpoint. Vectors from different checkpoints are not
   comparable and the scores become meaningless without erroring — the worst
   kind of failure.
2. **L2-normalise both.** We compare with a plain dot product. If you do not
   normalise, say so and we will normalise our side instead — but please pick one
   and document it.
3. **Batching on the location model.** This is the one that matters for cost: we
   score a few hundred candidate cells per listing and want that in **one**
   request, not one per cell. `input` as an array must return one vector per
   element, in input order.
4. **`index` must reflect input order** (or at least be present so we can sort by
   it). We map results back to grid cells positionally.
5. **Deterministic.** Same input, same vector. We cache by coordinate.

Nice to have, not required: report the true width in `usage`/`dimensions`, and
list both ids in `GET /v1/models` so we can detect availability.

## Reference implementation

The `geoclip` pip package ships both encoders and the weights (~1.7 GB, first
load downloads them).

```python
import torch, torch.nn.functional as F
from geoclip import GeoCLIP

model = GeoCLIP().to(device).eval()

@torch.inference_mode()
def embed_image(pil_image):
    batch = model.image_encoder.preprocess_image(pil_image).to(device)
    return F.normalize(model.image_encoder(batch), dim=-1)[0].tolist()   # 512 floats

@torch.inference_mode()
def embed_locations(pairs):            # [(lat, lon), ...]
    gps = torch.tensor(pairs, dtype=torch.float32, device=device)
    return F.normalize(model.location_encoder(gps), dim=-1).tolist()     # [[512 floats], ...]
```

The location encoder is tiny (an MLP over random Fourier features) — it is the
image encoder that carries the weight and wants the GPU. If it helps, a fuller
sketch including a FastAPI wrapper is in the HomeHunter repo at
`deploy/geoclip/app.py`; it was written before we settled on serving this through
CoderAI, so treat it as reference for the two torch calls only.

## How we will verify it

```sh
# 1. image side returns 512 dims
curl -s $GW/v1/embeddings -H "Authorization: Bearer $TOK" \
  -d '{"model":"geoclip","input":"data:image/jpeg;base64,'"$(base64 -w0 photo.jpg)"'"}' \
  | jq '.data[0].embedding | length'

# 2. location side batches, in order
curl -s $GW/v1/embeddings -H "Authorization: Bearer $TOK" \
  -d '{"model":"geoclip-location","input":["-33.92,18.42","-33.90,18.60"]}' \
  | jq '[.data[] | {index, n: (.embedding|length)}]'
```

Then the real test, which is ours to run: a photo taken in a known place should
score its true location higher than a location 20 km away. If that ordering does
not hold, the two encoders are not in the same space.

After that we measure it against ground truth (listings whose source published a
coordinate, re-located with that coordinate hidden). The bar is the
suburb-centroid baseline, currently **~1.1 km median error**. If GeoCLIP cannot
beat that, we will not ship it — the same way we rejected the DINOv2 attempt.

## What we do not need

- No `/locate` or top-k prediction endpoint. GeoCLIP's built-in worldwide gallery
  answers "where on Earth", but our listings already claim a suburb; the useful
  question is *where inside that area*, which we answer ourselves by scoring our
  own grid.
- No new auth, no streaming, no changes to any existing model.

## Contact

Questions about the HomeHunter side, the grid, or the evaluation: ask in the
HomeHunter repo (`/working/homehunter`, see
`docs/geolocation-visual-matching-design.md` for the full design and
`app/services/geo_eval.py` for how accuracy is measured).
