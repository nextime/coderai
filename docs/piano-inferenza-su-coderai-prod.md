# Istruzioni per l'agente coderai — tutta l'inferenza sotto la coderai di produzione

> **Decisione (09/10/2026).** Tutti i modelli che oggi girano **su RunPod** e
> **sulla coderai di zeiss** vanno spostati a girare **orchestrati dalla coderai
> di produzione** (ru002848). La coderai di prod diventa l'**unico piano di
> controllo** e l'**unico endpoint OpenAI-compatibile** (`/v1`): non ha GPU,
> quindi **orchestra pod RunPod A40 on-demand** (come fanno oggi i launcher di
> Digesta `app/casi_pod.py` e `app/ocr_pod.py`, che vanno assorbiti qui).
>
> **Uniche eccezioni che NON si spostano:**
> 1. i **reranker locali** di Digesta (restano in-container, vedi sotto);
> 2. il modello **cloud Opus** (`claude-opus-5`) usato nelle chat (resta su Anthropic).
>
> I modelli cloud di terzi (OpenAI `gpt-5.6-luna`, Mistral `mistral-ocr-latest`,
> OpenAI `text-embedding-3-small`) **non fanno parte di questo spostamento**: non
> sono «su RunPod o coderai», non sono auto-ospitabili; cosa farne si decide a
> parte (vedi §4).

Chi avvia materialmente l'orchestrazione dei pod lo dice l'utente/gli altri
agenti; **qui c'è solo la configurazione richiesta alla coderai di prod**.

---

## 1. Obiettivo

- **Un solo `/v1`**: `ru002848` coderai (porta interna 8776, pubblicata
  `127.0.0.1:8777`, davanti il proxy SSO di Digesta su `digesta.tech/coderai/`;
  l'API `/v1` resta coi suoi bearer). Tutti i container chiamano **questo**.
- **coderai = control plane**: accende/spegne pod **RunPod A40** a richiesta e
  instrada le richieste; nessuna GPU locale a ru002848. Riusare la logica di
  `casi_pod.py`/`ocr_pod.py` (served-model-name, regioni, `RUNPOD_API_KEY`,
  ledger) dentro coderai.

### Dove stanno i pesi
**Tutti i file dei modelli vivono su uno storage RunPod** (network volume
persistente): i GGUF (`lisa`/gemma), le basi HF/AWQ (Qwen, Llama, olmOCR), gli
adattatori **QLoRA** (`sib4` e futuri), i modelli Surya. I pod A40 che coderai
accende **montano quello storage** e caricano da lì — niente download da zeiss
né ri-upload a ogni avvio. coderai deve sapere il path di ciascun modello/
adattatore su quel volume (non più percorsi locali di zeiss tipo
`/AI/guffcache/…`).

## 2. Modelli che la coderai di prod deve servire (orchestrando RunPod)

| served-model-name | motore | pesi / fonte | quant | GPU | ruolo | consumatori |
|---|---|---|---|---|---|---|
| `qwen38-awq` (+ alias `qwen38`) | vLLM | `cyankiwi/Qwen3.8-27B-AWQ-INT4` (HF) | AWQ int4 | A40 | classificazione L2 (`modello`), estrazione casi, chat | Digesta (`classificazione_l2`, `estrazione_casi`, compositore `coderai-qwen38`), verti/omnia `coderai-qwen38` |
| `diogene-alpha` | vLLM + LoRA | base `cyankiwi/Qwen3.8-27B-AWQ-INT4` **+ adattatore QLoRA `sib4`** (sullo storage RunPod) | AWQ int4 + LoRA | A40 (può condividere la base con `qwen38-awq` via multi-LoRA) | il nostro **reader addestrato** per l'estrazione casi | Digesta (`estrazione_casi` modo `alpha`), verti/omnia `diogene-alpha`. **Assorbe anche il ruolo di alpha-ctl :8628** (accensione/stato del pod) |
| `lisa` e `lisa-32k` | llama.cpp (GGUF), `parser: gemma`, **multimodale** (mmproj) | `gemma-4-26B-A4B-it-qat-q4_0-uncensored-heretic-Q4_0.gguf` + `…-mmproj-BF16.gguf` (sullo storage RunPod) | GGUF Q4_0 | A40 | chat/RAG con visione; `lisa`=178k ctx, `lisa-32k`=32k ctx | Digesta compositore `coderai-lisa`, verti/omnia `coderai-lisa` |
| `Gemma-4-14B-A4B-Heretic.i1-IQ4_XS.gguf` | llama.cpp (GGUF) | file omonimo | GGUF IQ4_XS | A40 | chat/RAG più piccolo/veloce | Digesta compositore `coderai-gemma14-heretic`, verti/omnia idem |
| `allenai/olmOCR-2-7B-1025-FP8` | vLLM | HF omonimo | FP8 | A40 | OCR vision di ripiego | Digesta OCR (ripiego). **Assorbe `ocr_pod.py`** |
| `surya` (servizio OCR) | server Surya (`/v1/ocr`) | modelli Surya | — | A40 | OCR primario | Digesta OCR (prima passata) |
| `gemma-4-31B` | vLLM | Gemma-4-31B | AWQ 8-bit | A40 | leg del **trio** di estrazione campi / generazione del secondo gold-set | Digesta `estrazione` (trio) |
| `llama-3.3-70b` | vLLM | Llama-3.3-70B | AWQ | A40 | leg del **trio** + opzione chat | Digesta `estrazione` (trio) e compositore `serverless`; verti/omnia `runpod`. **Unifica il pod-trio E il serverless RunPod in un solo modello orchestrato** |
| `bge-m3` | backend embedding (vLLM / FlagEmbedding) | `BAAI/bge-m3` | fp16 | GPU piccola o CPU | **embedding** (dense 1024 + sparse) del corpus e delle query | Digesta (indicizzazione + query) |
| `BAAI/bge-reranker-v2-m3` | HF seq-classification | `BAAI/bge-reranker-v2-m3` (storage RunPod o pull HF) | — | **CPU, locale su ru002848** (non RunPod) | reranker profondo | Digesta (reranking, off di default) |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | HF seq-classification | id omonimo | — | **CPU, locale su ru002848** | reranker veloce | Digesta (reranking, off di default) |

> **Reranker su coderai in CPU.** Oggi girano locali su CPU dentro il container
> Digesta; siccome coderai-prod è anch'essa una macchina CPU, si servono **lì**
> in locale (CPU), senza accendere un pod RunPod. Così anche i reranker passano
> per l'unico `/v1`.

Note:
- **Il trio NON si ritira** (richiesta esplicita): `gemma-4-31B`, `llama-3.3-70b`
  e `qwen38-awq` restano disponibili, ma orchestrati da coderai. Il reader
  addestrato `diogene-alpha` resta il singolo modello di punta per i campi già
  coperti; il trio serve per i nuovi campi (secondo gold-set).
- **`qwen38` / `Qwen3.8-27B` del trio** = lo stesso checkpoint `qwen38-awq`: un
  solo modello servito, non tre istanze diverse.
- **`llama-3.3-70b`**: oggi esiste in due posti (pod A40 del trio + endpoint
  RunPod *serverless*); vanno unificati in **un** modello servito da coderai.

## 3. NON spostare (unica eccezione)

- **`claude-opus-5`** (Anthropic), resta cloud: è il modello di chat di
  default e il teacher dell'addestramento di `diogene-alpha`.

(I reranker, prima eccezione, **passano anch'essi a coderai** ma in CPU locale su
ru002848 — vedi §2.)

## 4. Cloud di terzi — DISMESSI (09/10/2026)

**Non si usano più, nessun impiego.** L'unico modello cloud che resta è
`claude-opus-5` (§3).
- `gpt-5.6-luna` (OpenAI) — **via**: il compositore `openai` di Digesta e il
  profilo `openai` di verti/omnia diventano morti.
- `mistral-ocr-latest` (Mistral) — **via**: la catena OCR cloud di ultima
  istanza resta solo su Anthropic (`claude-opus-5`): surya → olmOCR → opus.
- `text-embedding-3-small` (OpenAI) — **via**: sostituito da `bge-m3`;
  `DIGESTA_EMBED_MODEL` va messo a `bge-m3` (niente più segnaposto OpenAI).

Al cutover di Digesta: togliere/lasciare non valorizzati `OPENAI_*` e
`MISTRAL_API_KEY`, l'opzione `OCR_CLOUD=mistral`, la voce compositore `openai`;
in verti/omnia il profilo `openai` sparisce dal catalogo utile.

## 4b. Ritiro di alpha-ctl (dopo il cutover)

**alpha-ctl** (il controllore della scheda `diogene-alpha` su `127.0.0.1:8628`:
accensione/stato/spegnimento del pod) **va ritirato** una volta che coderai
orchestra il pod `diogene-alpha`: quel ruolo (power/stato) lo fa coderai. È un
passo **sequenziato** — si spegne alpha-ctl solo quando `diogene-alpha` è servito
e verificato su coderai-prod. Pulizie collegate:
- **verti/omnia**: l'impostazione `diogene_alpha_controllore` (default
  `http://127.0.0.1:8628`) diventa inutile → rimuoverla; resta solo
  `diogene_alpha_url` puntato al `/v1` di coderai.
- **mcp-aggregator**: togliere il backend che espone alpha-ctl
  (`registra_rest.py` upstream `http://127.0.0.1:8628`, doc `GATEWAY_REST.md`).
- spegnere il servizio alpha-ctl su ru002848 e il relativo proxy
  `127.0.0.1:8628/v1`.

## 5. Ripuntamento dei consumatori (lo guida l'utente/gli altri agenti)

Quando i modelli sono su coderai-prod, ripuntare:
- **Digesta** → tutti al `/v1` di coderai-prod: `DIGESTA_LLM_BASE_URL`
  (embedding `bge-m3` + base chat), `DIGESTA_OCR_BASE_URL` (surya),
  `DIGESTA_OCR_VLM_BASE_URL` (olmOCR), `DIGESTA_CASI_ALPHA_URL` /
  `DIGESTA_CASI_ESTRATTORE_*`, `SERVERLESS_BASE_URL`+`SERVERLESS_MODEL`
  (`llama-3.3-70b`), `coderai_url` dei compositori; impostare
  `DIGESTA_EMBED_MODEL=bge-m3`.
- **verti / omnia / groupama** → impostazioni admin: `coderai_url`,
  `diogene_alpha_url`, `runpod_url` tutti al `/v1` di coderai-prod.
- **sibilla / vitruvio / veridoc**: nessuna modifica (delegano a Digesta).
