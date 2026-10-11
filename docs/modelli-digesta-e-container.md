# I modelli di Digesta e delle altre postazioni — cosa deve servire coderai

> Documento per l'agente di **coderai**. Elenca tutti i modelli (LLM, embedding,
> reranker, OCR) usati da Digesta e dai container fratelli, distinguendo:
> **(A)** i modelli *nostri* che coderai deve ospitare, **(B)** i modelli esterni
> (cloud / RunPod) che restano dove sono, **(C)** i modelli locali in-container.
> Aggiornato al 09/10/2026. Stringhe e default verificati nel codice (file:riga).

## Topologia dell'inferenza (per orientarsi)

- **coderai (zeiss, 192.168.42.3:8000, 3090 condivisa)** = la *nostra* inferenza
  dietro la *nostra* API OpenAI-compatibile. È qui che devono stare i modelli
  della sezione **A**.
- **coderai su prod (ru002848:8777)** = orchestratore **senza GPU** (CPU);
  oggi `models.json` è vuoto. Non serve modelli di chat/embedding.
- **Pod RunPod A40 on-demand** = oggi ospitano OCR-VLM, l'estrattore casi
  (`qwen38-awq`/`diogene-alpha`) e il trio di estrazione. Accesi a richiesta.
- **diogene-alpha** = il nostro modello condiviso (base + LoRA), servito da un pod
  vLLM; il controllo accensione/spegnimento è **alpha-ctl** su
  `127.0.0.1:8628` (NON è l'endpoint di inferenza, è il controllore).
- **Cloud** = OpenAI / Anthropic / Mistral (sezione B), con chiavi nostre.

---

## A. Modelli che coderai deve servire (inferenza nostra)

Nomi serviti (`--served-model-name`) che i client già si aspettano. Dove un
modello oggi gira su un pod RunPod, l'indirizzo d'elezione è coderai.

| nome servito | modello base / file | quant. | tipo | chi lo usa |
|---|---|---|---|---|
| `qwen38` | Qwen3.8-27B | AWQ | chat/RAG | Digesta compositore `coderai-qwen38`; verti/omnia `coderai-qwen38` |
| `qwen38-awq` | `cyankiwi/Qwen3.8-27B-AWQ-INT4` | AWQ int4 | chat/estrazione (JSON) | Digesta: classificazione L2 (`modello`), estrattore casi, opzione residenze; `casi_pod.py` lo serve con `--served-model-name qwen38-awq` |
| `diogene-alpha` | `cyankiwi/Qwen3.8-27B-AWQ-INT4` + adattatore QLoRA **`sib4`** | AWQ int4 + LoRA | chat/estrazione (il nostro «reader» addestrato) | verti/omnia (`diogene-alpha`, default del confronto admin); Digesta estrattore casi modo `alpha` |
| `lisa-32k` | **Gemma-4-26B-A4B-it** QAT q4_0 «uncensored/heretic», file `/AI/guffcache/gemma-4-26B-A4B-it-qat-q4_0-uncensored-heretic-Q4_0.gguf` + mmproj BF16 (multimodale, image_to_text); `parser: gemma`, contesto 32768 (alias `lisa` = stesso file a 178k) | GGUF Q4_0 | chat/RAG (vision) | Digesta compositore `coderai-lisa`; verti/omnia `coderai-lisa` |
| `Gemma-4-14B-A4B-Heretic.i1-IQ4_XS.gguf` | Gemma-4-14B-A4B «Heretic» | GGUF IQ4_XS | chat/RAG | Digesta compositore `coderai-gemma14-heretic`; verti/omnia idem |
| `bge-m3` | `BAAI/bge-m3` | fp16 | **embedding** (dense 1024 + sparse) | Digesta: indicizzazione+query del corpus (vedi nota embeddings) |
| `allenai/olmOCR-2-7B-1025-FP8` | olmOCR-2-7B | FP8 | OCR vision (ripiego) | Digesta: OCR di ripiego per le pagine lette male da Surya |
| `surya` (servizio OCR) | Surya | — | OCR primario | Digesta: prima passata OCR (il codice la chiama «surya su coderai») |

**Reranker** (oggi locali nel container Digesta; **passano a coderai-prod in CPU
locale**, non su pod GPU; reranking **off** di default):

| nome | modello | tipo | dove |
|---|---|---|---|
| profondo | `BAAI/bge-reranker-v2-m3` | cross-encoder | coderai-prod, CPU |
| veloce | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | cross-encoder | coderai-prod, CPU |

**Trio di estrazione (campagna alpha)** — il modello è passato dal chiamante a
`estrazione.estrai(modello=…)`, non è un default di config. I tre usati:

| modello | quant. | dove gira oggi |
|---|---|---|
| `gemma-4-31B` | AWQ 8-bit | pod A40 |
| `llama-3.3-70B` | AWQ | pod A40 |
| `Qwen3.8-27B` | AWQ int4 | pod A40 (= `qwen38-awq`) |

> Nota embeddings (richiesta esplicita): gli **embedding vanno su coderai**. Il
> modello reale è `BAAI/bge-m3`; il default nel codice di Digesta
> (`EMBED_MODEL = "text-embedding-3-small"`, `app/config.py:315`) è solo un
> segnaposto OpenAI. In pratica Digesta deve puntare `DIGESTA_LLM_BASE_URL` al
> `/v1` di coderai (oggi vale `https://api.openai.com/v1`) e usare `bge-m3`.
> Esiste anche un bge-m3 *locale* in-container (`FlagEmbedding.BGEM3FlagModel`,
> `app/embed_locale.py`) usato solo come ripiego per la query quando la scheda
> condivisa sfratta il modello remoto.

---

## B. Modelli cloud

**Unico cloud rimasto: Anthropic `claude-opus-5`.**

| provider | modello | dove / default | uso |
|---|---|---|---|
| Anthropic | `claude-opus-5` | `ANTHROPIC_CHAT_MODEL` `config.py:352`; `OCR_CLOUD_MODEL_ANTHROPIC` `config.py:104` | compositore `anthropic` (default chat di verti/omnia/sibilla); OCR cloud di ultima istanza; teacher dell'addestramento di `diogene-alpha` |

**Dismessi il 09/10/2026 (nessun impiego):**
- `gpt-5.6-luna` (OpenAI, `OPENAI_CHAT_MODEL`) — compositore `openai` /
  verti-omnia `openai` ora morti.
- `mistral-ocr-latest` (Mistral, `OCR_CLOUD_MODEL_MISTRAL`) — OCR cloud di
  ultima istanza resta solo Anthropic (surya → olmOCR → opus).
- `text-embedding-3-small` (OpenAI, `EMBED_MODEL` segnaposto) — sostituito da
  `bge-m3`; mettere `DIGESTA_EMBED_MODEL=bge-m3`.

Nota: `llama-3.3-70b` NON è più un endpoint RunPod *serverless* — è servito da
coderai-prod (sezione A), che unifica il pod del trio e l'ex-serverless.

---

## C. Modelli locali in-container (NON coderai)

- **Digesta — classificazione L2 CPU**: `tfidf1`, scikit-learn LogisticRegression
  su TF-IDF, file `DATA_DIR/classificazione/cpu-tfidf1.joblib`
  (`app/classificazione_l2.py`). Nessuna GPU.
- **Digesta — bge-m3 ripiego**: query embedding locale su CPU (vedi nota).

> I due **reranker** (`BAAI/bge-reranker-v2-m3`, `cross-encoder/mmarco-mMiniLMv2-
> L12-H384-v1`), prima qui in-container, **passano a coderai-prod servita in CPU
> locale** (coderai è una macchina CPU): sono nella sezione A.
- **nexus**: modello **proprio** di triage = regressione logistica a pesi
  (`app/coda.py`, pesi in DB `pesi_coda`); niente LLM/embedding, nessun endpoint.

---

## D. Chi usa cosa, per container

- **Digesta** (il nucleo): tutto quanto sopra — OCR (surya + olmOCR + cloud di
  ripiego), embeddings (`bge-m3`), reranker (bge), classificazione L2 (`tfidf1`
  + `qwen38-awq`), estrazione campi (trio), estrattore casi
  (`qwen38-awq`/`diogene-alpha`+`sib4`), compositore RAG (menu: `qwen38`,
  `lisa-32k`, `Gemma-4-14B…gguf`, `openai`, `anthropic`, `serverless`), reader e
  guardia-linguaggio (modello da env/admin, nessun default letterale).
- **verti / omnia / groupama** (stesso codice): modello **scelto dall'admin**,
  salvato nella tabella SQLite `impostazioni`. Catalogo: `anthropic`
  (`claude-opus-5`), `openai` (`gpt-5.6-luna`), `coderai-qwen38` (`qwen38`),
  `coderai-lisa` (`lisa-32k`), `coderai-gemma14-heretic`
  (`Gemma-4-14B-A4B-Heretic.i1-IQ4_XS.gguf`), `runpod` (`llama-3.3-70b`),
  `diogene-alpha`. Default chat = `anthropic`; confronto admin = `diogene-alpha`.
  L'URL dei profili `coderai-*` è il campo `coderai_url` (l'admin lo punta al
  `/v1` di coderai).
- **sibilla-face**: **nessun modello proprio** — delega al nucleo
  (`/api/cluster/ask`, `/api/cluster/compositori`); conserva solo la scelta di
  default del menu (`"anthropic"`). I modelli e le chiavi stanno in Digesta.
- **vitruvio-face**: **nessun passo LLM** ad oggi; usa solo Aristotele
  (pseudonimizzazione, cripto) e le API di Digesta. In futuro lo stesso menu di
  Sibilla.
- **veridoc-face**: **nessun modello** — delega tutta l'analisi al nucleo.
- **nexus**: solo il suo logreg locale (sezione C).

### Container senza alcun modello
`cautio` (fatturazione SdI), `caelum` (meteo Open-Meteo, «modello» = modello
numerico meteo, non ML), `netlex`, `topos`, `ae-console`, `fasti`,
`giustiziamap`, `ianua`, `aristotele` (cripto), `mcp-aggregator` (solo gateway;
espone il controllore alpha-ctl su 8628, non un `/v1` di chat).

---

## In breve, per coderai

Ospitati/orchestrati da coderai-prod (RunPod A40 on-demand): **`qwen38` /
`qwen38-awq`**, **`diogene-alpha`** (base + LoRA `sib4`), **`lisa` / `lisa-32k`**,
**`Gemma-4-14B-A4B-Heretic.i1-IQ4_XS.gguf`**, **`bge-m3`** (embedding),
**`allenai/olmOCR-2-7B-1025-FP8`**, il servizio **`surya`**, e il trio
(`gemma-4-31B`, **`llama-3.3-70b`** — che unifica pod-trio + ex-serverless).
Piano in `piano-inferenza-su-coderai-prod.md`.

I due **reranker** stanno anch'essi su coderai-prod ma in **CPU locale** (non pod
GPU). L'unico **cloud** è **`claude-opus-5`** (chat + teacher). Dismessi il 09/10:
`gpt-5.6-luna`, `mistral-ocr-latest`, `text-embedding-3-small` (sostituito da
`bge-m3`).
