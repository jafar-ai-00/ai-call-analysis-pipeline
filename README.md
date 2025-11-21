# 📞 AI Call Analysis Pipeline

AI-powered pipeline that ingests daily call recordings for a **single client**, runs multi-step LLM analysis on each call, and exposes the results in a **Streamlit dashboard**.

The pipeline currently works on `.wav` and `.mp3` files that you drop into a folder and is built in small, composable steps so you can easily extend it.

---

## ✨ Features

### Per-call processing

For each audio recording:

1. **Ingestion**
   - Automatically discovers `.wav` and `.mp3` files in `recordings/` (recursively).
   - Creates a `RecordingFile` object with path, size, and modified time.

2. **Transcription (OpenAI Whisper)**
   - Uses `whisper-1` via the OpenAI API.
   - Produces a text transcript + optional detected language.

3. **Structured Call Object (Pydantic)**
   - Wraps each call as a `CallAnalysis` object with:
     - `metadata` (`CallMetadata`): call_id, client_id, audio file path, timestamps & extra metadata.
     - `transcript`: full text transcript.
     - `sentiment`: sentiment & emotion analysis.
     - `intent_and_topics`: main intent + topics.
     - `call_quality`: quality & agent performance.
     - `compliance_and_risk`: compliance flags & risk level.
     - `outcome_and_followup`: resolution status & follow-up actions.
     - `custom_metrics`: reserved for future business KPIs.
     - `raw_llm_outputs`: raw JSON from LLM calls for debugging.

4. **Per-call analysis (LLM, via OpenAI + LangChain)**

Each analyzed call gets:

- **Sentiment & emotion** (`SentimentAnalysis`)
  - Overall sentiment (`positive | neutral | negative`)
  - Numeric score (-1.0 to +1.0)
  - Emotion tags
  - Sentiment timeline over segments.

- **Intent & topics** (`IntentTopicsAnalysis`)
  - Primary/secondary intents
  - Topics and key phrases
  - Confidence score and notes.

- **Call quality & agent performance** (`CallQualityAnalysis`)
  - Overall quality score (0–100)
  - Subscores (greeting, empathy, clarity, professionalism, script adherence)
  - Strengths & improvement suggestions.

- **Compliance & risk** (`ComplianceRiskAnalysis`)
  - Required phrases present / missing
  - Forbidden phrases detected
  - PII detections with masked values
  - Risk level (`low | medium | high | critical`).

- **Outcome & follow-up** (`OutcomeFollowupAnalysis`)
  - Resolution status (`resolved | partially_resolved | unresolved`)
  - Final outcome label (e.g. `appointment_booked`, `information_provided`, `no_clear_outcome`)
  - Follow-up actions (description, owner, due date)
  - Escalation flag & reason.

All data is stored as JSON under `data/calls/`, one file per call.

---

## 🧱 Tech Stack

- **Language:** Python
- **Package manager:** `uv`
- **LLM & ASR:** OpenAI (Whisper, GPT-4o)
- **Orchestration / prompting:** LangChain (`langchain-openai`)
- **Web UI:** Streamlit
- **Config:** YAML + environment variables
- **Data modeling:** Pydantic v2 models

---

## 📂 Project Structure

```text
call_analysis_pipeline/
├── app/
│   ├── __init__.py
│   ├── ingestion.py            # Discover audio recordings
│   ├── transcription.py        # Whisper transcription (OpenAI)
│   ├── storage.py              # Save CallAnalysis JSON files
│   ├── schemas.py              # Pydantic models (CallMetadata, CallAnalysis, etc.)
│   ├── analysis_sentiment.py   # Sentiment & emotion analysis
│   ├── analysis_intent_topics.py
│   ├── analysis_quality.py
│   ├── analysis_compliance.py
│   ├── analysis_outcome.py
│   ├── analysis_runner.py      # Helpers to run analyses over all calls
│   └── dashboard.py            # Streamlit dashboard
├── config/
│   └── config.example.yaml     # Example config (copy to config.yaml locally)
├── recordings/                 # Input .wav/.mp3 recordings (ignored in git)
├── data/
│   └── calls/                  # Per-call JSON outputs (ignored in git)
├── run_full_pipeline.py        # End-to-end pipeline runner
├── main.py                     # Ingest + transcribe + save CallAnalysis
├── pyproject.toml              # uv / project configuration
└── .gitignore
````

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone <your-repo-url> call_analysis_pipeline
cd call_analysis_pipeline
```

### 2. Install dependencies with `uv`

```bash
uv sync
```

This uses `pyproject.toml` to install all dependencies into a virtual environment.

### 3. Configure OpenAI

Set your OpenAI API key as an environment variable (for the current shell session):

```bash
export OPENAI_API_KEY="sk-..."
```

Create your config file:

```bash
cp config/config.example.yaml config/config.yaml
```

Adjust values inside `config/config.yaml` as needed.

---

## 🔁 Pipeline: From audio to analyzed calls

### 0. Drop audio recordings

Place your `.wav` or `.mp3` files into:

```text
recordings/
```

Example:

```bash
cp /path/to/your_call.wav recordings/your_call.wav
```

### 1. Run the full pipeline

```bash
uv run python run_full_pipeline.py
```

This will:

1. Discover all `.wav` and `.mp3` files under `recordings/`.
2. Transcribe each with OpenAI Whisper.
3. Save one JSON per call in `data/calls/` as a `CallAnalysis` structure.
4. Run all analysis stages over all calls in `data/calls/`:

   * Sentiment & emotion
   * Intent & topics
   * Call quality
   * Compliance & risk
   * Outcome & follow-up

The analysis runner functions are idempotent: if a call already has a given section (e.g. sentiment), it will be skipped.

---

## 📊 Dashboard

### Run Streamlit

From the project root:

```bash
uv run streamlit run app/dashboard.py
```

Open the URL shown (typically `http://localhost:8501`).

### Dashboard features

* **Summary**

  * Total calls
  * Average quality score
  * High/critical risk count
  * Sentiment distribution
  * Risk distribution

* **Filters (sidebar)**

  * Sentiment (positive / neutral / negative / unknown)
  * Risk level (low / medium / high / critical / unknown)
  * Minimum quality score

* **Calls list**

  * Each call is shown in an expandable card with:

    * Call ID, sentiment, risk
    * Primary intent
    * Quality score
    * Resolution status
    * Transcript preview
  * Deeper details in nested expanders:

    * Full transcript
    * Sentiment details
    * Intent & topics
    * Call quality breakdown
    * Compliance & risk details
    * Outcome & follow-up

---

## ✅ Current Status

Implemented end-to-end:

* [x] Ingestion of `.wav` and `.mp3` files
* [x] Transcription via OpenAI Whisper
* [x] Structured `CallAnalysis` Pydantic models
* [x] Sentiment & emotion analysis
* [x] Intent & topics analysis
* [x] Call quality scoring
* [x] Compliance & risk checks
* [x] Outcome & follow-up analysis
* [x] JSON storage per call under `data/calls/`
* [x] Streamlit dashboard to explore analyzed calls
* [x] Single `run_full_pipeline.py` to chain all stages

---

## 🛠️ Possible Next Steps

* Per-day **aggregated reports** under `reports/` (JSON/Markdown/PDF).
* Multi-client support (separate client IDs + filter in dashboard).
* Charts and visualizations for trends over time.
* Vector search over transcripts (ChromaDB) for semantic search across calls.

```
