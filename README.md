# AI Exam Review

An interactive Streamlit application for ML/AI students and professors to generate exams, practice with quizzes, and review concepts — all powered by large language models.

---

## Features

| Feature | Description |
|---------|-------------|
| **Generate Exam** | Create multiple-choice exams (5–100 questions) across 30+ ML/AI subtopics with one click. Download as Student Exam, Answer Key, or Study Guide. |
| **Practice Mode** | Interactive question-by-question quiz with instant feedback, score summary, and wrong-answer review. |
| **Concept Review** | Chat with an AI tutor on any ML/AI topic with full conversation history. |
| **Deduplication** | Questions never repeat within a session (exam ↔ practice). |
| **Answer Verification** | Optional second-pass re-checks every answer at temperature 0 where supported, recomputing arithmetic and correcting mistakes. |
| **Mixed Difficulty** | Easy / Medium / Hard — single level or any combination in one exam. |

---

## Supported Providers & Models

| Provider | Models | Notes |
|----------|--------|-------|
| **OpenAI** | gpt-4.1-mini, gpt-4.1, gpt-5-mini, gpt-5.2, gpt-4o-mini, gpt-4o | Best for most use cases |
| **Gemini** | gemini-3.5-flash-lite, gemini-3.8-flash, gemini-3.1-pro-preview | Fast & cost-effective |
| **Claude** | claude-haiku-4-5-20251001, claude-sonnet-5, claude-opus-5-5, claude-sonnet-4-6, claude-opus-4-7 | Strong reasoning |
| **Groq (Free)** | llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768, gemma2-9b-it | Free tier, great for students |

---

## Installation

```bash
git clone https://github.com/snerur/exam_review.git
cd exam_review
pip install -r requirements.txt
```

Python 3.10+ is required.

---

## Running the App

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

---

## API Key Setup

API keys are entered directly in the sidebar — **never hardcoded or stored in files**.

| Provider | Where to get a key | Format |
|----------|--------------------|--------|
| OpenAI | https://platform.openai.com/api-keys | `sk-...` |
| Gemini | https://aistudio.google.com/app/apikey | `AIza...` |
| Claude | https://console.anthropic.com/ | `sk-ant-api03-...` |
| Groq | https://console.groq.com/keys | `gsk_...` |

After entering a key, click **Validate Connection** to test the selected model before generating. Changing the provider, model, or key clears the validation status. Access still depends on your account permissions, quota, and model availability.

> **Groq** offers a free tier — ideal for students who want to experiment at no cost.

---

## Cost Optimisation

Several techniques are applied to keep API costs as low as possible:

- **Anthropic prompt caching** — The system prompt is cached server-side for 5 minutes using Anthropic's `cache_control: ephemeral` feature. Back-to-back generation batches and the answer-verification pass reuse the cached tokens at ~10 % of the normal rate, reducing costs by up to 90 % on those tokens.
- **Batched generation** — Questions are generated in batches of 20 per API call rather than one at a time, dramatically reducing per-question overhead and latency.
- **Conservative `max_tokens` caps** — Each provider has a token ceiling set just above what is needed, preventing runaway billing from unexpectedly long responses.
- **Free-tier support** — Groq's free tier is surfaced as a first-class option so students can use the app at zero cost.
- **Cheap-model defaults** — The fastest/cheapest model for each provider (gpt-4.1-mini, gemini-3.5-flash-lite, claude-haiku, llama-3.3-70b-versatile) is listed first.

---

## Project Structure

```
exam_review/
├── app.py               # Streamlit UI and session state
├── llm_providers.py     # Provider dispatch: OpenAI, Gemini, Claude, Groq
├── exam_generator.py    # Batch question generation, deduplication, verification
├── exam_utils.py        # Text formatting (exam / answer key / study guide) and scoring
└── requirements.txt     # Python dependencies
```

---

## Topics Covered

**Machine Learning**
Regression, Classification, SVM, Decision Trees & Ensembles, Clustering, Dimensionality Reduction, Boosting (XGBoost/LightGBM), Model Evaluation, Feature Engineering, Bias-Variance & Regularisation

**Deep Learning**
ANNs & Backpropagation, DNNs, CNNs, RNNs, LSTM & GRU, Autoencoders & VAE, GANs, Optimisation (SGD/Adam), Regularisation (Dropout/BatchNorm), Transfer Learning

**LLMs & Generative AI**
Transformer Architecture, Positional Encoding & Tokenisation, BERT/GPT/T5, Pre-training Objectives, Fine-tuning & RLHF, Prompt Engineering, RAG & Vector Databases, Hallucinations & Alignment, AI Security, Multimodal Models & Agents

---

## Security & Privacy

- API keys are entered via a password-masked input field and never written to disk or logs.
- No user data, questions, or API keys are transmitted anywhere other than the selected LLM provider.
- The `.gitignore` excludes `.env`, `*.env`, and `__pycache__` to prevent accidental credential commits.

## Model compatibility

Sampling overrides are sent only to known compatible models. OpenAI reasoning models,
newer Claude models (including Opus 4.7), and Gemini 3 use their default sampling settings.
Gemini uses the Developer API, preserves conversation roles, and requests JSON for exams.
OpenAI uses `max_completion_tokens`, including room for reasoning tokens.

Model IDs and settings were checked against the official [OpenAI guidance](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.2),
[Gemini model catalog](https://ai.google.dev/gemini-api/docs/models), and
[Anthropic model catalog](https://platform.claude.com/docs/en/models/overview).
Retired Gemini 1.5/2.0 models are no longer offered.

Run offline regression checks (no API keys or paid requests):

```bash
python -m unittest discover -s tests -v
```
