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
| **Answer Verification** | Optional second-pass re-checks every answer at temperature 0, recomputing arithmetic and correcting mistakes. |
| **Mixed Difficulty** | Easy / Medium / Hard — single level or any combination in one exam. |

---

## Supported Providers & Models

| Provider | Models | Notes |
|----------|--------|-------|
| **OpenAI** | gpt-4o-mini, gpt-4o, gpt-3.5-turbo | Best for most use cases |
| **Gemini** | gemini-2.0-flash, gemini-2.5-flash, gemini-1.5-flash, gemini-1.5-pro | Fast & cost-effective |
| **Claude** | claude-haiku-4-5-20251001, claude-sonnet-4-6, claude-opus-4-7 | Strong reasoning |
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

After entering a key, click **Validate Key** to confirm it works before generating.

> **Groq** offers a free tier — ideal for students who want to experiment at no cost.

---

## Cost Optimisation

Several techniques are applied to keep API costs as low as possible:

- **Anthropic prompt caching** — The system prompt is cached server-side for 5 minutes using Anthropic's `cache_control: ephemeral` feature. Back-to-back generation batches and the answer-verification pass reuse the cached tokens at ~10 % of the normal rate, reducing costs by up to 90 % on those tokens.
- **Batched generation** — Questions are generated in batches of 20 per API call rather than one at a time, dramatically reducing per-question overhead and latency.
- **Conservative `max_tokens` caps** — Each provider has a token ceiling set just above what is needed, preventing runaway billing from unexpectedly long responses.
- **Free-tier support** — Groq's free tier is surfaced as a first-class option so students can use the app at zero cost.
- **Cheap-model defaults** — The fastest/cheapest model for each provider (gpt-4o-mini, gemini-2.0-flash, claude-haiku, llama-3.1-8b-instant) is listed first.

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
