"""
Exam generation logic — prompt construction, batched API calls, JSON parsing.
"""
from __future__ import annotations
import json
import random
import re
import time
from typing import Callable

from llm_providers import call_llm

# ─── Topic definitions ────────────────────────────────────────────────────────

TOPICS: dict[str, list[str]] = {
    "Machine Learning": [
        "Supervised Learning — Regression (Linear, Polynomial, Ridge, Lasso)",
        "Supervised Learning — Classification (Logistic Regression, Naive Bayes, KNN)",
        "Support Vector Machines (SVM, kernels, margin maximization)",
        "Decision Trees, Random Forests & Ensemble Methods",
        "Unsupervised Learning — Clustering (K-Means, DBSCAN, Hierarchical)",
        "Unsupervised Learning — Dimensionality Reduction (PCA, t-SNE, UMAP)",
        "Boosting Algorithms (AdaBoost, Gradient Boosting, XGBoost, LightGBM)",
        "Model Evaluation, Cross-Validation & Hyperparameter Tuning",
        "Feature Engineering & Selection",
        "Bias-Variance Tradeoff & Regularization (L1/L2)",
    ],
    "Deep Learning": [
        "Artificial Neural Networks & Backpropagation",
        "Deep Neural Networks (DNN) — Architecture & Activation Functions",
        "Convolutional Neural Networks (CNN) — Filters, Pooling, Architectures",
        "Recurrent Neural Networks (RNN) — Vanishing Gradient, BPTT",
        "Long Short-Term Memory (LSTM) & Gated Recurrent Units (GRU)",
        "Autoencoders & Variational Autoencoders (VAE)",
        "Generative Adversarial Networks (GAN)",
        "Optimization — SGD, Adam, Learning Rate Scheduling",
        "Regularization — Dropout, Batch Normalization, Weight Decay",
        "Transfer Learning & Fine-tuning",
    ],
    "LLMs & Generative AI": [
        "Transformer Architecture — Self-Attention, Multi-Head Attention",
        "Positional Encoding & Tokenization",
        "BERT, GPT, T5 and Foundational Model Families",
        "Pre-training Objectives (MLM, CLM, Seq2Seq)",
        "Fine-tuning, RLHF & Constitutional AI",
        "Prompt Engineering — Zero-shot, Few-shot, Chain-of-Thought",
        "Retrieval-Augmented Generation (RAG) & Vector Databases",
        "Hallucinations, Alignment & Model Evaluation",
        "AI Security — Jailbreaking, Prompt Injection, Adversarial Attacks",
        "Multimodal Models & AI Agents",
    ],
}

BATCH_SIZE = 20  # Questions per API call (stays well within context limits)

# ─── Public API ───────────────────────────────────────────────────────────────

Question = dict  # Type alias for a question dict


def generate_exam(
    provider: str,
    model: str,
    api_key: str,
    topics: list[str],
    num_questions: int,
    difficulties: list[str],
    progress_cb: Callable[[int, int, str], None] | None = None,
    groq_delay: float = 2.0,
    avoid_questions: list[str] | None = None,
    verify: bool = True,
) -> list[Question]:
    """
    Generate `num_questions` MCQ questions covering `topics` at the specified
    `difficulties`.  Large exams are split into batches of BATCH_SIZE.

    Questions are deduplicated both within this call and against `avoid_questions`
    (e.g. questions already shown elsewhere in the session), so practice and exam
    sets do not repeat.  When `verify` is set, a deterministic second pass
    re-checks every answer — recomputing arithmetic — and corrects mistakes.

    Args:
        provider: LLM provider name.
        model: Model identifier.
        api_key: Provider API key.
        topics: List of topic/subtopic strings.
        num_questions: Total questions to generate (1–100).
        difficulties: Non-empty list of "Easy", "Medium", "Hard".
        progress_cb: Optional callback(batch_done, total_batches, status_msg).
        groq_delay: Seconds to sleep between Groq batches (free-tier rate limit).
        avoid_questions: Question stems already used elsewhere — never repeat them.
        verify: Run an accuracy-correction pass over the generated questions.

    Returns:
        List of question dicts.
    """
    topics_str = "\n".join(f"  - {t}" for t in topics)
    diff_instruction = _build_difficulty_instruction(difficulties)
    json_mode = provider in ("OpenAI", "Groq", "Groq (Free)")

    all_questions: list[Question] = []
    # Stems we must not repeat: previously-seen (passed in) + everything we add.
    seen_stems: set[str] = {_stem(s) for s in (avoid_questions or []) if s}
    # Full question texts to show the model as "already asked, do not repeat".
    avoid_texts: list[str] = list(avoid_questions or [])

    total_batches = (num_questions + BATCH_SIZE - 1) // BATCH_SIZE
    # Allow a few extra rounds to top up questions dropped as duplicates.
    max_attempts = total_batches + 3
    attempt = 0
    sleep_pending = False

    while len(all_questions) < num_questions and attempt < max_attempts:
        remaining = num_questions - len(all_questions)
        # Over-request a little so duplicates dropped during dedup don't leave us
        # short — extra questions are simply discarded once the target is met.
        batch_n = min(BATCH_SIZE, remaining + 5)

        if progress_cb:
            upper = min(num_questions, len(all_questions) + batch_n)
            progress_cb(
                attempt,
                total_batches,
                f"Generating questions {len(all_questions) + 1}–"
                f"{upper} of {num_questions}…",
            )

        if sleep_pending:
            time.sleep(groq_delay)
            sleep_pending = False

        messages = _build_messages(
            topics_str,
            batch_n,
            diff_instruction,
            len(all_questions) + 1,
            avoid_texts=avoid_texts,
            variation_seed=attempt,
        )
        # Higher temperature → genuinely different questions across batches/runs.
        raw = call_llm(
            provider, model, api_key, messages, json_mode=json_mode, temperature=0.9
        )
        batch = _parse_response(raw)

        for q in batch:
            stem = _stem(q.get("question", ""))
            if not stem or stem in seen_stems:
                continue  # blank or duplicate — skip it
            seen_stems.add(stem)
            avoid_texts.append(q.get("question", ""))
            q["id"] = len(all_questions) + 1
            all_questions.append(q)
            if len(all_questions) >= num_questions:
                break

        attempt += 1
        if provider in ("Groq", "Groq (Free)"):
            sleep_pending = True

    if verify and all_questions:
        all_questions = _verify_questions(
            provider, model, api_key, all_questions, json_mode, groq_delay, progress_cb
        )

    if progress_cb:
        progress_cb(total_batches, total_batches, "Done!")

    return all_questions[:num_questions]


def generate_concept_response(
    provider: str,
    model: str,
    api_key: str,
    topic: str,
    history: list[dict],
    user_question: str,
) -> str:
    """
    Return a tutor-style explanation for `user_question` in the context of `topic`.

    Args:
        history: Previous messages in [{"role": ..., "content": ...}] format
                 (excluding the latest user question).
        user_question: The student's latest question.
    """
    system = (
        f"You are an expert AI/ML tutor specializing in {topic}. "
        "Provide clear, accurate, and educational explanations. "
        "Use concrete examples, analogies, and code snippets where helpful. "
        "Format your response with Markdown headings, bullet points, and code blocks."
    )
    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_question})
    return call_llm(provider, model, api_key, messages)


# ─── Prompt construction ──────────────────────────────────────────────────────

def _build_difficulty_instruction(difficulties: list[str]) -> str:
    if len(difficulties) == 1:
        d = difficulties[0]
        desc = {
            "Easy": "basic definitions, terminology, and recall of fundamental facts",
            "Medium": "conceptual understanding, application, and comparisons",
            "Hard": "deep analysis, edge cases, implementation nuances, and multi-step reasoning",
        }.get(d, d)
        return f"ALL questions must be {d} difficulty: {desc}."

    parts = []
    for d in difficulties:
        desc = {
            "Easy": "recall/definitions",
            "Medium": "understanding/application",
            "Hard": "analysis/edge-cases",
        }.get(d, d)
        parts.append(f"{d} ({desc})")
    return (
        f"Distribute difficulty EVENLY across: {', '.join(parts)}. "
        "Mix them throughout the list — do not group by difficulty."
    )


# Rotating angles that push each batch toward different question styles so a
# topic gets comprehensive coverage instead of the same canonical questions.
_VARIATION_ANGLES: list[str] = [
    "definitions and core terminology",
    "comparisons and trade-offs between related techniques",
    "applied scenarios — 'which method/approach fits this situation'",
    "interpreting formulas, metrics, or numerical results",
    "common pitfalls, misconceptions, and failure modes",
    "step-by-step reasoning about how a method behaves",
]


def _build_messages(
    topics_str: str,
    batch_n: int,
    diff_instruction: str,
    start_id: int,
    avoid_texts: list[str] | None = None,
    variation_seed: int = 0,
) -> list[dict]:
    system = (
        "You are an expert educator and exam writer specializing in computer science, "
        "machine learning, and artificial intelligence. "
        "You create rigorous, unambiguous, high-quality multiple-choice questions. "
        "Every question must be DISTINCT — never paraphrase or re-use a concept you "
        "have already been asked to avoid. "
        "You ALWAYS respond with valid JSON only — no markdown fences, no extra text."
    )

    example = json.dumps(
        {
            "id": start_id,
            "topic": "Supervised Learning",
            "difficulty": "Medium",
            "question": "Which of the following best describes the bias-variance tradeoff?",
            "options": {
                "A": "A model with high bias underfits and high variance overfits the training data.",
                "B": "Bias and variance always move in the same direction as model complexity increases.",
                "C": "Regularization increases both bias and variance simultaneously.",
                "D": "A high-variance model always generalizes better to unseen data.",
            },
            "correct_answer": "A",
            "explanation": (
                "High bias → the model is too simple and underfits (misses signal). "
                "High variance → the model is too complex and overfits (fits noise). "
                "The tradeoff is about finding the complexity sweet-spot."
            ),
        },
        indent=2,
    )

    angle = _VARIATION_ANGLES[variation_seed % len(_VARIATION_ANGLES)]
    avoid_block = _build_avoid_block(avoid_texts)

    user = f"""Generate exactly {batch_n} multiple-choice exam questions.

TOPICS (cover comprehensively and proportionally):
{topics_str}

DIFFICULTY: {diff_instruction}

COVERAGE & VARIETY:
- Span the full breadth of each topic — its sub-concepts, methods, and edge cases.
- For THIS batch, emphasize questions about: {angle}.
- Do NOT cluster around the single most obvious fact of a topic; each question must
  test a genuinely different idea.
{avoid_block}
REQUIREMENTS:
- Each question has exactly 4 options: A, B, C, D
- Exactly one option is correct; the others are plausible but clearly wrong
- Questions must be academically rigorous, self-contained, and unambiguous
- IDs run from {start_id} to {start_id + batch_n - 1}
- Vary which letter (A/B/C/D) holds the correct answer — do NOT default to any letter

MATHEMATICAL ACCURACY (compute carefully, then re-check before output):
- Work through every calculation step by step and confirm the marked answer is the
  result of that calculation — not an approximation or a distractor.
- CNN output spatial size: floor((input + 2·padding − filter_size) / stride) + 1
  Example — 32×32 input, 3×3 filter, stride 1, no padding: (32 − 3)/1 + 1 = 30
- Verify each numerical value appearing in the question stem AND in all four options.
- Only include a math question if you are certain the arithmetic is correct.

OUTPUT FORMAT — return ONLY a valid JSON array:
[
{example},
  ... (remaining {batch_n - 1} questions)
]

No markdown, no explanation, no text outside the JSON array."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _build_avoid_block(avoid_texts: list[str] | None) -> str:
    """Render a 'do not repeat these' section from recently-used question stems."""
    if not avoid_texts:
        return ""
    # Only the most recent stems matter and keep the prompt compact.
    recent = avoid_texts[-40:]
    listed = "\n".join(f"  - {t[:140]}" for t in recent)
    return (
        "\nALREADY ASKED — do NOT repeat, rephrase, or test the same idea as any of "
        f"these {len(recent)} questions:\n{listed}\n"
    )


# ─── Response parsing ─────────────────────────────────────────────────────────

def _parse_response(raw: str) -> list[Question]:
    """
    Extract a list of question dicts from the LLM's raw text response.
    Handles: bare JSON array, wrapped {"questions": [...]}, markdown fences.
    """
    text = _strip_fences(raw.strip())

    # Attempt 1: direct parse
    try:
        data = json.loads(text)
        return _extract_list(data)
    except json.JSONDecodeError:
        pass

    # Attempt 2: find first [...] in text
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            data = json.loads(match.group())
            return _extract_list(data)
        except json.JSONDecodeError:
            pass

    # Attempt 3: find first {...} (wrapped object)
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            data = json.loads(match.group())
            return _extract_list(data)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Could not parse questions from the model's response. "
        "Try again — the model may have returned malformed JSON."
    )


def _strip_fences(text: str) -> str:
    """Remove markdown code fences that some models add despite instructions."""
    if text.startswith("```"):
        lines = text.split("\n")
        # Drop first line (``` or ```json) and last ``` line
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner)
    return text.strip()


def _extract_list(data) -> list[Question]:
    """Accept either a bare list or a dict with a 'questions' key."""
    if isinstance(data, list):
        return [_normalize(q) for q in data]
    if isinstance(data, dict):
        for key in ("questions", "exam", "items", "data"):
            if key in data and isinstance(data[key], list):
                return [_normalize(q) for q in data[key]]
    raise ValueError("JSON structure is not a question list.")


def _normalize(q: dict) -> dict:
    """Ensure required fields exist and randomly shuffle option positions."""
    letters = ["A", "B", "C", "D"]
    raw_opts = {
        k: str(v)
        for k, v in q.get("options", {}).items()
        if k in letters
    }
    original_correct = str(q.get("correct_answer", "A")).upper()

    # Shuffle option positions to eliminate answer-position bias (e.g., always B)
    indices = list(range(len(letters)))
    random.shuffle(indices)

    shuffled_values = [raw_opts.get(letters[i], "") for i in indices]
    shuffled_opts = {letters[pos]: shuffled_values[pos] for pos in range(len(letters))}

    # Determine which new letter now holds the originally correct answer
    original_idx = letters.index(original_correct) if original_correct in letters else 0
    new_correct = letters[indices.index(original_idx)]

    return {
        "id": int(q.get("id", 0)),
        "topic": str(q.get("topic", "General")),
        "difficulty": str(q.get("difficulty", "Medium")).capitalize(),
        "question": str(q.get("question", "")),
        "options": shuffled_opts,
        "correct_answer": new_correct,
        "explanation": str(q.get("explanation", "")),
    }


# ─── Deduplication ────────────────────────────────────────────────────────────

def _stem(text: str) -> str:
    """
    Normalize a question into a comparison key for near-duplicate detection:
    lowercase, strip punctuation, collapse whitespace, keep the leading content.
    """
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text[:120]


# ─── Answer verification ──────────────────────────────────────────────────────

def _verify_questions(
    provider: str,
    model: str,
    api_key: str,
    questions: list[Question],
    json_mode: bool,
    groq_delay: float,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> list[Question]:
    """
    Re-check every question with a deterministic (temperature 0) pass and correct
    any mistakes — wrong `correct_answer`, miscomputed numbers, or explanations
    that contradict the marked answer.  Questions that cannot be verified are
    returned unchanged so generation never silently loses content.
    """
    verified: list[Question] = []
    total_batches = (len(questions) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        chunk = questions[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
        if progress_cb:
            progress_cb(
                batch_idx,
                total_batches,
                f"Verifying answers (batch {batch_idx + 1}/{total_batches})…",
            )
        try:
            messages = _build_verify_messages(chunk)
            raw = call_llm(
                provider, model, api_key, messages,
                json_mode=json_mode, temperature=0.0,
            )
            corrected = _parse_response(raw)
            merged = _merge_corrections(chunk, corrected)
            verified.extend(merged)
        except Exception:
            # Verification is best-effort: on any failure keep the originals.
            verified.extend(chunk)

        if provider in ("Groq", "Groq (Free)") and batch_idx < total_batches - 1:
            time.sleep(groq_delay)

    return verified


def _build_verify_messages(questions: list[Question]) -> list[dict]:
    system = (
        "You are a meticulous exam answer-key checker for machine learning and AI. "
        "For each question you recompute any arithmetic from scratch, confirm exactly "
        "one option is correct, and fix the answer key when it is wrong. "
        "You ALWAYS respond with valid JSON only — no markdown fences, no extra text."
    )

    payload = json.dumps(
        [
            {
                "id": q["id"],
                "question": q["question"],
                "options": q["options"],
                "correct_answer": q["correct_answer"],
                "explanation": q.get("explanation", ""),
            }
            for q in questions
        ],
        indent=2,
    )

    user = f"""Review the following multiple-choice questions for correctness.

For EACH question:
1. Re-derive the answer independently. For any math, compute it step by step.
2. Decide which single option (A/B/C/D) is actually correct given the option TEXT.
3. If the option texts contain a wrong number (e.g. a miscomputed result), correct
   that option's text so the correct answer is accurate.
4. Set "correct_answer" to the letter of the truly correct option.
5. Make the explanation briefly justify the correct answer (include the computation
   for math questions).
6. Keep the same "id" and keep all four options A–D present.

Return ONLY a JSON array with the corrected questions, each as:
{{"id": <int>, "question": <str>, "options": {{"A":..,"B":..,"C":..,"D":..}},
  "correct_answer": <letter>, "explanation": <str>}}

QUESTIONS TO REVIEW:
{payload}

No markdown, no commentary — only the JSON array."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _merge_corrections(
    originals: list[Question], corrected: list[Question]
) -> list[Question]:
    """
    Apply verifier output back onto the original questions by id. Only the answer,
    explanation, and (when provided) option text are updated; topic/difficulty/id
    are preserved. Originals without a valid correction are kept as-is.
    """
    by_id: dict[int, dict] = {}
    for c in corrected:
        try:
            by_id[int(c.get("id", -1))] = c
        except (TypeError, ValueError):
            continue

    letters = ("A", "B", "C", "D")
    result: list[Question] = []
    for q in originals:
        c = by_id.get(q["id"])
        if not c:
            result.append(q)
            continue

        new_correct = str(c.get("correct_answer", q["correct_answer"])).upper()
        if new_correct not in letters:
            new_correct = q["correct_answer"]

        # Accept corrected option text only if all four letters are present.
        c_opts = c.get("options", {})
        if isinstance(c_opts, dict) and all(
            letter in c_opts and str(c_opts[letter]).strip() for letter in letters
        ):
            options = {letter: str(c_opts[letter]) for letter in letters}
        else:
            options = q["options"]

        merged = dict(q)
        merged["options"] = options
        merged["correct_answer"] = new_correct
        merged["explanation"] = str(c.get("explanation", q.get("explanation", "")))
        result.append(merged)

    return result
