"""
AI Exam Review — Streamlit application.

Modes:
  1. Generate Exam  — configure topics, difficulty, count; download student
                      exam + answer key + study guide.
  2. Practice Mode  — interactive question-by-question quiz with feedback.
  3. Concept Review — chat-style tutor for any ML/AI topic.
"""
from __future__ import annotations

import streamlit as st

from exam_generator import TOPICS, generate_exam, generate_concept_response
from exam_utils import (
    format_student_exam,
    format_answer_key,
    format_study_guide,
    calculate_score,
    safe_filename,
)
from llm_providers import PROVIDER_MODELS, validate_api_key

# ─── Constants ────────────────────────────────────────────────────────────────

PROVIDER_KEY_HINTS = {
    "OpenAI": ("sk-...", "https://platform.openai.com/api-keys"),
    "Gemini": ("AIza...", "https://aistudio.google.com/app/apikey"),
    "Claude": ("sk-ant-...", "https://console.anthropic.com/"),
    "Groq (Free)": ("gsk_...", "https://console.groq.com/keys"),
}

DIFFICULTY_OPTIONS = ["Easy", "Medium", "Hard"]

CONCEPT_STARTERS = [
    "Explain how attention mechanisms work in Transformers.",
    "What is the difference between LSTM and GRU?",
    "How does backpropagation compute gradients?",
    "What is the bias-variance tradeoff?",
    "Explain prompt injection attacks.",
    "How does K-Means clustering work?",
    "What makes CNNs good at image recognition?",
    "Explain RLHF and why it matters.",
]

# ─── Page setup ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Exam Review",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* Wider tabs */
.stTabs [data-baseweb="tab-list"] { gap: 2rem; }
.stTabs [data-baseweb="tab"] { font-size: 1rem; padding: 0.5rem 0; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #f0f2f6;
    border-radius: 8px;
    padding: 12px 16px;
}

/* Answer feedback colours */
.ans-correct { color: #2e7d32; font-weight: 600; }
.ans-wrong   { color: #c62828; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)


# ─── Session state ────────────────────────────────────────────────────────────

_DEFAULTS: dict = {
    # Shared
    "api_key_valid": False,
    # Every question stem shown this session (exam + practice) — used to
    # guarantee new generations never repeat earlier questions.
    "seen_question_stems": [],
    # Generate Exam tab
    "exam_questions": [],
    "exam_title": "",
    "generate_exam_requested": False,
    # Practice tab
    "practice_questions": [],
    "practice_idx": 0,
    "practice_answers": {},   # {question_id: chosen_letter}
    "practice_started": False,
    "practice_complete": False,
    # Concept Review tab
    "concept_history": [],    # [{"role": ..., "content": ...}]
    "concept_topic": "",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🎓 AI Exam Review")
    st.caption("For students & professors — powered by LLMs")
    st.divider()

    # Provider
    provider = st.selectbox("LLM Provider", list(PROVIDER_MODELS.keys()))
    model = st.selectbox("Model", PROVIDER_MODELS[provider])

    hint, url = PROVIDER_KEY_HINTS.get(provider, ("", ""))
    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder=hint,
        help=f"Obtain your key at: {url}",
    )

    if api_key:
        if st.button("✔ Validate Key", use_container_width=True):
            with st.spinner("Validating…"):
                ok, msg = validate_api_key(provider, model, api_key)
            if ok:
                st.session_state.api_key_valid = True
                st.success(msg)
            else:
                st.session_state.api_key_valid = False
                st.error(msg)
        if st.session_state.api_key_valid:
            st.success("Key valid", icon="✅")
    else:
        st.session_state.api_key_valid = False
        st.info(f"[Get an API key]({url})", icon="🔑")
        if provider == "Groq (Free)":
            st.caption("Groq offers a **free tier** — great for students!")

    st.divider()
    st.caption("**Topics covered:**")
    for main_topic in TOPICS:
        st.caption(f"• {main_topic}")


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _internal_provider(p: str) -> str:
    """Strip ' (Free)' suffix for llm_providers dispatch."""
    return p  # llm_providers accepts "Groq (Free)" directly


def _require_api_key() -> bool:
    if not api_key:
        st.warning("Enter your API key in the sidebar to continue.")
        return False
    return True


def _record_seen(questions: list[dict]) -> None:
    """Remember question stems so future generations don't repeat them.

    Capped to the most recent 200 so the avoid-list stays a manageable size.
    """
    seen = st.session_state.seen_question_stems
    seen.extend(q.get("question", "") for q in questions if q.get("question"))
    st.session_state.seen_question_stems = seen[-200:]


# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab_gen, tab_practice, tab_concept = st.tabs(
    ["📋  Generate Exam", "🎯  Practice Mode", "📚  Concept Review"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — GENERATE EXAM
# ══════════════════════════════════════════════════════════════════════════════

with tab_gen:
    st.header("Generate Exam")
    st.caption(
        "Select topics, set your parameters, and generate a full multiple-choice exam "
        "ready to download."
    )

    # ── Display existing exam FIRST ────────────────────────────────────────
    if st.session_state.exam_questions:
        qs = st.session_state.exam_questions
        title = st.session_state.exam_title
        fname = safe_filename(title)

        st.subheader(f"Preview — {title}")

        # Metrics row
        diff_counts: dict[str, int] = {}
        for q in qs:
            d = q.get("difficulty", "?")
            diff_counts[d] = diff_counts.get(d, 0) + 1

        metric_cols = st.columns(1 + len(diff_counts))
        metric_cols[0].metric("Total", len(qs))
        for idx, (d, c) in enumerate(sorted(diff_counts.items())):
            icon = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}.get(d, "⚪")
            metric_cols[idx + 1].metric(f"{icon} {d}", c)

        # Download row
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            st.download_button(
                "📥 Student Exam",
                data=format_student_exam(qs, title),
                file_name=f"{fname}_exam.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "📥 Answer Key",
                data=format_answer_key(qs, title),
                file_name=f"{fname}_answer_key.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with dl3:
            st.download_button(
                "📥 Study Guide",
                data=format_study_guide(qs, title),
                file_name=f"{fname}_study_guide.txt",
                mime="text/plain",
                use_container_width=True,
            )

        st.divider()

        # Paginated question preview
        PAGE_SIZE = 10
        total_pages = max(1, (len(qs) + PAGE_SIZE - 1) // PAGE_SIZE)
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
        page_qs = qs[(page - 1) * PAGE_SIZE : page * PAGE_SIZE]
        st.caption(f"Showing questions {(page-1)*PAGE_SIZE+1}–{(page-1)*PAGE_SIZE+len(page_qs)} of {len(qs)}")

        for q in page_qs:
            diff = q.get("difficulty", "")
            icon = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}.get(diff, "⚪")
            correct = q.get("correct_answer", "")

            with st.expander(
                f"**Q{q['id']}** {icon} *{q.get('topic', '')}* — {q['question'][:90]}…",
                expanded=False,
            ):
                st.markdown(f"**{q['question']}**")
                for letter in ("A", "B", "C", "D"):
                    opt = q.get("options", {}).get(letter, "")
                    if opt:
                        if letter == correct:
                            st.markdown(
                                f"<span class='ans-correct'>✅ {letter}) {opt}</span>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.write(f"{letter}) {opt}")
                if q.get("explanation"):
                    st.info(f"💡 {q['explanation']}")

        st.divider()

    # ── Configuration ──────────────────────────────────────────────────────
    st.subheader("Generate New Exam" if st.session_state.exam_questions else "Configure Exam")

    left, right = st.columns([3, 2], gap="large")

    # ── Left: topic selection ──────────────────────────────────────────────
    with left:
        st.subheader("Topics")

        selected_topics: list[str] = []

        for main_topic, subtopics in TOPICS.items():
            with st.expander(f"**{main_topic}**", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    if st.button(
                        "Select All",
                        key=f"selall_{main_topic}",
                        use_container_width=True,
                    ):
                        for i in range(len(subtopics)):
                            st.session_state[f"st_{main_topic}_{i}"] = True
                with c2:
                    if st.button(
                        "Clear",
                        key=f"clr_{main_topic}",
                        use_container_width=True,
                    ):
                        for i in range(len(subtopics)):
                            st.session_state[f"st_{main_topic}_{i}"] = False

                for i, sub in enumerate(subtopics):
                    key = f"st_{main_topic}_{i}"
                    if st.checkbox(sub, key=key, value=st.session_state.get(key, False)):
                        selected_topics.append(sub)

        st.subheader("Custom Topic")
        custom_topic = st.text_input(
            "Add a topic not listed above (optional)",
            max_chars=200,
            placeholder="e.g., Quantum Machine Learning, Time-Series Forecasting…",
        )
        if custom_topic.strip():
            selected_topics.append(custom_topic.strip())

    # ── Right: settings ────────────────────────────────────────────────────
    with right:
        st.subheader("Exam Settings")

        exam_title = st.text_input(
            "Exam Title",
            value="AI & Machine Learning Exam",
            max_chars=100,
        )

        num_questions = st.slider(
            "Number of Questions",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
        )

        difficulties = st.multiselect(
            "Difficulty Level(s)",
            DIFFICULTY_OPTIONS,
            default=["Medium"],
            help="Select multiple to generate a mixed exam.",
        )
        if not difficulties:
            difficulties = ["Medium"]
            st.caption("Defaulting to Medium.")

        verify_answers = st.checkbox(
            "Verify answers for accuracy",
            value=True,
            help="Runs a second pass that re-checks every answer and recomputes any "
            "math. More accurate, but slower and uses extra API calls.",
        )

    # ── Generate button — full-width below columns ─────────────────────────
    st.divider()

    summary_col, btn_col = st.columns([3, 1], gap="large")
    with summary_col:
        if not selected_topics:
            st.warning("Select at least one topic above.", icon="⚠")
        elif not api_key:
            st.warning("Enter your API key in the sidebar.", icon="🔑")
        else:
            st.markdown(
                f"**{len(selected_topics)}** topic(s) &nbsp;·&nbsp; "
                f"**{num_questions}** questions &nbsp;·&nbsp; "
                f"{' + '.join(difficulties)}"
            )
    with btn_col:
        generate_btn = st.button(
            "🚀  Generate Exam",
            type="primary",
            use_container_width=True,
        )

    # ── Generation ────────────────────────────────────────────────────────
    if generate_btn:
        if not selected_topics:
            st.error("Select at least one topic before generating.", icon="⚠")
        elif not api_key:
            st.error("Enter your API key in the sidebar.", icon="🔑")
        else:
            # Capture params immediately so a rerun mid-generation can't lose them
            st.session_state.generate_exam_requested = True
            _gen_topics = list(selected_topics)
            _gen_n = num_questions
            _gen_diff = list(difficulties)
            _gen_title = exam_title
            _gen_verify = verify_answers
            _gen_avoid = list(st.session_state.seen_question_stems)

            progress_bar = st.progress(0)
            status_area = st.empty()

            def _progress(done: int, total: int, msg: str) -> None:
                pct = int(done / total * 100) if total else 0
                progress_bar.progress(pct)
                status_area.caption(msg)

            try:
                with st.spinner(f"Generating {_gen_n} questions…"):
                    questions = generate_exam(
                        provider=provider,
                        model=model,
                        api_key=api_key,
                        topics=_gen_topics,
                        num_questions=_gen_n,
                        difficulties=_gen_diff,
                        progress_cb=_progress,
                        avoid_questions=_gen_avoid,
                        verify=_gen_verify,
                    )
                progress_bar.progress(100)
                status_area.empty()
                _record_seen(questions)
                st.session_state.exam_questions = questions
                st.session_state.exam_title = _gen_title
                st.session_state.generate_exam_requested = False
                st.success(f"✅ Generated {len(questions)} questions!")
                st.rerun()
            except Exception as exc:
                progress_bar.empty()
                status_area.empty()
                st.session_state.generate_exam_requested = False
                st.error(f"Generation failed: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PRACTICE MODE
# ══════════════════════════════════════════════════════════════════════════════

with tab_practice:
    st.header("Practice Mode")
    st.caption("Take an interactive quiz and get instant question-by-question feedback.")

    # ── Setup screen ──────────────────────────────────────────────────────
    if not st.session_state.practice_started:
        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            st.subheader("Topic")

            # Flat list for selectbox
            all_subs: list[str] = [
                f"{main} › {sub}"
                for main, subs in TOPICS.items()
                for sub in subs
            ]
            practice_topic_sel = st.selectbox("Select subtopic", all_subs)
            practice_custom = st.text_input(
                "Or enter a custom topic",
                max_chars=200,
                placeholder="e.g., Bayesian Networks, Graph Neural Networks…",
            )

        with col_b:
            st.subheader("Quiz Settings")
            practice_n = st.slider("Questions", min_value=3, max_value=20, value=10)
            practice_diff = st.select_slider(
                "Difficulty",
                options=DIFFICULTY_OPTIONS,
                value="Medium",
            )
            practice_verify = st.checkbox(
                "Verify answers for accuracy",
                value=True,
                key="practice_verify",
                help="Re-checks every answer and recomputes any math before the "
                "quiz starts. More accurate, but slower.",
            )

        if not api_key:
            st.warning("Enter your API key in the sidebar.", icon="🔑")

        if st.button(
            "🚀  Start Quiz",
            type="primary",
            disabled=not api_key,
        ):
            topic_str = (
                practice_custom.strip()
                if practice_custom.strip()
                else practice_topic_sel
            )
            with st.spinner(f"Preparing {practice_n}-question quiz on '{topic_str}'…"):
                try:
                    qs = generate_exam(
                        provider=provider,
                        model=model,
                        api_key=api_key,
                        topics=[topic_str],
                        num_questions=practice_n,
                        difficulties=[practice_diff],
                        avoid_questions=list(st.session_state.seen_question_stems),
                        verify=practice_verify,
                    )
                    _record_seen(qs)
                    st.session_state.practice_questions = qs
                    st.session_state.practice_idx = 0
                    st.session_state.practice_answers = {}
                    st.session_state.practice_started = True
                    st.session_state.practice_complete = False
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to generate quiz: {exc}")

    # ── Active quiz ───────────────────────────────────────────────────────
    elif not st.session_state.practice_complete:
        qs = st.session_state.practice_questions
        if not qs:
            st.error("No questions loaded. Please restart the quiz.")
            st.session_state.practice_started = False
            st.rerun()

        idx = st.session_state.practice_idx
        total = len(qs)
        q = qs[idx]
        qid = q["id"]
        answered = qid in st.session_state.practice_answers

        # Header / progress
        st.progress((idx) / total)
        diff = q.get("difficulty", "")
        diff_icon = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}.get(diff, "⚪")
        st.caption(
            f"Question **{idx + 1}** of **{total}** &nbsp;·&nbsp; "
            f"{q.get('topic', '')} &nbsp;·&nbsp; {diff_icon} {diff}"
        )
        st.subheader(q["question"])

        options = q.get("options", {})
        option_labels = [f"{k})  {v}" for k, v in options.items() if k in "ABCD"]

        if not answered:
            # ── Answering state ─────────────────────────────────────────
            choice = st.radio(
                "Choose your answer:",
                option_labels,
                index=None,
                key=f"radio_{idx}",
                label_visibility="collapsed",
            )
            submit = st.button(
                "Submit Answer",
                type="primary",
                disabled=(choice is None),
            )
            if submit and choice is not None:
                selected_letter = choice[0]  # "A", "B", "C", or "D"
                st.session_state.practice_answers[qid] = selected_letter
                st.rerun()

        else:
            # ── Review state ────────────────────────────────────────────
            user_ans = st.session_state.practice_answers[qid]
            correct = q.get("correct_answer", "")

            for letter, text in options.items():
                if letter not in "ABCD":
                    continue
                if letter == correct:
                    st.success(f"✅  **{letter})  {text}**  ← Correct answer")
                elif letter == user_ans and user_ans != correct:
                    st.error(f"❌  **{letter})  {text}**  ← Your answer")
                else:
                    st.write(f"&nbsp;&nbsp;&nbsp;{letter})  {text}")

            if user_ans == correct:
                st.success("🎉  Correct!")
            else:
                st.error(f"Incorrect. The right answer is **{correct}**.")

            if q.get("explanation"):
                st.info(f"**Explanation:** {q['explanation']}")

            # Navigation
            nav_prev, nav_next = st.columns(2)
            with nav_prev:
                if idx > 0:
                    if st.button("← Previous", use_container_width=True):
                        st.session_state.practice_idx -= 1
                        st.rerun()
            with nav_next:
                if idx < total - 1:
                    if st.button("Next →", type="primary", use_container_width=True):
                        st.session_state.practice_idx += 1
                        st.rerun()
                else:
                    if st.button(
                        "View Results 🏆", type="primary", use_container_width=True
                    ):
                        st.session_state.practice_complete = True
                        st.rerun()

        # Allow quitting early
        with st.expander("⚙  Quiz options"):
            if st.button("Quit and see results"):
                st.session_state.practice_complete = True
                st.rerun()
            if st.button("Start over"):
                st.session_state.practice_started = False
                st.session_state.practice_questions = []
                st.session_state.practice_answers = {}
                st.session_state.practice_idx = 0
                st.session_state.practice_complete = False
                st.rerun()

    # ── Results screen ────────────────────────────────────────────────────
    else:
        qs = st.session_state.practice_questions
        answers = st.session_state.practice_answers
        summary = calculate_score(qs, answers)

        pct = summary["percentage"]
        grade = (
            ("🌟  Excellent!", "success")
            if pct >= 80
            else ("👍  Good job!", "info")
            if pct >= 60
            else ("📚  Keep practicing!", "warning")
        )

        st.subheader("Quiz Results")

        c1, c2, c3 = st.columns(3)
        c1.metric("Score", f"{summary['correct']} / {summary['total']}")
        c2.metric("Percentage", f"{pct}%")
        c3.metric("Result", grade[0])

        # By difficulty
        if summary["by_difficulty"]:
            st.subheader("Performance by Difficulty")
            for d, stats in sorted(summary["by_difficulty"].items()):
                if stats["total"] > 0:
                    d_pct = round(stats["correct"] / stats["total"] * 100, 1)
                    icon = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}.get(d, "⚪")
                    st.write(
                        f"{icon} **{d}**: {stats['correct']}/{stats['total']} ({d_pct}%)"
                    )
                    st.progress(d_pct / 100)

        # Wrong answers review
        if summary["wrong_questions"]:
            st.subheader(f"Review — {len(summary['wrong_questions'])} Incorrect Answers")
            for wq in summary["wrong_questions"]:
                with st.expander(f"Q{wq['id']}: {wq['question'][:80]}…"):
                    user_a = answers.get(wq["id"], "—")
                    correct_a = wq.get("correct_answer", "")
                    opts = wq.get("options", {})
                    for letter, text in opts.items():
                        if letter == correct_a:
                            st.success(f"✅ {letter})  {text}")
                        elif letter == user_a:
                            st.error(f"❌ {letter})  {text}  (your answer)")
                        else:
                            st.write(f"&nbsp;&nbsp;{letter})  {text}")
                    if wq.get("explanation"):
                        st.info(wq["explanation"])

        st.divider()
        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button("Retry Same Quiz", use_container_width=True):
                st.session_state.practice_answers = {}
                st.session_state.practice_idx = 0
                st.session_state.practice_complete = False
                st.rerun()
        with btn2:
            if st.button("New Quiz", type="primary", use_container_width=True):
                for _k in (
                    "practice_questions",
                    "practice_answers",
                ):
                    st.session_state[_k] = [] if _k.endswith("questions") else {}
                st.session_state.practice_idx = 0
                st.session_state.practice_started = False
                st.session_state.practice_complete = False
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CONCEPT REVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab_concept:
    st.header("Concept Review")
    st.caption(
        "Chat with an AI tutor about any ML/AI topic. "
        "Ask for explanations, examples, comparisons, or deep-dives."
    )

    col_l, col_r = st.columns([1, 3], gap="large")

    with col_l:
        st.subheader("Topic")
        main_topics = list(TOPICS.keys()) + ["Other / Custom"]
        chosen_main = st.selectbox("Main area", main_topics, key="cr_main")

        if chosen_main != "Other / Custom":
            chosen_sub = st.selectbox(
                "Subtopic", TOPICS[chosen_main], key="cr_sub"
            )
            concept_topic = chosen_sub
        else:
            concept_topic = st.text_input(
                "Enter topic",
                max_chars=200,
                placeholder="e.g., Reinforcement Learning, Kalman Filters…",
                key="cr_custom",
            )

        st.session_state.concept_topic = concept_topic or "Machine Learning"

        st.divider()
        if st.button("🗑  Clear Chat", use_container_width=True):
            st.session_state.concept_history = []
            st.rerun()

        st.divider()
        st.caption("**Quick starters:**")
        for starter in CONCEPT_STARTERS[:4]:
            if st.button(
                starter[:45] + ("…" if len(starter) > 45 else ""),
                key=f"starter_{starter[:20]}",
                use_container_width=True,
            ):
                if api_key:
                    st.session_state.concept_history.append(
                        {"role": "user", "content": starter}
                    )
                    with st.spinner("Thinking…"):
                        try:
                            reply = generate_concept_response(
                                provider=provider,
                                model=model,
                                api_key=api_key,
                                topic=st.session_state.concept_topic,
                                history=st.session_state.concept_history[:-1],
                                user_question=starter,
                            )
                            st.session_state.concept_history.append(
                                {"role": "assistant", "content": reply}
                            )
                        except Exception as exc:
                            st.session_state.concept_history.pop()
                            st.error(str(exc))
                    st.rerun()
                else:
                    st.warning("Enter your API key first.")

    with col_r:
        # Display conversation history
        history = st.session_state.concept_history
        if not history:
            st.info(
                "👋 **Ask anything about " + (concept_topic or "ML/AI") + ".**\n\n"
                "Examples:\n"
                "- *Explain how self-attention works with an example.*\n"
                "- *What is the vanishing gradient problem and how does LSTM solve it?*\n"
                "- *Compare RAG vs fine-tuning for domain adaptation.*\n"
                "- *What security risks exist with LLMs in production?*"
            )
        else:
            for msg in history:
                with st.chat_message(
                    "user" if msg["role"] == "user" else "assistant"
                ):
                    st.markdown(msg["content"])

        # Input
        if not api_key:
            st.warning("Enter your API key in the sidebar to chat.", icon="🔑")
        else:
            user_input = st.chat_input(
                f"Ask about {concept_topic or 'any AI/ML topic'}…"
            )
            if user_input:
                st.session_state.concept_history.append(
                    {"role": "user", "content": user_input}
                )
                with st.spinner("Thinking…"):
                    try:
                        reply = generate_concept_response(
                            provider=provider,
                            model=model,
                            api_key=api_key,
                            topic=st.session_state.concept_topic,
                            history=st.session_state.concept_history[:-1],
                            user_question=user_input,
                        )
                        st.session_state.concept_history.append(
                            {"role": "assistant", "content": reply}
                        )
                    except Exception as exc:
                        st.session_state.concept_history.pop()
                        st.error(f"Error: {exc}")
                st.rerun()
