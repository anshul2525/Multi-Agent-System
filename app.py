"""
Streamlit UI for the Multi-Agent Research System.
Supports ⚡ Standard Mode and 🔬 Deep Mode with smart token-bucket cooldown guard.
"""

import contextlib
import io
import os
import time
import traceback
import streamlit as st

# --------------------------------------------------------------------------
# Inject Streamlit Secrets into OS Environment BEFORE importing pipeline
# --------------------------------------------------------------------------
_secrets_error = None
_secrets_seen = []
try:
    for key, value in st.secrets.items():
        _secrets_seen.append(key)
        if isinstance(value, str):
            os.environ[key] = value
except Exception as e:
    _secrets_error = str(e)

if _secrets_error:
    st.error(f"Could not read st.secrets: {_secrets_error}")
    st.stop()

if "GROQ_API_KEY" not in os.environ:
    st.error(
        "GROQ_API_KEY is missing from Streamlit secrets. "
        f"Keys currently visible: {_secrets_seen or 'none'}. "
        "Go to Manage app -> Settings -> Secrets, add GROQ_API_KEY, save, and reboot."
    )
    st.stop()

from pipeline import run_research_pipeline

# --------------------------------------------------------------------------
# Helper: PDF Generation
# --------------------------------------------------------------------------
def generate_simple_pdf(title: str, report: str, feedback: str) -> bytes | None:
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    class DossierPDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 10)
            self.cell(0, 8, "Research Dossier", align="R")
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    pdf = DossierPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    effective_width = pdf.w - pdf.l_margin - pdf.r_margin

    pdf.set_font("Helvetica", "B", 16)
    clean_title = title.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(effective_width, 8, clean_title)
    pdf.ln(4)

    full_text = f"--- RESEARCH REPORT ---\n\n{report}\n\n--- CRITIC EVALUATION ---\n\n{feedback}"
    pdf.set_font("Helvetica", size=9)

    for raw_line in full_text.split("\n"):
        clean_line = raw_line.encode("latin-1", "replace").decode("latin-1")

        if set(clean_line.strip()).issubset({"-", "|", " "}) and len(clean_line.strip()) > 3:
            clean_line = "-" * 40

        words = clean_line.split(" ")
        formatted_words = []
        for word in words:
            if len(word) > 75:
                chunked = [word[i : i + 75] for i in range(0, len(word), 75)]
                formatted_words.append(" ".join(chunked))
            else:
                formatted_words.append(word)
        safe_line = " ".join(formatted_words)

        try:
            pdf.multi_cell(effective_width, 5, safe_line)
        except Exception:
            continue

    return bytes(pdf.output())


# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Multi-Agent Research System")
st.caption("Search Agent → Reader Agent → Writer Chain → Critic Chain")

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "result_state" not in st.session_state:
    st.session_state.result_state = None
if "logs" not in st.session_state:
    st.session_state.logs = ""
if "error" not in st.session_state:
    st.session_state.error = None
if "last_topic" not in st.session_state:
    st.session_state.last_topic = ""
if "last_run_timestamp" not in st.session_state:
    st.session_state.last_run_timestamp = 0.0

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Research Mode")
    selected_mode = st.radio(
        "Select Pipeline Mode:",
        [
            "⚡ Standard Mode",
            "🔬 Deep Mode",
        ],
        index=0,
        help=(
            "⚡ Standard Mode: Fast (~15s), lightweight, safe for continuous back-to-back runs.\n\n"
            "🔬 Deep Mode: Exhaustive 800-1,000 word technical dossier with advanced web crawling."
        ),
    )
    is_deep = selected_mode == "🔬 Deep Mode"

    if is_deep:
        st.info("💡 Deep Mode uses smart pacing. If you run multiple searches, a ~45s safety countdown clears the token bucket automatically.")
    else:
        st.success("⚡ Standard Mode is active. Fast runs with minimal token usage.")

    st.divider()
    show_logs = st.checkbox("Show raw console logs", value=True)
    if st.button("Clear results", use_container_width=True):
        st.session_state.result_state = None
        st.session_state.logs = ""
        st.session_state.error = None
        st.session_state.last_topic = ""
        st.rerun()

# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------
topic = st.text_input(
    "Research topic",
    placeholder="e.g. Next-generation solid-state battery commercialization",
)

btn_label = "Run Deep Research 🔬" if is_deep else "Run Quick Research ⚡"
run_clicked = st.button(btn_label, type="primary", disabled=not topic.strip())

# --------------------------------------------------------------------------
# Run pipeline with Token-Bucket Cooldown Guard
# --------------------------------------------------------------------------
if run_clicked:
    st.session_state.result_state = None
    st.session_state.error = None
    st.session_state.last_topic = topic.strip()

    # Smart Token-Bucket Guard: Prevent back-to-back 429 errors
    now = time.time()
    elapsed = now - st.session_state.last_run_timestamp
    required_gap = 45.0 if is_deep else 8.0

    if elapsed < required_gap:
        wait_seconds = int(required_gap - elapsed) + 1
        with st.status(f"⏳ Cooling down token bucket ({wait_seconds}s)...", expanded=True) as cd_box:
            for rem in range(wait_seconds, 0, -1):
                cd_box.write(f"Clearing Groq rolling 60s window... {rem}s remaining")
                time.sleep(1)
            cd_box.update(label="Token bucket replenished! Starting pipeline...", state="complete")

    st.session_state.last_run_timestamp = time.time()

    log_buffer = io.StringIO()
    status_label = "Executing Deep Research Pipeline..." if is_deep else "Executing Standard Pipeline..."
    status_box = st.status(status_label, expanded=True)

    try:
        status_box.write("Step 1 — Search agent gathering web intelligence...")

        with contextlib.redirect_stdout(log_buffer):
            result = run_research_pipeline(topic.strip(), deep_mode=is_deep)

        status_box.write("Step 2 — Reader agent extracted primary sources.")
        status_box.write("Step 3 — Writer chain synthesized report.")
        status_box.write("Step 4 — Critic chain reviewed and scored report.")

        st.session_state.result_state = result
        st.session_state.logs = log_buffer.getvalue()
        status_box.update(label="Research Complete ✅", state="complete", expanded=False)

    except Exception as e:
        st.session_state.logs = log_buffer.getvalue()
        st.session_state.error = f"{e}\n\n{traceback.format_exc()}"
        status_box.update(label="Execution failed ❌", state="error", expanded=True)

# --------------------------------------------------------------------------
# Error display
# --------------------------------------------------------------------------
if st.session_state.error:
    st.error("The pipeline encountered an execution error:")
    st.code(st.session_state.error, language="text")

# --------------------------------------------------------------------------
# Results display
# --------------------------------------------------------------------------
state = st.session_state.result_state

if state:
    st.divider()
    st.subheader(f"Results: {st.session_state.last_topic}")

    tab_report, tab_feedback, tab_search, tab_scraped = st.tabs(
        ["📄 Final Report", "🧐 Critic Evaluation", "🔍 Search Findings", "📖 Primary Sources"]
    )

    report_text = str(state.get("report", ""))
    feedback_text = str(state.get("feedback", ""))
    sanitized_name = st.session_state.last_topic.replace(" ", "_").lower()

    combined_dossier = (
        f"# Research Dossier: {st.session_state.last_topic}\n\n"
        f"{report_text}\n\n"
        f"---\n\n"
        f"# Peer Evaluation & Critique\n\n"
        f"{feedback_text}\n"
    )

    with tab_report:
        st.markdown(report_text)
        st.divider()

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.download_button(
                "📥 Download Report (.md)",
                data=report_text,
                file_name=f"report_{sanitized_name}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                "📦 Full Dossier (.md)",
                data=combined_dossier,
                file_name=f"dossier_{sanitized_name}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col3:
            pdf_bytes = generate_simple_pdf(st.session_state.last_topic, report_text, feedback_text)
            if pdf_bytes:
                st.download_button(
                    "📄 Download PDF",
                    data=pdf_bytes,
                    file_name=f"dossier_{sanitized_name}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

    with tab_feedback:
        st.markdown(feedback_text)

    with tab_search:
        st.markdown(state.get("search_results", ""))

    with tab_scraped:
        st.markdown(state.get("scraped_content", ""))

if show_logs and st.session_state.logs:
    with st.expander("Live console logs", expanded=False):
        st.code(st.session_state.logs, language="text")

if not state and not run_clicked:
    st.info("Choose a mode in the sidebar, enter a topic, and click **Run**.")
