"""
Streamlit UI for the multi-agent research pipeline (pipeline.py).

Place this file in the SAME folder as pipeline.py, agents.py, and tools.py:
    .
    ├── agents.py
    ├── app.py          <- this file
    ├── pipeline.py
    ├── requirements.txt
    └── tools.py

Run with:
    streamlit run app.py
"""

import contextlib
import io
import os
import traceback
import streamlit as st

# --------------------------------------------------------------------------
# Inject Streamlit Secrets into OS Environment BEFORE importing pipeline
# --------------------------------------------------------------------------
try:
    if hasattr(st, "secrets"):
        for key, value in st.secrets.items():
            if isinstance(value, str):
                os.environ[key] = value
except Exception:
    pass

from pipeline import run_research_pipeline

# --------------------------------------------------------------------------
# Helper: PDF Generation
# --------------------------------------------------------------------------
def generate_simple_pdf(title: str, report: str, feedback: str) -> bytes | None:
    """Generates a downloadable PDF dossier using fpdf2 if installed, handling wide tables/strings."""
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    class DossierPDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 10)
            self.cell(0, 8, "Autonomous Research Dossier", align="R")
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    pdf = DossierPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    effective_width = pdf.w - pdf.l_margin - pdf.r_margin

    # Title
    pdf.set_font("Helvetica", "B", 16)
    clean_title = title.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(effective_width, 8, clean_title)
    pdf.ln(4)

    # Content
    full_text = f"--- RESEARCH REPORT ---\n\n{report}\n\n--- CRITIC EVALUATION ---\n\n{feedback}"
    pdf.set_font("Helvetica", size=9)
    
    for raw_line in full_text.split("\n"):
        clean_line = raw_line.encode("latin-1", "replace").decode("latin-1")
        
        # Collapse markdown table divider rows
        if set(clean_line.strip()).issubset({"-", "|", " "}) and len(clean_line.strip()) > 3:
            clean_line = "-" * 40
            
        # Break up any continuous string exceeding 75 characters
        words = clean_line.split(" ")
        formatted_words = []
        for word in words:
            if len(word) > 75:
                chunked = [word[i:i+75] for i in range(0, len(word), 75)]
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
st.caption("Search agent → Reader agent → Writer chain → Critic chain")


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


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "This UI drives `run_research_pipeline` across 4 autonomous stages: "
        "Search, Web Scraping, Report Synthesis, and Peer Critique."
    )
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
    placeholder="e.g. Latest advances in solid-state batteries",
)

run_clicked = st.button("Run pipeline 🚀", type="primary", disabled=not topic.strip())


# --------------------------------------------------------------------------
# Run pipeline
# --------------------------------------------------------------------------
if run_clicked:
    st.session_state.result_state = None
    st.session_state.error = None
    st.session_state.last_topic = topic.strip()

    log_buffer = io.StringIO()
    status_box = st.status("Executing research pipeline...", expanded=True)

    try:
        status_box.write("Step 1 — Search agent gathering sources via Tavily...")
        
        with contextlib.redirect_stdout(log_buffer):
            result = run_research_pipeline(topic.strip())

        status_box.write("Step 2 — Reader agent scraped top sources.")
        status_box.write("Step 3 — Writer chain synthesized report draft.")
        status_box.write("Step 4 — Critic chain scored and reviewed final findings.")

        st.session_state.result_state = result
        st.session_state.logs = log_buffer.getvalue()
        status_box.update(label="Pipeline run complete ✅", state="complete", expanded=False)

    except Exception as e:
        st.session_state.logs = log_buffer.getvalue()
        st.session_state.error = f"{e}\n\n{traceback.format_exc()}"
        status_box.update(label="Pipeline execution failed ❌", state="error", expanded=True)


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
    st.subheader(f"Research Results: {st.session_state.last_topic}")

    tab_report, tab_feedback, tab_search, tab_scraped = st.tabs(
        ["📄 Final Report", "🧐 Critic Feedback", "🔍 Search Results", "📖 Scraped Content"]
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
        st.text(state.get("search_results", ""))

    with tab_scraped:
        st.text(state.get("scraped_content", ""))

if show_logs and st.session_state.logs:
    with st.expander("Raw console output", expanded=False):
        st.code(st.session_state.logs, language="text")

if not state and not run_clicked:
    st.info("Enter a topic above and click **Run pipeline 🚀** to start.")
