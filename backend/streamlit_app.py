"""
streamlit_app.py — Nyaya Legal AI · Streamlit Interface

A beautiful, full-featured UI to test the Nyaya legal judgment analysis pipeline.
Connects to the local FastAPI backend at http://localhost:8000.

Run:
    streamlit run streamlit_app.py
"""

import json
import time

import streamlit as st
import requests
import httpx

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Nyaya Legal AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"

# ── Session state initialisation (MUST happen before any widget) ──────────────
if "judgment_text" not in st.session_state:
    st.session_state["judgment_text"] = ""
if "source_label" not in st.session_state:
    st.session_state["source_label"] = ""
if "final_output" not in st.session_state:
    st.session_state["final_output"] = None
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False
if "stream_mode" not in st.session_state:
    st.session_state["stream_mode"] = True

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0e1a !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] {
    background: #0f1525 !important;
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] .block-container { padding-top: 2rem; }

.nyaya-hero {
    background: linear-gradient(135deg, #1a2744 0%, #0f1a35 50%, #141f3d 100%);
    border: 1px solid #2a3d6e;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.nyaya-hero::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 40%; height: 200%;
    background: radial-gradient(ellipse, rgba(99,179,237,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.nyaya-hero h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #63b3ed, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0 !important;
}
.nyaya-hero p { color: #94a3b8 !important; font-size: 0.95rem !important; margin: 0 !important; }
.hero-badge {
    display: inline-block;
    background: rgba(99,179,237,0.12);
    border: 1px solid rgba(99,179,237,0.3);
    color: #63b3ed;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-bottom: 0.75rem;
}

.result-card {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.result-card:hover { border-color: #2a4a7f; }
.result-card h3 {
    color: #63b3ed !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600 !important;
    margin: 0 0 0.5rem 0 !important;
}
.result-card .value { color: #e2e8f0; font-size: 1rem; line-height: 1.6; }

.conf-bar-wrap {
    background: #1a2744;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
    border: 1px solid #1e2d4a;
}
.conf-label { display: flex; justify-content: space-between; margin-bottom: 0.4rem; font-size: 0.85rem; color: #94a3b8; }
.conf-bar { height: 6px; border-radius: 3px; background: #1e2d4a; overflow: hidden; }
.conf-fill { height: 100%; border-radius: 3px; }
.conf-fill.high { background: linear-gradient(90deg, #48bb78, #68d391); }
.conf-fill.med  { background: linear-gradient(90deg, #ed8936, #f6ad55); }
.conf-fill.low  { background: linear-gradient(90deg, #f56565, #fc8181); }

.tag { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.78rem; font-weight: 500; margin: 0.15rem; }
.tag-verified     { background: rgba(72,187,120,0.15); color: #68d391; border: 1px solid rgba(72,187,120,0.3); }
.tag-hallucinated { background: rgba(245,101,101,0.15); color: #fc8181; border: 1px solid rgba(245,101,101,0.3); }
.tag-issue        { background: rgba(99,179,237,0.12); color: #90cdf4; border: 1px solid rgba(99,179,237,0.25); }
.tag-unresolved   { background: rgba(237,137,54,0.15); color: #f6ad55; border: 1px solid rgba(237,137,54,0.3); }

.status-pill { display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
.status-ok    { background: rgba(72,187,120,0.15); color: #68d391; border: 1px solid rgba(72,187,120,0.3); }
.status-error { background: rgba(245,101,101,0.15); color: #fc8181; border: 1px solid rgba(245,101,101,0.3); }

.event-row {
    display: flex; align-items: center; gap: 0.75rem;
    padding: 0.5rem 0.75rem; border-radius: 8px; margin-bottom: 0.35rem;
    background: rgba(30,45,74,0.4); font-size: 0.85rem; color: #94a3b8;
    animation: fadeIn 0.3s ease;
}
.event-row .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.dot-green  { background: #48bb78; box-shadow: 0 0 6px #48bb78; }
.dot-blue   { background: #63b3ed; box-shadow: 0 0 6px #63b3ed; }
.dot-orange { background: #ed8936; box-shadow: 0 0 6px #ed8936; }
.dot-purple { background: #a78bfa; box-shadow: 0 0 6px #a78bfa; }
.dot-red    { background: #f56565; box-shadow: 0 0 6px #f56565; }
@keyframes fadeIn { from { opacity:0; transform:translateX(-8px); } to { opacity:1; transform:translateX(0); } }

.stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; padding: 0.5rem 1.5rem !important;
    font-size: 0.9rem !important; transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stTextArea textarea {
    background: #111827 !important; border: 1px solid #1e2d4a !important;
    border-radius: 8px !important; color: #e2e8f0 !important;
    font-family: 'Inter', monospace !important; font-size: 0.88rem !important;
}
.stTextArea textarea:focus { border-color: #2563eb !important; box-shadow: 0 0 0 2px rgba(37,99,235,0.2) !important; }
button[data-baseweb="tab"] { background: transparent !important; color: #64748b !important; font-weight: 500 !important; border-bottom: 2px solid transparent !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #63b3ed !important; border-bottom-color: #63b3ed !important; }
[data-testid="stMetricValue"] { color: #63b3ed !important; font-size: 1.8rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.8rem !important; }
div.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def check_api_health() -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=4)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def confidence_class(score: float) -> str:
    if score >= 0.75: return "high"
    if score >= 0.5:  return "med"
    return "low"


def confidence_color(score: float) -> str:
    if score >= 0.75: return "#48bb78"
    if score >= 0.5:  return "#ed8936"
    return "#f56565"


def node_dot_class(node: str) -> str:
    return {"extract":"dot-blue","validate_statutes":"dot-green",
            "resolve_precedents":"dot-orange","score_confidence":"dot-purple",
            "assemble":"dot-green"}.get(node, "dot-blue")


def node_label(node: str) -> str:
    return {"extract":"🧠 Extracting structured data (Nyaya-7B)",
            "validate_statutes":"📚 Validating statutes against corpus",
            "resolve_precedents":"🔍 Resolving legal citations",
            "score_confidence":"📊 Scoring confidence per field",
            "assemble":"✅ Assembling final output"}.get(node, f"⚙️ {node}")


def render_confidence_bars(scores: dict):
    for field, score in scores.items():
        css_class = confidence_class(score)
        pct = int(score * 100)
        st.markdown(f"""
        <div class="conf-bar-wrap">
            <div class="conf-label">
                <span>{field.replace('_',' ').title()}</span>
                <span style="color:{confidence_color(score)};font-weight:600">{pct}%</span>
            </div>
            <div class="conf-bar">
                <div class="conf-fill {css_class}" style="width:{pct}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def analyze_text_streaming(judgment_text: str):
    payload = {"judgment_text": judgment_text, "stream": True}
    with httpx.stream("POST", f"{API_BASE}/analyze/text", json=payload,
                      timeout=600.0, headers={"Accept": "text/event-stream"}) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip():
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        pass


def analyze_text_blocking(judgment_text: str) -> dict:
    payload = {"judgment_text": judgment_text, "stream": False}
    r = requests.post(f"{API_BASE}/analyze/text", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()


def analyze_file_streaming(file_bytes: bytes, filename: str):
    with httpx.stream("POST", f"{API_BASE}/analyze",
                      files={"file": (filename, file_bytes, "application/octet-stream")},
                      timeout=600.0) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip():
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        pass


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;margin-bottom:1.5rem">
        <span style="font-size:2.5rem">⚖️</span>
        <h2 style="color:#63b3ed;margin:0.5rem 0 0 0;font-family:'Playfair Display',serif">Nyaya AI</h2>
        <p style="color:#475569;font-size:0.8rem;margin-top:0.25rem">Indian Legal Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    health = check_api_health()
    if health:
        st.markdown('<div class="status-pill status-ok">✓ API Online</div>', unsafe_allow_html=True)
        st.caption(f"Model loaded: {'Yes' if health.get('model_loaded') else 'No (loads on first request)'}")
        st.caption(f"Statutes in index: {health.get('statute_index_size', 0):,}")
        st.caption(f"Precedents in index: {health.get('precedent_index_size', 0):,}")
    else:
        st.markdown('<div class="status-pill status-error">✗ API Offline</div>', unsafe_allow_html=True)
        st.error("Start: `uvicorn api.main:app --port 8000` in the backend dir.", icon="🚫")

    st.divider()
    st.markdown("### ⚙️ Settings")
    # Give the toggle a stable key
    st.session_state["stream_mode"] = st.toggle(
        "Live streaming mode", value=st.session_state["stream_mode"],
        key="toggle_stream_mode",
        help="Stream results agent-by-agent as they complete"
    )

    st.divider()
    st.markdown("### 📖 About")
    st.markdown("""
    <div style="color:#64748b;font-size:0.82rem;line-height:1.7">
    Nyaya-7B is a <b style="color:#94a3b8">finetuned Mistral-7B</b> model trained on 10,000+ Indian court judgments.
    <br><br>
    The pipeline runs 5 agents:<br>
    <b style="color:#63b3ed">1.</b> Extraction (Nyaya-7B)<br>
    <b style="color:#63b3ed">2.</b> Statute validation (RAG)<br>
    <b style="color:#63b3ed">3.</b> Citation resolution<br>
    <b style="color:#63b3ed">4.</b> Confidence scoring<br>
    <b style="color:#63b3ed">5.</b> Final assembly
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    if st.button("🔄 Refresh API status", key="btn_refresh"):
        st.rerun()


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nyaya-hero">
    <div class="hero-badge">⚖️ Powered by Local Nyaya-7B</div>
    <h1>Nyaya Legal AI</h1>
    <p>Analyze Indian court judgments with a finetuned LLM — extract parties, statutes, citations, holdings and more with confidence scoring and hallucination detection.</p>
</div>
""", unsafe_allow_html=True)

# ── Input tabs ────────────────────────────────────────────────────────────────
SAMPLE = """IN THE SUPREME COURT OF INDIA
Criminal Appeal No. 1481 of 2018
(Arising out of SLP (Crl.) No. 9525 of 2017)

STATE OF PUNJAB                          ...Appellant
                 Versus
GURPREET SINGH @ GOPI & ANR.             ...Respondents

JUDGMENT

J. Chelameswar, J.

The State of Punjab challenges the judgment dated 20.09.2017 passed by the
High Court of Punjab and Haryana at Chandigarh in Criminal Appeal No. 1234-SB
of 2010, whereby the High Court acquitted the respondents of the offences
punishable under Sections 302 and 34 of the Indian Penal Code.

The prosecution case, in brief, is that on 15.03.2009, the deceased Harjinder
Singh was allegedly murdered by the respondents. The Trial Court convicted the
respondents under Section 302 IPC read with Section 34 IPC and sentenced them
to undergo imprisonment for life.

The High Court, on reappreciation of the evidence on record, acquitted both
the accused on the ground that the prosecution failed to prove the case beyond
reasonable doubt.

The learned counsel for the State relied upon Sharad Birdhichand Sarda v. State
of Maharashtra, AIR 1984 SC 1622 to urge that the High Court erred in
reappreciating the evidence. The court further considered (2019) 5 SCC 1 as a
relevant precedent on the standard of proof in criminal matters.

HELD: The appeal is allowed. The judgment of the High Court is set aside and
the conviction recorded by the Trial Court is restored. The case involved
eyewitness testimony supported by forensic evidence, which the High Court
erroneously discarded. The chain of evidence was complete and consistent with
the guilt of the accused. Therefore, the respondents are convicted under
Section 302 read with Section 34 of the Indian Penal Code."""

tab_text, tab_pdf, tab_sample = st.tabs(["📝 Paste Judgment Text", "📄 Upload PDF / TXT", "💡 Try a Sample"])

with tab_text:
    pasted = st.text_area(
        "Paste judgment text",
        height=300,
        placeholder="IN THE SUPREME COURT OF INDIA\nCriminal Appeal No. ...\n\nJUDGMENT\n...",
        label_visibility="collapsed",
        key="textarea_judgment",
    )
    if pasted and pasted.strip():
        st.session_state["judgment_text"] = pasted
        st.session_state["source_label"]  = "pasted text"

with tab_pdf:
    uploaded = st.file_uploader(
        "Upload judgment PDF or TXT (max 10 MB)",
        type=["pdf", "txt", "text"],
        label_visibility="collapsed",
        key="file_uploader_judgment",
    )
    if uploaded:
        st.success(f"Loaded: **{uploaded.name}** ({uploaded.size / 1024:.1f} KB)")
        st.session_state["source_label"] = uploaded.name

with tab_sample:
    st.markdown("""
    <div style="color:#94a3b8;font-size:0.88rem;margin-bottom:1rem">
    Use this pre-loaded Supreme Court judgment to test the pipeline without your own data.
    </div>
    """, unsafe_allow_html=True)
    st.code(SAMPLE[:600] + "\n...[truncated for display]", language="text")
    if st.button("📋 Use this sample", use_container_width=True, key="btn_use_sample"):
        st.session_state["judgment_text"] = SAMPLE
        st.session_state["source_label"]  = "sample judgment"
        st.session_state["analysis_done"] = False
        st.session_state["final_output"]  = None
        st.success("Sample loaded! Click **⚡ Analyze Judgment** below.")

st.markdown("<br>", unsafe_allow_html=True)

# Resolve final judgment text (file upload overrides paste if both present)
judgment_text = ""
source_label  = st.session_state.get("source_label", "")

if uploaded:
    # Will be read at analysis time
    judgment_text = "__FILE__"
elif st.session_state.get("judgment_text"):
    judgment_text = st.session_state["judgment_text"]

# ── Analyze button row ────────────────────────────────────────────────────────
col_btn, col_info = st.columns([1, 3])
with col_btn:
    analyze_clicked = st.button("⚡ Analyze Judgment", use_container_width=True,
                                type="primary", key="btn_analyze")
with col_info:
    if judgment_text:
        text_len = len(st.session_state.get("judgment_text", "")) if judgment_text != "__FILE__" else (uploaded.size if uploaded else 0)
        st.markdown(
            f'<div style="padding:0.5rem 0;color:#64748b;font-size:0.85rem">'
            f'📄 {text_len:,} chars from <b style="color:#94a3b8">{source_label}</b>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── Run analysis ──────────────────────────────────────────────────────────────
if analyze_clicked:
    if not health:
        st.error("❌ Cannot connect to Nyaya API at http://localhost:8000. Is uvicorn running?")
        st.stop()

    # Resolve actual text / file bytes
    actual_text  = None
    file_bytes   = None
    file_name    = None

    if uploaded:
        file_bytes = uploaded.read()
        file_name  = uploaded.name
    elif st.session_state.get("judgment_text"):
        actual_text = st.session_state["judgment_text"]
    else:
        st.warning("⚠️ Please paste judgment text, upload a file, or use the sample.")
        st.stop()

    if actual_text and len(actual_text.strip()) < 200:
        st.warning("⚠️ Text too short (minimum 200 characters).")
        st.stop()

    st.markdown("---")

    progress_header = st.empty()
    progress_header.markdown("### 🔄 Processing Pipeline")
    event_area  = st.empty()
    final_output = None
    had_error    = False
    t_start      = time.time()

    stream_mode = st.session_state.get("stream_mode", True)

    if stream_mode:
        try:
            if file_bytes:
                gen = analyze_file_streaming(file_bytes, file_name)
            else:
                gen = analyze_text_streaming(actual_text)

            events_html = []
            with st.spinner("🧠 Loading Nyaya-7B and running pipeline (first call ~1-2 min)..."):
                for evt in gen:
                    etype = evt.get("event") or evt.get("node")
                    if etype == "start":
                        events_html.append('<div class="event-row"><div class="dot dot-blue"></div>Pipeline started</div>')
                    elif etype == "complete":
                        elapsed = time.time() - t_start
                        events_html.append(f'<div class="event-row"><div class="dot dot-green"></div>✅ Done in {elapsed:.1f}s</div>')
                    elif etype == "error":
                        events_html.append(f'<div class="event-row"><div class="dot dot-red"></div>❌ {evt.get("error","unknown")}</div>')
                        had_error = True
                    else:
                        node = evt.get("node", etype or "")
                        dot  = node_dot_class(node)
                        lbl  = node_label(node)
                        conf = evt.get("overall_confidence")
                        conf_str = f" · {conf:.0%} confidence" if conf is not None else ""
                        events_html.append(f'<div class="event-row"><div class="dot {dot}"></div>{lbl}{conf_str}</div>')
                        if evt.get("final_output"):
                            final_output = evt["final_output"]

                    event_area.markdown("<div>" + "".join(events_html) + "</div>", unsafe_allow_html=True)

        except httpx.HTTPStatusError as e:
            st.error(f"API error {e.response.status_code}: {e.response.text[:400]}")
            had_error = True
        except Exception as e:
            st.error(f"Connection error: {e}")
            had_error = True
    else:
        with st.spinner("🧠 Running pipeline (may take 1–5 min on first run)..."):
            try:
                data = analyze_text_blocking(actual_text)
                final_output = data.get("final_output")
                for err in data.get("errors", []):
                    st.warning(f"Pipeline warning: {err}")
            except Exception as e:
                st.error(f"Error: {e}")
                had_error = True

    if had_error:
        st.stop()

    if final_output is None:
        st.info("⏳ No output yet. The model may still be loading — please try again shortly.", icon="⏳")
        st.stop()

    st.session_state["final_output"]  = final_output
    st.session_state["analysis_done"] = True

    elapsed_total = time.time() - t_start
    progress_header.markdown(
        f"### ✅ Analysis Complete  <span style='font-size:0.85rem;color:#64748b;font-weight:400'>({elapsed_total:.1f}s)</span>",
        unsafe_allow_html=True,
    )

# ── Results (persisted in session_state) ─────────────────────────────────────
if st.session_state.get("analysis_done") and st.session_state.get("final_output"):
    final_output = st.session_state["final_output"]

    st.markdown("---")
    st.markdown("## 📋 Extraction Results")

    overall_conf = final_output.get("overall_confidence", 0)
    n_statutes   = len(final_output.get("statutes_cited", []))
    n_hall       = len(final_output.get("hallucinated_statutes", []))
    n_prec       = len(final_output.get("precedents_cited", []))
    needs_review = final_output.get("needs_human_review", False)

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Overall Confidence", f"{overall_conf:.0%}")
    with m2: st.metric("Statutes Found", n_statutes)
    with m3: st.metric("Precedents", n_prec)
    with m4: st.metric("Hallucinations", n_hall,
                        delta="⚠️ Review" if n_hall > 0 else "✓ Clean",
                        delta_color="inverse" if n_hall > 0 else "normal")

    if needs_review:
        st.warning("⚠️ **Human review recommended** — low confidence fields or hallucinated statutes detected.", icon="🔍")
    else:
        st.success("✅ **High confidence** — all fields passed validation.", icon="✔️")

    r1, r2, r3, r4, r5 = st.tabs(["🏛️ Case Info", "📚 Statutes", "⚖️ Precedents", "📊 Confidence", "🔍 Raw JSON"])

    def card(label: str, value, empty_msg="—"):
        val = value if value else empty_msg
        st.markdown(f"""
        <div class="result-card">
            <h3>{label}</h3>
            <div class="value">{val}</div>
        </div>
        """, unsafe_allow_html=True)

    with r1:
        st.markdown("#### Core Case Information")
        col_a, col_b = st.columns(2)
        with col_a:
            card("Case Name",      final_output.get("case_name"))
            card("Petitioner",     final_output.get("petitioner"))
            card("Court",          final_output.get("court"))
            card("Year",           str(final_output.get("year") or "—"))
        with col_b:
            card("Citation",       final_output.get("citation"))
            card("Respondent",     final_output.get("respondent"))
            card("Subject Matter", final_output.get("subject_matter"))
            card("Outcome",        final_output.get("outcome"))
        card("Holding", final_output.get("holding"))
        issues = final_output.get("legal_issues", [])
        if issues:
            tags = "".join(f'<span class="tag tag-issue">{i}</span>' for i in issues)
            st.markdown(f'<div class="result-card"><h3>Legal Issues ({len(issues)})</h3><div style="margin-top:0.25rem">{tags}</div></div>', unsafe_allow_html=True)
        bench = final_output.get("bench", [])
        if bench:
            card("Bench", " · ".join(bench))

    with r2:
        st.markdown("#### Statutes Cited")
        statutes = final_output.get("statutes_cited", [])
        if not statutes:
            st.info("No statutes extracted.")
        for s in statutes:
            verified    = s.get("verified", False)
            tag_cls     = "tag-verified" if verified else "tag-hallucinated"
            tag_txt     = "✓ Verified" if verified else "⚠ Flagged"
            s_act       = s.get("act", "")
            s_sec       = s.get("section", "")
            desc        = s.get("description") or ""
            actual      = s.get("actual_text") or ""
            desc_html   = f'<div style="color:#94a3b8;font-size:0.88rem;margin-top:0.4rem">{desc}</div>' if desc else ""
            actual_html = f'<div style="font-size:0.78rem;color:#64748b;margin-top:0.5rem;font-style:italic">{actual[:200]}</div>' if actual else ""
            st.html(f"""<div class="result-card" style="background:#111827;border:1px solid #1e2d4a;border-radius:12px;padding:1.25rem 1.5rem;margin-bottom:1rem">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div><b style="color:#e2e8f0">{s_act}</b><span style="color:#63b3ed;margin-left:0.5rem">§{s_sec}</span></div>
    <span style="display:inline-block;padding:0.2rem 0.6rem;border-radius:6px;font-size:0.78rem;font-weight:500;{'background:rgba(72,187,120,0.15);color:#68d391;border:1px solid rgba(72,187,120,0.3)' if verified else 'background:rgba(245,101,101,0.15);color:#fc8181;border:1px solid rgba(245,101,101,0.3)'}">{tag_txt}</span>
  </div>{desc_html}{actual_html}</div>""")
        hall = final_output.get("hallucinated_statutes", [])
        if hall:
            st.markdown("##### ⚠️ Hallucinated / Unverified")
            st.html("".join(f'<span style="display:inline-block;padding:0.2rem 0.6rem;border-radius:6px;font-size:0.78rem;font-weight:500;background:rgba(245,101,101,0.15);color:#fc8181;border:1px solid rgba(245,101,101,0.3);margin:0.15rem">{h}</span>' for h in hall))

    with r3:
        st.markdown("#### Precedents Cited")
        precedents = final_output.get("precedents_cited", [])
        if not precedents:
            st.info("No precedents extracted.")
        for p in precedents:
            resolved      = p.get("resolved", False)
            tag_cls       = "tag-verified" if resolved else "tag-unresolved"
            tag_txt       = ("✓ " + p.get("source", "resolved")) if resolved else "⚠ Unresolved"
            p_citation    = p.get("citation", "")
            p_case        = p.get("case_name", "")
            p_court       = p.get("court", "")
            p_year        = str(p.get("year", "")) if p.get("year") else ""
            summary       = p.get("summary") or ""
            # Build sub-parts as clean strings (no nesting)
            case_html    = f'<div style="color:#94a3b8;font-size:0.85rem;margin-top:0.2rem">{p_case}</div>' if p_case else ""
            meta_parts   = []
            if p_court: meta_parts.append(f'<span>🏛️ {p_court}</span>')
            if p_year:  meta_parts.append(f'<span>📅 {p_year}</span>')
            meta_html    = f'<div style="display:flex;gap:1rem;margin-top:0.5rem;font-size:0.8rem;color:#64748b">{"".join(meta_parts)}</div>' if meta_parts else ""
            summary_html = f'<div style="color:#64748b;font-size:0.78rem;margin-top:0.4rem;font-style:italic">{summary[:200]}</div>' if summary else ""
            tag_style    = "background:rgba(72,187,120,0.15);color:#68d391;border:1px solid rgba(72,187,120,0.3)" if resolved else "background:rgba(237,137,54,0.15);color:#f6ad55;border:1px solid rgba(237,137,54,0.3)"
            st.html(f"""<div style="background:#111827;border:1px solid #1e2d4a;border-radius:12px;padding:1.25rem 1.5rem;margin-bottom:1rem">
  <div style="display:flex;justify-content:space-between;align-items:flex-start">
    <div><b style="color:#e2e8f0">{p_citation}</b>{case_html}</div>
    <span style="display:inline-block;padding:0.2rem 0.6rem;border-radius:6px;font-size:0.78rem;font-weight:500;flex-shrink:0;margin-left:0.75rem;{tag_style}">{tag_txt}</span>
  </div>{meta_html}{summary_html}</div>""")

    with r4:
        st.markdown("#### Per-Field Confidence Scores")
        conf_scores = final_output.get("confidence_scores", {})
        if conf_scores:
            render_confidence_bars(conf_scores)
        else:
            st.info("No confidence scores available.")
        uncertain = final_output.get("uncertain_fields", [])
        if uncertain:
            st.markdown("##### ⚠️ Fields Flagged for Review")
            st.html("".join(f'<span style="display:inline-block;padding:0.2rem 0.6rem;border-radius:6px;font-size:0.78rem;font-weight:500;background:rgba(245,101,101,0.15);color:#fc8181;border:1px solid rgba(245,101,101,0.3);margin:0.15rem">{f.replace("_"," ").title()}</span>' for f in uncertain))
        else:
            st.success("✅ All fields ≥ 70% confidence.")

    with r5:
        st.markdown("#### Raw JSON Output")
        st.json(final_output, expanded=False)

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📥 Export")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            label="⬇️ Download JSON",
            data=json.dumps(final_output, indent=2, ensure_ascii=False),
            file_name=f"nyaya_analysis_{int(time.time())}.json",
            mime="application/json",
            use_container_width=True,
            key="dl_json",
        )
    with c2:
        lines = [
            "NYAYA LEGAL AI ANALYSIS",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M')}",
            "",
            f"Case: {final_output.get('case_name','')}",
            f"Court: {final_output.get('court','')}",
            f"Year: {final_output.get('year','')}",
            f"Outcome: {final_output.get('outcome','')}",
            f"Overall Confidence: {overall_conf:.0%}",
            "",
            "HOLDING:",
            final_output.get("holding", ""),
            "",
            "STATUTES CITED:",
        ]
        for s in final_output.get("statutes_cited", []):
            lines.append(f"  - {s.get('act','')} §{s.get('section','')} [{'verified' if s.get('verified') else 'FLAGGED'}]")
        st.download_button(
            label="⬇️ Download Text Summary",
            data="\n".join(lines),
            file_name=f"nyaya_summary_{int(time.time())}.txt",
            mime="text/plain",
            use_container_width=True,
            key="dl_txt",
        )

    if st.button("🗑️ Clear Results", key="btn_clear"):
        st.session_state["final_output"]  = None
        st.session_state["analysis_done"] = False
        st.session_state["judgment_text"] = ""
        st.session_state["source_label"]  = ""
        st.rerun()

elif not st.session_state.get("analysis_done"):
    st.markdown("""
    <div style="text-align:center;padding:2.5rem;color:#475569">
        <div style="font-size:3rem;margin-bottom:1rem">⚖️</div>
        <p style="font-size:1rem;color:#64748b">
            Paste a judgment, upload a PDF, or try the sample above —
            then click <b style="color:#63b3ed">⚡ Analyze Judgment</b>.
        </p>
        <p style="font-size:0.85rem;color:#374151;margin-top:0.5rem">
            First analysis loads <b style="color:#94a3b8">Nyaya-7B</b> locally (~1–2 min). Subsequent runs are faster.
        </p>
    </div>
    """, unsafe_allow_html=True)
