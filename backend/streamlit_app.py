"""
streamlit_app.py — Nyaya Legal AI · Streamlit interface.

Talks to the local FastAPI backend at http://127.0.0.1:8000. Theme (colours,
fonts, light/dark) lives in .streamlit/config.toml.

Run from the backend/ directory:
    run.bat                           # starts the API + this UI with the project venv
    streamlit run streamlit_app.py    # UI only (venv activated, API already running)
"""

import html
import json
import sys
import time
from urllib.parse import quote

import httpx
import streamlit as st

API_BASE = "http://127.0.0.1:8000"
CONFIDENCE_THRESHOLD = 0.70   # matches agents/confidence_agent.py
MIN_STREAMLIT = (1, 58)       # theming + layout APIs used below; the project venv (backend\.nyaya) has it

# A plain `streamlit run` in a new terminal picks up Anaconda's older Streamlit, which
# crashes half-way through the page. Stop early with instructions instead (before
# set_page_config, whose arguments the old version also rejects).
if tuple(int(p) for p in st.__version__.split(".")[:2]) < MIN_STREAMLIT:
    st.error(
        f"**Wrong Python environment.** Nyaya needs Streamlit {'.'.join(map(str, MIN_STREAMLIT))} or newer, "
        f"but this is Streamlit {st.__version__} from `{sys.executable}`.\n\n"
        "Close this and start the app from the `backend` folder with **`run.bat`**, or activate the "
        "project environment first:"
    )
    st.code(".nyaya\\Scripts\\activate\nstreamlit run streamlit_app.py", language="bat")
    st.stop()

st.set_page_config(
    page_title="Nyaya · Judgment Analysis",
    page_icon=":material/balance:",
    layout="wide",
    initial_sidebar_state="expanded",
)

for key, default in {"result": None, "textarea_judgment": "", "stream_mode": True}.items():
    st.session_state.setdefault(key, default)


# ── Design tokens ─────────────────────────────────────────────────────────────
# Status colours follow the fixed status palette (good / warning / critical) and
# always ship with an icon + label, never colour alone.

TOKENS = {
    "light": {
        "surface": "#ffffff", "sunken": "#f3f2ed", "border": "#e2e0d8",
        "ink": "#1b1b19", "ink2": "#55544d", "muted": "#8a8880",
        "accent": "#22407a", "accent-fill": "#22407a", "accent-soft": "#e7ecf5",
        "good": "#0ca30c", "good-ink": "#0a6b0a", "good-soft": "#e3f3e3",
        "warn": "#fab219", "warn-ink": "#7a5200", "warn-soft": "#fdf1d6",
        "crit": "#d03b3b", "crit-ink": "#a32929", "crit-soft": "#f8e3e3",
    },
    "dark": {
        "surface": "#1b1b1a", "sunken": "#222220", "border": "#31312e",
        "ink": "#f2f1ec", "ink2": "#c3c2b7", "muted": "#898781",
        "accent": "#8fb3ef", "accent-fill": "#3f73d6", "accent-soft": "#1d2a40",
        "good": "#0ca30c", "good-ink": "#5fd35f", "good-soft": "#16291a",
        "warn": "#fab219", "warn-ink": "#fbc85a", "warn-soft": "#2e2614",
        "crit": "#d03b3b", "crit-ink": "#f08a8a", "crit-soft": "#331b1b",
    },
}


def _active_theme() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def inject_css() -> None:
    tokens = "\n".join(f"  --n-{k}: {v};" for k, v in TOKENS[_active_theme()].items())
    st.html(f"""<style>
:root {{
{tokens}
  --n-serif: "Source Serif 4", Georgia, "Times New Roman", serif;
  --n-mono: "JetBrains Mono", ui-monospace, Consolas, monospace;
}}
[data-testid="stDecoration"] {{ display: none; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ max-width: 1180px; padding-top: 2.25rem; padding-bottom: 3rem; }}

/* Top bar */
.n-topbar {{ display: flex; align-items: center; justify-content: space-between; gap: 1rem;
  padding-bottom: 1.25rem; border-bottom: 1px solid var(--n-border); flex-wrap: wrap; }}
.n-brand {{ display: flex; align-items: center; gap: .9rem; }}
.n-logo {{ width: 42px; height: 42px; border-radius: 10px; background: var(--n-accent-fill); color: #fff;
  display: grid; place-items: center; flex-shrink: 0; }}
.n-title {{ font-family: var(--n-serif); font-size: 1.6rem; font-weight: 600; line-height: 1.15;
  color: var(--n-ink); letter-spacing: -.01em; }}
.n-subtitle {{ color: var(--n-ink2); font-size: .92rem; margin-top: .15rem; }}
.n-chip {{ display: inline-flex; align-items: center; gap: .45rem; padding: .35rem .75rem; border-radius: 999px;
  border: 1px solid var(--n-border); background: var(--n-surface); color: var(--n-ink2); font-size: .8rem; font-weight: 500; }}

/* Section labels */
.n-label {{ font-size: .72rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
  color: var(--n-muted); margin: 0 0 .5rem; }}

/* Badges: icon + label, tinted background */
.n-badge {{ display: inline-flex; align-items: center; gap: .35rem; padding: .22rem .6rem; border-radius: 6px;
  font-size: .78rem; font-weight: 600; white-space: nowrap; line-height: 1.4; }}
.n-ico {{ display: inline-block; flex-shrink: 0; background-color: currentColor;
  -webkit-mask: var(--n-mask) center / contain no-repeat; mask: var(--n-mask) center / contain no-repeat; }}
{_icon_css()}
.n-good {{ background: var(--n-good-soft); color: var(--n-good-ink); }}
.n-warn {{ background: var(--n-warn-soft); color: var(--n-warn-ink); }}
.n-crit {{ background: var(--n-crit-soft); color: var(--n-crit-ink); }}
.n-neutral {{ background: var(--n-sunken); color: var(--n-ink2); }}
.n-accent {{ background: var(--n-accent-soft); color: var(--n-accent); }}

/* Case header */
.n-case {{ background: var(--n-surface); border: 1px solid var(--n-border); border-radius: 12px; padding: 1.5rem 1.75rem; }}
.n-eyebrow {{ font-size: .75rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--n-accent); }}
.n-case-title {{ font-family: var(--n-serif); font-size: 1.75rem; font-weight: 600; line-height: 1.25;
  color: var(--n-ink); margin: .4rem 0 .6rem; overflow-wrap: anywhere; }}
.n-case-meta {{ display: flex; flex-wrap: wrap; gap: .35rem 1.25rem; color: var(--n-ink2); font-size: .9rem; }}
.n-case-meta b {{ color: var(--n-ink); font-weight: 600; }}
.n-badges {{ display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1rem; }}

/* KPI row */
.n-kpis {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .85rem; }}
@media (max-width: 860px) {{ .n-kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.n-kpi {{ background: var(--n-surface); border: 1px solid var(--n-border); border-radius: 12px; padding: 1rem 1.15rem; }}
.n-kpi-label {{ font-size: .82rem; color: var(--n-ink2); font-weight: 500; }}
.n-kpi-value {{ font-size: 2rem; font-weight: 600; color: var(--n-ink); line-height: 1.2; margin: .3rem 0 .35rem; }}
.n-kpi-note {{ display: flex; align-items: center; gap: .35rem; font-size: .8rem; color: var(--n-ink2); }}

/* Cards, definition lists, tables */
.n-card {{ background: var(--n-surface); border: 1px solid var(--n-border); border-radius: 12px; padding: 1.25rem 1.5rem; }}
.n-fields {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.1rem 2rem; }}
@media (max-width: 700px) {{ .n-fields {{ grid-template-columns: 1fr; }} }}
.n-field-label {{ font-size: .72rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--n-muted); }}
.n-field-value {{ margin-top: .2rem; color: var(--n-ink); font-size: .95rem; overflow-wrap: anywhere; }}
.n-empty {{ color: var(--n-muted); }}
.n-issues {{ margin: 0; padding-left: 1.25rem; color: var(--n-ink); }}
.n-issues li {{ margin: .35rem 0; line-height: 1.55; }}
.n-holding {{ font-family: var(--n-serif); font-size: 1.08rem; line-height: 1.65; color: var(--n-ink);
  border-left: 3px solid var(--n-accent); padding: .1rem 0 .1rem 1rem; margin: 0; }}
.n-table-wrap {{ overflow-x: auto; border: 1px solid var(--n-border); border-radius: 12px; background: var(--n-surface); }}
.n-table {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
.n-table th {{ text-align: left; font-size: .72rem; font-weight: 600; letter-spacing: .07em; text-transform: uppercase;
  color: var(--n-muted); padding: .75rem 1rem; border-bottom: 1px solid var(--n-border); background: var(--n-sunken); }}
.n-table td {{ padding: .8rem 1rem; border-bottom: 1px solid var(--n-border); vertical-align: top; color: var(--n-ink); }}
.n-table tr:last-child td {{ border-bottom: none; }}
.n-table .n-num {{ font-family: var(--n-mono); font-size: .85rem; white-space: nowrap; }}
.n-sub {{ color: var(--n-ink2); font-size: .82rem; margin-top: .15rem; }}
.n-details summary {{ cursor: pointer; color: var(--n-accent); font-size: .8rem; font-weight: 500; margin-top: .35rem; }}
.n-details div {{ margin-top: .45rem; padding: .6rem .75rem; background: var(--n-sunken); border-radius: 8px;
  color: var(--n-ink2); font-size: .82rem; line-height: 1.55; white-space: pre-wrap; }}
.n-legend {{ display: flex; flex-wrap: wrap; gap: .5rem 1rem; align-items: center; margin-top: .75rem;
  color: var(--n-ink2); font-size: .8rem; }}

/* Confidence meters */
.n-meter-row {{ display: grid; grid-template-columns: 150px minmax(0, 1fr) 44px 96px; align-items: center;
  gap: 1rem; padding: .6rem 0; border-bottom: 1px solid var(--n-border); }}
.n-meter-row:last-child {{ border-bottom: none; }}
@media (max-width: 640px) {{ .n-meter-row {{ grid-template-columns: 110px minmax(0, 1fr) 40px; }} .n-meter-status {{ display: none; }} }}
.n-meter-label {{ font-size: .9rem; color: var(--n-ink); }}
.n-meter {{ position: relative; height: 8px; border-radius: 4px; }}
.n-meter-fill {{ height: 100%; border-radius: 4px; }}
.n-meter-threshold {{ position: absolute; top: -4px; bottom: -4px; width: 2px; background: var(--n-muted); border-radius: 1px; }}
.n-meter-value {{ font-size: .88rem; font-weight: 600; color: var(--n-ink); text-align: right; font-variant-numeric: tabular-nums; }}
.n-meter-status {{ display: flex; align-items: center; gap: .35rem; font-size: .8rem; color: var(--n-ink2); }}

/* Sidebar status */
.n-status-row {{ display: flex; justify-content: space-between; align-items: center; gap: .5rem;
  padding: .45rem 0; border-bottom: 1px solid var(--n-border); font-size: .86rem; }}
.n-status-row:last-child {{ border-bottom: none; }}
.n-status-row span:first-child {{ color: var(--n-ink2); }}
.n-status-val {{ display: inline-flex; align-items: center; gap: .35rem; color: var(--n-ink); font-weight: 500; }}

/* How it works */
.n-steps {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .85rem; }}
@media (max-width: 860px) {{ .n-steps {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.n-step {{ border: 1px dashed var(--n-border); border-radius: 12px; padding: 1rem 1.1rem; }}
.n-step-num {{ width: 26px; height: 26px; border-radius: 50%; background: var(--n-accent-soft); color: var(--n-accent);
  display: grid; place-items: center; font-size: .8rem; font-weight: 600; }}
.n-step-title {{ font-weight: 600; color: var(--n-ink); margin: .6rem 0 .25rem; font-size: .95rem; }}
.n-step-text {{ color: var(--n-ink2); font-size: .84rem; line-height: 1.5; }}

.n-footer {{ margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid var(--n-border);
  color: var(--n-muted); font-size: .8rem; line-height: 1.5; }}
</style>""")


# ── Small HTML helpers ────────────────────────────────────────────────────────

def esc(value) -> str:
    return html.escape("" if value is None else str(value))


_ICON_PATHS = {
    "check": "<circle cx='12' cy='12' r='9'/><path d='m8 12.5 2.7 2.7L16 9.8'/>",
    "alert": "<path d='M12 3.8 2.8 19.5h18.4z'/><path d='M12 10v4.2'/><path d='M12 17h.01'/>",
    "cross": "<circle cx='12' cy='12' r='9'/><path d='m9 9 6 6M15 9l-6 6'/>",
    "dot":   "<circle cx='12' cy='12' r='5' fill='black'/>",
    "info":  "<circle cx='12' cy='12' r='9'/><path d='M12 11v5'/><path d='M12 8h.01'/>",
    "logo":  ("<path d='M12 3v18'/><path d='M7 21h10'/><path d='M4 7h16'/>"
              "<path d='M12 5.5 4 7l-2.5 6a3.2 3.2 0 0 0 5 0z'/><path d='M12 5.5 20 7l2.5 6a3.2 3.2 0 0 1-5 0z'/>"),
}
_TONE_ICON  = {"good": "check", "warn": "alert", "crit": "cross", "neutral": "info", "accent": "dot"}
_TONE_COLOR = {"good": "var(--n-good)", "warn": "var(--n-warn)", "crit": "var(--n-crit)",
               "neutral": "var(--n-muted)", "accent": "var(--n-accent)"}


def _icon_css() -> str:
    """Icons are CSS masks over currentColor, so they follow the theme and survive st.html's sanitiser."""
    rules = []
    for name, paths in _ICON_PATHS.items():
        svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' "
               f"stroke-width='{1.8 if name == 'logo' else 2}' stroke-linecap='round' stroke-linejoin='round'>{paths}</svg>")
        rules.append(f'.n-ico-{name} {{ --n-mask: url("data:image/svg+xml,{quote(svg)}"); }}')
    return "\n".join(rules)


def icon(name: str, size: int = 14, color: str = "currentColor") -> str:
    return (f'<span class="n-ico n-ico-{name}" aria-hidden="true" '
            f'style="width:{size}px;height:{size}px;color:{color}"></span>')


def badge(label: str, tone: str, title: str = "") -> str:
    tip = f' title="{esc(title)}"' if title else ""
    return f'<span class="n-badge n-{tone}"{tip}>{icon(_TONE_ICON[tone], 13)}{esc(label)}</span>'


def status_note(text: str, tone: str) -> str:
    """Neutral-ink text with a coloured status icon (text never wears the status colour)."""
    return f'{icon(_TONE_ICON[tone], 14, _TONE_COLOR[tone])}<span>{esc(text)}</span>'


# ── API ───────────────────────────────────────────────────────────────────────

def fetch_health() -> dict | None:
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def _sse_events(response: httpx.Response):
    response.raise_for_status()
    for line in response.iter_lines():
        if line.startswith("data: ") and line[6:].strip():
            try:
                yield json.loads(line[6:])
            except json.JSONDecodeError:
                continue


def stream_text(judgment_text: str):
    with httpx.stream("POST", f"{API_BASE}/analyze/text", json={"judgment_text": judgment_text, "stream": True},
                      timeout=900.0) as resp:
        yield from _sse_events(resp)


def stream_file(file_bytes: bytes, filename: str):
    with httpx.stream("POST", f"{API_BASE}/analyze", timeout=900.0,
                      files={"file": (filename, file_bytes, "application/octet-stream")}) as resp:
        yield from _sse_events(resp)


def analyze_blocking(judgment_text: str) -> dict:
    r = httpx.post(f"{API_BASE}/analyze/text", json={"judgment_text": judgment_text, "stream": False}, timeout=900.0)
    r.raise_for_status()
    return r.json()


# ── Pipeline step descriptions ────────────────────────────────────────────────

STEP_NAMES = {
    "extraction":          "Extraction",
    "statute_validation":  "Statute validation",
    "citation_resolution": "Precedent resolution",
    "confidence_scoring":  "Confidence scoring",
    "assembly":            "Assembly",
}


def describe_step(entry: dict) -> tuple[str, str]:
    """Return (tone, detail) for one audit-trail entry."""
    agent, status = entry.get("agent"), entry.get("status")
    if agent == "extraction":
        if status == "success":
            return "good", f"{entry.get('field_count', 0)} fields extracted by Nyaya-7B in {entry.get('latency', 0):.1f} s"
        if status == "json_parse_failed":
            return "crit", "The model output was not valid JSON"
        if status == "skipped":
            return "crit", "Judgment text is too short"
        return "crit", entry.get("error") or "Extraction failed"
    if agent == "statute_validation":
        if status == "no_statutes_to_validate":
            return "good", "No statutes to validate"
        flagged = entry.get("hallucinated_count", 0)
        detail = (f"{entry.get('grounded_count', 0)} of {entry.get('total_statutes', 0)} found in the judgment, "
                  f"{entry.get('in_corpus_count', 0)} matched in the statute corpus")
        return ("warn", f"{detail}, {flagged} flagged") if flagged else ("good", detail)
    if agent == "citation_resolution":
        return "good", (f"{entry.get('total_citations', 0)} citations, "
                        f"{entry.get('resolved_count', 0)} resolved in the local index")
    if agent == "confidence_scoring":
        flagged = entry.get("flagged_count", 0)
        detail = f"Overall {entry.get('overall_confidence', 0):.0%}, {flagged} field{'s' if flagged != 1 else ''} below {CONFIDENCE_THRESHOLD:.0%}"
        return ("warn" if flagged else "good"), detail
    if agent == "assembly":
        return ("good", "Final output assembled") if status == "complete" else ("crit", "No output — extraction failed")
    return "neutral", str(status or "")


_MD_TONE = {"good": ":green[:material/check_circle:]", "warn": ":orange[:material/error:]",
            "crit": ":red[:material/cancel:]", "neutral": ":gray[:material/info:]"}


def step_markdown(entry: dict) -> str:
    tone, detail = describe_step(entry)
    detail = detail.replace("*", "\\*").replace("_", "\\_").replace("`", "'")
    return f"{_MD_TONE[tone]} **{STEP_NAMES.get(entry.get('agent'), entry.get('agent'))}** — {detail}"


# ── Sample judgment ───────────────────────────────────────────────────────────

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


def _load_sample() -> None:
    st.session_state["textarea_judgment"] = SAMPLE


def _clear_text() -> None:
    st.session_state["textarea_judgment"] = ""


def _new_analysis() -> None:
    st.session_state["result"] = None
    st.session_state["textarea_judgment"] = ""


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _status_row(label: str, value: str, tone: str) -> str:
    return (f'<div class="n-status-row"><span>{esc(label)}</span>'
            f'<span class="n-status-val">{icon(_TONE_ICON[tone], 13, _TONE_COLOR[tone])}{esc(value)}</span></div>')


def render_system_status(polling: bool) -> None:
    health = fetch_health()
    if polling and health and (health.get("model_loaded") or health.get("model_error")):
        st.rerun()   # backend finished starting — full rerun stops the polling
    if health is None:
        rows = _status_row("API", "Offline", "crit")
    else:
        if health.get("model_loaded"):
            model = ("Ready", "good")
        elif health.get("model_error"):
            model = ("Failed", "crit")
        else:
            model = ("Loading…", "warn")
        statutes, precedents = health.get("statute_index_size", 0), health.get("precedent_index_size", 0)
        rows = (_status_row("API", "Online", "good")
                + _status_row("Nyaya-7B", *model)
                + _status_row("Statute index", f"{statutes:,} chunks" if statutes else "Loading…", "good" if statutes else "warn")
                + _status_row("Precedent index", f"{precedents:,} cases" if precedents else "Loading…", "good" if precedents else "warn"))
    st.html(f'<div>{rows}</div>')

    if health is None:
        st.caption("Start the backend from the `backend` folder with `run.bat`.")
    elif health.get("model_error"):
        st.error(health["model_error"], icon=":material/error:")
    elif not health.get("model_loaded"):
        st.caption("The model loads onto the GPU in about 30–40 seconds. This panel refreshes on its own.")
    st.button("Refresh", icon=":material/refresh:", type="tertiary", key="btn_refresh")


def render_sidebar() -> None:
    with st.sidebar:
        st.html('<div class="n-label">System status</div>')
        # Poll while the backend is still starting, but never during an analysis run.
        health = fetch_health()
        ready = bool(health and health.get("model_loaded"))
        polling = not ready and not st.session_state.get("btn_analyze") and not (health or {}).get("model_error")
        st.fragment(render_system_status, run_every="4s" if polling else None)(polling)

        st.divider()
        st.html('<div class="n-label">Settings</div>')
        st.toggle("Stream pipeline progress", key="stream_mode",
                  help="Show each agent as it finishes. Turn off to wait for the complete result.")

        st.divider()
        st.html('<div class="n-label">About</div>')
        st.caption(
            "Nyaya-7B is Mistral-7B finetuned with QLoRA on Indian Supreme Court and High Court judgments. "
            "It runs on this machine's GPU; judgments are not sent to any external service."
        )


# ── Header and input ──────────────────────────────────────────────────────────

def render_topbar() -> None:
    st.html(f"""
<div class="n-topbar">
  <div class="n-brand">
    <div class="n-logo">{icon("logo", 24, "#ffffff")}</div>
    <div>
      <div class="n-title">Nyaya</div>
      <div class="n-subtitle">Structured analysis of Indian court judgments</div>
    </div>
  </div>
  <span class="n-chip">{icon("dot", 12, "var(--n-good)")}Runs locally · Nyaya-7B</span>
</div>""")


def render_input(collapsed: bool):
    """Render the input card (collapsed into an expander once results are shown).
    Returns (kind, payload, label) or None when there is no input."""
    box = (st.expander("Judgment input", icon=":material/description:", expanded=False) if collapsed
           else st.container(border=True))
    with box:
        if not collapsed:
            st.html('<div class="n-label">Judgment</div>')
        tab_text, tab_file = st.tabs([":material/edit_note: Paste text", ":material/upload_file: Upload file"])

        with tab_text:
            text = st.text_area(
                "Judgment text", key="textarea_judgment", height=260, label_visibility="collapsed",
                placeholder="Paste the full text of a judgment — e.g. IN THE SUPREME COURT OF INDIA …",
            )
            with st.container(horizontal=True, vertical_alignment="center"):
                st.caption(f"{len(text):,} characters" if text else "Minimum 200 characters.")
                st.space("stretch")
                st.button("Load sample", icon=":material/description:", type="tertiary",
                          on_click=_load_sample, key="btn_sample")
                st.button("Clear", icon=":material/close:", type="tertiary",
                          on_click=_clear_text, key="btn_clear_text", disabled=not text)

        with tab_file:
            uploaded = st.file_uploader("Judgment file", type=["pdf", "txt"], label_visibility="collapsed",
                                        key="file_uploader_judgment",
                                        help="PDF or plain text, up to 10 MB. Scanned PDFs without a text layer are not supported.")

    if uploaded is not None:
        return "file", uploaded, f"{uploaded.name} ({uploaded.size / 1024:,.0f} KB)"
    if text.strip():
        return "text", text, f"pasted text ({len(text):,} characters)"
    return None


# ── Running an analysis ───────────────────────────────────────────────────────

def run_analysis(source) -> None:
    kind, payload, label = source
    if kind == "text" and len(payload.strip()) < 200:
        st.warning("The judgment text is too short — paste at least 200 characters.", icon=":material/warning:")
        return
    if fetch_health() is None:
        st.error(f"Cannot reach the Nyaya API at {API_BASE}. Start the backend with `run.bat`.", icon=":material/cloud_off:")
        return

    audit, errors, final_output = [], [], None
    t0 = time.time()
    with st.status("Extracting structured data with Nyaya-7B…", expanded=True) as status:
        st.caption(f"Source: {label}. Extraction takes about a minute on a laptop GPU.")
        try:
            if kind == "file" or st.session_state["stream_mode"]:
                events = stream_file(payload.getvalue(), payload.name) if kind == "file" else stream_text(payload)
                for evt in events:
                    if evt.get("event") == "error":
                        errors.extend(evt.get("errors") or [evt.get("error", "Unknown error")])
                        continue
                    if not evt.get("node"):
                        continue
                    errors.extend(evt.get("errors") or [])
                    for entry in evt.get("audit_trail") or []:
                        audit.append(entry)
                        st.markdown(step_markdown(entry))
                    if evt["node"] == "extract":
                        status.update(label="Validating statutes and resolving precedents…")
                    if evt.get("final_output"):
                        final_output = evt["final_output"]
            else:
                data = analyze_blocking(payload)
                final_output, audit, errors = data.get("final_output"), data.get("audit_trail") or [], data.get("errors") or []
                for entry in audit:
                    st.markdown(step_markdown(entry))
        except httpx.HTTPStatusError as e:
            errors.append(f"API error {e.response.status_code}: {e.response.text[:300]}")
        except httpx.HTTPError as e:
            errors.append(f"Connection to the API failed: {e}")

        elapsed = time.time() - t0
        if final_output:
            status.update(label=f"Analysis complete in {elapsed:.0f} s", state="complete", expanded=False)
        else:
            status.update(label="Analysis failed", state="error", expanded=True)

    st.session_state["result"] = {
        "final_output": final_output,
        "audit": audit,
        "errors": list(dict.fromkeys(errors)),
        "elapsed": elapsed,
        "source": label,
    }
    st.rerun()   # redraw with the input collapsed and the results in place (steps stay in the Audit trail tab)


# ── Results ───────────────────────────────────────────────────────────────────

STATUTE_STATUS = {
    "verified":        ("In judgment + corpus", "good", "Mentioned in the judgment; statutory text found in the local corpus"),
    "grounded":        ("In judgment", "good", "Mentioned in the judgment; the local corpus does not cover this section"),
    "incomplete":      ("No section given", "warn", "The model named the act but not a section, so it cannot be checked"),
    "not_in_judgment": ("Not in judgment", "crit", "The judgment never mentions this section — possible hallucination"),
}

FIELD_LABELS = {
    "case_name": "Case name", "court": "Court", "year": "Year", "parties": "Parties", "outcome": "Outcome",
    "holding": "Holding", "statutes_cited": "Statutes", "precedents_cited": "Precedents",
    "legal_issues": "Legal issues", "citation_format": "Citation format",
}


def _confidence_level(score: float) -> tuple[str, str]:
    if score >= 0.85:
        return "High", "good"
    if score >= CONFIDENCE_THRESHOLD:
        return "Moderate", "warn"
    return "Review", "crit"


def _section_ref(statute: dict) -> str:
    prefix = "Art." if "constitution" in (statute.get("act") or "").lower() else "§"
    return f"{prefix} {statute.get('section') or '—'}"


def case_header_html(out: dict) -> str:
    eyebrow = " · ".join(esc(x) for x in (out.get("subject_matter"), out.get("court")) if x) or "Judgment"
    meta = []
    meta.append(f"<span><b>Citation</b> {esc(out.get('citation'))}</span>" if out.get("citation")
                else '<span class="n-empty">No reported citation</span>')
    if out.get("year"):
        meta.append(f"<span><b>Year</b> {esc(out['year'])}</span>")
    if out.get("bench"):
        meta.append(f"<span><b>Bench</b> {esc(', '.join(out['bench']))}</span>")

    badges = []
    if out.get("outcome"):
        badges.append(f'<span class="n-badge n-accent">Outcome · {esc(out["outcome"].capitalize())}</span>')
    uncertain, hallucinated = out.get("uncertain_fields") or [], out.get("hallucinated_statutes") or []
    if out.get("needs_human_review"):
        reasons = []
        if uncertain:
            reasons.append(f"{len(uncertain)} low-confidence field{'s' if len(uncertain) != 1 else ''}")
        if hallucinated:
            reasons.append(f"{len(hallucinated)} flagged statute{'s' if len(hallucinated) != 1 else ''}")
        badges.append(badge("Needs human review", "warn", "; ".join(reasons)))
    else:
        badges.append(badge("All checks passed", "good"))

    return f"""
<div class="n-case">
  <div class="n-eyebrow">{eyebrow}</div>
  <div class="n-case-title">{esc(out.get('case_name') or 'Untitled judgment')}</div>
  <div class="n-case-meta">{''.join(meta)}</div>
  <div class="n-badges">{''.join(badges)}</div>
</div>"""


def kpi_row_html(out: dict) -> str:
    statutes = out.get("statutes_cited") or []
    precedents = out.get("precedents_cited") or []
    hallucinated = out.get("hallucinated_statutes") or []
    uncertain = out.get("uncertain_fields") or []
    overall = out.get("overall_confidence") or 0.0

    grounded = sum(1 for s in statutes if s.get("verified"))
    resolved = sum(1 for p in precedents if p.get("resolved"))

    def tile(label, value, note_text, tone):
        return (f'<div class="n-kpi"><div class="n-kpi-label">{esc(label)}</div>'
                f'<div class="n-kpi-value">{esc(value)}</div>'
                f'<div class="n-kpi-note">{status_note(note_text, tone)}</div></div>')

    return '<div class="n-kpis">' + "".join([
        tile("Overall confidence", f"{overall:.0%}",
             f"{len(uncertain)} field{'s' if len(uncertain) != 1 else ''} to review" if uncertain else "No fields flagged",
             "warn" if uncertain else "good"),
        tile("Statutes cited", len(statutes),
             f"{grounded} of {len(statutes)} found in judgment" if statutes else "None extracted",
             "good" if grounded == len(statutes) else "warn"),
        tile("Precedents cited", len(precedents),
             f"{resolved} resolved in local index" if precedents else "None cited",
             "good" if precedents and resolved == len(precedents) else "neutral"),
        tile("Possible hallucinations", len(hallucinated),
             "Review flagged statutes" if hallucinated else "None detected",
             "crit" if hallucinated else "good"),
    ]) + "</div>"


def overview_html(out: dict) -> str:
    def field(label, value):
        if isinstance(value, list):
            value = ", ".join(value)
        shown = esc(value) if value not in (None, "", []) else '<span class="n-empty">Not found</span>'
        return f'<div><div class="n-field-label">{esc(label)}</div><div class="n-field-value">{shown}</div></div>'

    details = "".join([
        field("Petitioner / appellant", out.get("petitioner")),
        field("Respondent", out.get("respondent")),
        field("Court", out.get("court")),
        field("Year", out.get("year")),
        field("Subject matter", out.get("subject_matter")),
        field("Outcome", (out.get("outcome") or "").capitalize()),
        field("Citation", out.get("citation")),
        field("Bench", out.get("bench")),
    ])
    issues = out.get("legal_issues") or []
    issues_html = ("<ol class='n-issues'>" + "".join(f"<li>{esc(i)}</li>" for i in issues) + "</ol>"
                   if issues else '<div class="n-empty">No legal issues extracted.</div>')
    holding = out.get("holding")
    holding_html = (f'<blockquote class="n-holding">{esc(holding)}</blockquote>'
                    if holding else '<div class="n-empty">No holding extracted.</div>')

    return f"""
<div class="n-card"><div class="n-label">Holding</div>{holding_html}</div>
<div style="height:.85rem"></div>
<div class="n-card"><div class="n-label">Case details</div><div class="n-fields">{details}</div></div>
<div style="height:.85rem"></div>
<div class="n-card"><div class="n-label">Legal issues</div>{issues_html}</div>"""


def statutes_html(out: dict) -> str:
    statutes = out.get("statutes_cited") or []
    if not statutes:
        return '<div class="n-card n-empty">No statutes were extracted from this judgment.</div>'
    rows = []
    for s in statutes:
        label, tone, tip = STATUTE_STATUS.get(s.get("status"), ("Checked" if s.get("verified") else "Flagged",
                                                                 "good" if s.get("verified") else "crit", ""))
        text = s.get("actual_text")
        details = (f'<details class="n-details"><summary>Statutory text</summary><div>{esc(text)}</div></details>'
                   if text else "")
        rows.append(
            f"<tr><td>{esc(s.get('act') or 'Unknown act')}</td>"
            f"<td class='n-num'>{esc(_section_ref(s))}</td>"
            f"<td>{esc(s.get('description')) or '<span class=n-empty>—</span>'}{details}</td>"
            f"<td>{badge(label, tone, tip)}</td></tr>"
        )
    legend = ("<div class='n-legend'>"
              + badge("In judgment + corpus", "good") + badge("In judgment", "good")
              + badge("Not in judgment", "crit")
              + "<span>A statute is flagged when the judgment never mentions it. "
                "Corpus text is attached where the local statute corpus covers the section.</span></div>")
    return (f"<div class='n-table-wrap'><table class='n-table'><thead><tr><th>Act</th><th>Section</th>"
            f"<th>Description</th><th>Validation</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>{legend}")


def precedents_html(out: dict) -> str:
    precedents = out.get("precedents_cited") or []
    if not precedents:
        return '<div class="n-card n-empty">No precedents are cited in this judgment.</div>'
    rows = []
    for p in precedents:
        if p.get("resolved"):
            status = badge("Resolved", "good", f"Matched via {p.get('source') or 'local index'}")
        elif p.get("grounded"):
            status = badge("In judgment", "neutral", "Cited in the judgment; not in the local precedent index")
        else:
            status = badge("Not in judgment", "crit", "This citation does not appear in the judgment text")
        court_year = " · ".join(esc(x) for x in (p.get("court"), p.get("year") or None) if x)
        summary = p.get("summary")
        details = (f'<details class="n-details"><summary>Summary</summary><div>{esc(summary)}</div></details>'
                   if summary else "")
        rows.append(
            f"<tr><td class='n-num'>{esc(p.get('citation')) or '<span class=n-empty>—</span>'}</td>"
            f"<td>{esc(p.get('case_name')) or '<span class=n-empty>Unknown</span>'}"
            f"{f'<div class=n-sub>{court_year}</div>' if court_year else ''}{details}</td>"
            f"<td>{status}</td></tr>"
        )
    return (f"<div class='n-table-wrap'><table class='n-table'><thead><tr><th>Citation</th><th>Case</th>"
            f"<th>Status</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
            f"<div class='n-legend'><span>Citations are matched against the local precedent index; "
            f"those not in the index are still checked against the judgment text.</span></div>")


def confidence_html(out: dict) -> str:
    scores = out.get("confidence_scores") or {}
    if not scores:
        return '<div class="n-card n-empty">No confidence scores available.</div>'
    rows = []
    for field, score in scores.items():
        level, tone = _confidence_level(score)
        name = FIELD_LABELS.get(field, field.replace("_", " ").capitalize())
        rows.append(f"""
<div class="n-meter-row" title="{esc(name)}: {score:.0%} ({level})">
  <div class="n-meter-label">{esc(name)}</div>
  <div class="n-meter" style="background:var(--n-{tone}-soft)">
    <div class="n-meter-fill" style="width:{max(score, 0.02) * 100:.0f}%;background:var(--n-{tone})"></div>
    <div class="n-meter-threshold" style="left:{CONFIDENCE_THRESHOLD * 100:.0f}%"></div>
  </div>
  <div class="n-meter-value">{score:.0%}</div>
  <div class="n-meter-status">{status_note(level, tone)}</div>
</div>""")
    legend = ("<div class='n-legend'>"
              + badge("High ≥ 85%", "good") + badge("Moderate 70–84%", "warn") + badge("Review < 70%", "crit")
              + f"<span>The vertical mark is the {CONFIDENCE_THRESHOLD:.0%} review threshold.</span></div>")
    return f"<div class='n-card'>{''.join(rows)}</div>{legend}"


def audit_html(audit: list) -> str:
    if not audit:
        return '<div class="n-card n-empty">No audit trail recorded.</div>'
    rows = []
    for entry in audit:
        tone, detail = describe_step(entry)
        latency = entry.get("latency")
        rows.append(
            f"<tr><td>{esc(STEP_NAMES.get(entry.get('agent'), entry.get('agent')))}</td>"
            f"<td>{badge((entry.get('status') or '').replace('_', ' ').capitalize(), tone)}</td>"
            f"<td>{esc(detail)}</td>"
            f"<td class='n-num'>{f'{latency:.2f} s' if isinstance(latency, (int, float)) else '—'}</td></tr>"
        )
    return (f"<div class='n-table-wrap'><table class='n-table'><thead><tr><th>Agent</th><th>Status</th>"
            f"<th>Detail</th><th>Time</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


def text_summary(out: dict) -> str:
    lines = [
        "NYAYA — JUDGMENT ANALYSIS",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"Case:        {out.get('case_name', '')}",
        f"Court:       {out.get('court', '')}",
        f"Year:        {out.get('year') or ''}",
        f"Citation:    {out.get('citation') or '—'}",
        f"Outcome:     {out.get('outcome', '')}",
        f"Confidence:  {(out.get('overall_confidence') or 0):.0%}"
        + ("  (needs human review)" if out.get("needs_human_review") else ""),
        "",
        "HOLDING",
        out.get("holding", ""),
        "",
        "STATUTES CITED",
    ]
    for s in out.get("statutes_cited") or []:
        lines.append(f"  - {s.get('act', '')} {_section_ref(s)}  [{s.get('status') or ('verified' if s.get('verified') else 'flagged')}]")
    lines += ["", "PRECEDENTS CITED"]
    for p in out.get("precedents_cited") or []:
        lines.append(f"  - {p.get('citation', '')}  {p.get('case_name', '')}")
    lines += ["", "Machine-generated. Verify against the original judgment before relying on it."]
    return "\n".join(lines)


def render_results(result: dict) -> None:
    out = result.get("final_output")
    if not out:
        st.error("The pipeline could not produce a result for this judgment.", icon=":material/error:")
        for err in result.get("errors") or []:
            st.code(err, language=None, wrap_lines=True)
        st.button("Start over", icon=":material/restart_alt:", on_click=_new_analysis, key="btn_restart_failed")
        return

    for err in result.get("errors") or []:
        st.warning(err, icon=":material/warning:")

    st.html(case_header_html(out))
    st.html(kpi_row_html(out))

    n_statutes = len(out.get("statutes_cited") or [])
    n_precedents = len(out.get("precedents_cited") or [])
    tabs = st.tabs([
        ":material/article: Overview",
        f":material/menu_book: Statutes ({n_statutes})",
        f":material/account_balance: Precedents ({n_precedents})",
        ":material/speed: Confidence",
        ":material/timeline: Audit trail",
        ":material/data_object: JSON",
    ])
    with tabs[0]:
        st.html(overview_html(out))
    with tabs[1]:
        st.html(statutes_html(out))
    with tabs[2]:
        st.html(precedents_html(out))
    with tabs[3]:
        st.html(confidence_html(out))
    with tabs[4]:
        st.caption(f"Source: {result.get('source', '—')} · total time {result.get('elapsed', 0):.1f} s")
        st.html(audit_html(result.get("audit") or []))
    with tabs[5]:
        st.json(out, expanded=1)

    stamp = int(time.time())
    with st.container(horizontal=True, vertical_alignment="center"):
        st.download_button("Download JSON", data=json.dumps(out, indent=2, ensure_ascii=False),
                           file_name=f"nyaya_analysis_{stamp}.json", mime="application/json",
                           icon=":material/download:", key="dl_json")
        st.download_button("Download summary", data=text_summary(out), file_name=f"nyaya_summary_{stamp}.txt",
                           mime="text/plain", icon=":material/summarize:", key="dl_txt")
        st.button("New analysis", icon=":material/add:", type="tertiary", on_click=_new_analysis, key="btn_new")


def render_how_it_works() -> None:
    steps = [
        ("Extract", "Nyaya-7B reads the judgment on your GPU and returns parties, court, statutes, precedents and the holding as JSON."),
        ("Validate statutes", "Every cited section must appear in the judgment. Statutory text is attached from the local corpus."),
        ("Resolve precedents", "Citations are matched against the local precedent index and checked against the judgment text."),
        ("Score confidence", f"Each field is scored; anything under {CONFIDENCE_THRESHOLD:.0%} is flagged for human review."),
    ]
    cards = "".join(
        f'<div class="n-step"><div class="n-step-num">{i}</div><div class="n-step-title">{esc(title)}</div>'
        f'<div class="n-step-text">{esc(text)}</div></div>'
        for i, (title, text) in enumerate(steps, 1)
    )
    st.html(f'<div class="n-label" style="margin-top:.5rem">How it works</div><div class="n-steps">{cards}</div>')


# ── Page ──────────────────────────────────────────────────────────────────────

inject_css()
render_sidebar()
render_topbar()
st.space("small")

source = render_input(collapsed=bool(st.session_state.get("result")))
with st.container(horizontal=True, vertical_alignment="center", gap="medium"):
    analyze_clicked = st.button("Analyze judgment", type="primary", icon=":material/gavel:",
                                key="btn_analyze", disabled=source is None)
    st.caption(f"Will analyze: {source[2]}" if source else "Paste a judgment, upload a file, or load the sample to begin.")

if analyze_clicked and source:
    run_analysis(source)

result = st.session_state.get("result")
if result:
    st.space("small")
    render_results(result)
else:
    st.space("small")
    render_how_it_works()

st.html('<div class="n-footer">Nyaya produces machine-generated analysis. Verify every field against the original '
        'judgment before relying on it for legal work.</div>')
