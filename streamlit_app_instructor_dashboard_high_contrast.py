from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_OK = True
except Exception:
    AUTOREFRESH_OK = False


DEFAULT_DB = "attention_events.jsonl"
DEFAULT_TABLE = "attention_events"


# -----------------------------
# Data loading
# -----------------------------
def _load_sqlite(db_path: Path, table_name: str) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()

    with sqlite3.connect(str(db_path)) as conn:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
            conn,
        )
        if tables.empty:
            return pd.DataFrame()

        if table_name not in set(tables["name"].astype(str)):
            table_name = str(tables.iloc[0]["name"])

        df = pd.read_sql_query(f"SELECT * FROM {table_name} ORDER BY id ASC", conn)
    return df


def _load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def _load_jsonl(jsonl_path: Path) -> pd.DataFrame:
    if not jsonl_path.exists():
        return pd.DataFrame()

    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return pd.DataFrame(rows)


@st.cache_data(ttl=2)
def load_events(data_path: str, table_name: str) -> pd.DataFrame:
    path = Path(data_path)
    suffix = path.suffix.lower()

    if suffix in {".db", ".sqlite", ".sqlite3"}:
        df = _load_sqlite(path, table_name)
    elif suffix == ".csv":
        df = _load_csv(path)
    elif suffix in {".jsonl", ".ndjson"}:
        df = _load_jsonl(path)
    else:
        df = _load_sqlite(path, table_name)
    return df


# -----------------------------
# Event normalization + derived metrics
# -----------------------------
def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols = {c for c in columns}
    for c in candidates:
        if c in cols:
            return c
    return None


def _risk_label(v: float) -> str:
    if not np.isfinite(v):
        return "Unknown"
    if v <= 30:
        return "Low"
    if v <= 60:
        return "Moderate"
    return "High"


def _derive_cause(df: pd.DataFrame) -> pd.Series:
    cause = pd.Series(["unknown"] * len(df), index=df.index, dtype="object")

    eyes = (df["eye_closed"] > 0) | (df["eye_drowsy"] > 0)
    yawning = df["yawning"] > 0
    head = df["head_forward"] <= 0
    motion = df["body_stable"] <= 0

    cause.loc[motion] = "fidgeting/motion"
    cause.loc[head] = "looking_away/head_pose"
    cause.loc[yawning] = "yawning"
    cause.loc[eyes] = "drowsy/eyes"
    return cause


def _derive_issue(df: pd.DataFrame) -> pd.Series:
    issue = pd.Series(["none"] * len(df), index=df.index, dtype="object")
    rule_bad = (df["rule_ready"] > 0) & (df["att_score"] <= 1)
    drift_bad = (df["drift_ready"] > 0) & (df["drift_state"].astype(str).str.upper() == "DRIFT")
    issue.loc[rule_bad] = "rule"
    issue.loc[drift_bad] = "drift"
    return issue


def _derive_severity(df: pd.DataFrame) -> pd.Series:
    sev = pd.Series([0] * len(df), index=df.index, dtype="int64")

    drift_mask = df["issue_final"] == "drift"
    rule_mask = df["issue_final"] == "rule"
    p = pd.to_numeric(df["drift_prob"], errors="coerce")

    sev.loc[rule_mask & (df["att_score"] == 1)] = 2
    sev.loc[rule_mask & (df["att_score"] <= 0)] = 3

    sev.loc[drift_mask & (p >= 0.55)] = 2
    sev.loc[drift_mask & (p >= 0.75)] = 3
    sev.loc[drift_mask & (p < 0.55)] = 1
    return sev


def normalize_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    time_col = _first_existing(
        df.columns,
        ["event_time", "timestamp", "created_at", "ts", "datetime"],
    )
    if time_col is not None:
        df["event_dt"] = pd.to_datetime(df[time_col], errors="coerce")
    else:
        df["event_dt"] = pd.NaT

    if "student_id" not in df.columns:
        df["student_id"] = "Student A"
    df["student_id"] = df["student_id"].fillna("Student A").astype(str)

    if "session_id" not in df.columns:
        if df["event_dt"].notna().any():
            df["session_id"] = df["event_dt"].dt.strftime("session_%Y%m%d")
        else:
            df["session_id"] = "session_001"
    df["session_id"] = df["session_id"].fillna("session_001").astype(str)

    defaults = {
        "t": np.nan,
        "rule_ready": 0,
        "att_score": np.nan,
        "eye_closed": 0,
        "eye_drowsy": 0,
        "eye_open": 0,
        "yawning": 0,
        "head_forward": 0,
        "body_stable": 0,
        "face_ok": 0,
        "pose_ok": 0,
        "drift_ready": 0,
        "drift_prob": np.nan,
        "drift_state": "N/A",
        "alert_emitted": 0,
        "gaze_score": np.nan,
        "head_score": np.nan,
        "body_score": np.nan,
        "ari_raw": np.nan,
        "ari_ma": np.nan,
        "risk_level": None,
        "issue": None,
        "cause": None,
        "severity": np.nan,
        "banner": None,
        "message": None,
        "micro_task": None,
        "check_question": None,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    numeric_cols = [
        "t", "rule_ready", "att_score", "eye_closed", "eye_drowsy", "eye_open", "yawning",
        "head_forward", "body_stable", "face_ok", "pose_ok", "drift_ready", "drift_prob",
        "alert_emitted", "gaze_score", "head_score", "body_score", "ari_raw", "ari_ma", "severity",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["event_dt"].isna().all():
        base = pd.Timestamp.now().floor("S")
        if df["t"].notna().any():
            df["event_dt"] = base + pd.to_timedelta(df["t"].fillna(0), unit="s")
        else:
            df["event_dt"] = pd.date_range(base, periods=len(df), freq="S")

    df = df.sort_values(["student_id", "event_dt", "t"], kind="stable").reset_index(drop=True)

    ari_missing = ~np.isfinite(df["ari_raw"])

    component_mask = ari_missing & df[["gaze_score", "head_score", "body_score"]].notna().all(axis=1)
    if component_mask.any():
        df.loc[component_mask, "ari_raw"] = (
            0.4 * df.loc[component_mask, "gaze_score"]
            + 0.3 * df.loc[component_mask, "head_score"]
            + 0.3 * df.loc[component_mask, "body_score"]
        )

    fallback_mask = ~np.isfinite(df["ari_raw"]) & df["att_score"].notna()
    if fallback_mask.any():
        df.loc[fallback_mask, "ari_raw"] = np.clip((3.0 - df.loc[fallback_mask, "att_score"]) / 3.0 * 100.0, 0, 100)

    drift_mask = ~np.isfinite(df["ari_raw"]) & df["drift_prob"].notna()
    if drift_mask.any():
        df.loc[drift_mask, "ari_raw"] = np.clip(df.loc[drift_mask, "drift_prob"] * 100.0, 0, 100)

    ari_ma_missing = ~np.isfinite(df["ari_ma"]) & df["ari_raw"].notna()
    if ari_ma_missing.any():
        rolled = (
            df.groupby("student_id", sort=False)["ari_raw"]
            .transform(lambda s: s.rolling(window=5, min_periods=1).mean())
        )
        df.loc[ari_ma_missing, "ari_ma"] = rolled.loc[ari_ma_missing]

    df["risk_level"] = df["risk_level"].where(df["risk_level"].notna(), df["ari_ma"].map(_risk_label))
    df["attention_pct"] = np.clip(100.0 - df["ari_ma"], 0, 100)

    issue = _derive_issue(df)
    df["issue_final"] = df["issue"].fillna(issue)
    df["cause_final"] = df["cause"].fillna(_derive_cause(df))

    derived_sev = _derive_severity(df)
    df["severity_final"] = df["severity"].fillna(derived_sev).astype(int)

    df["alert_emitted"] = np.where(
        (df["alert_emitted"].fillna(0) > 0) | (df["issue_final"] != "none"),
        1,
        0,
    )

    return df


# -----------------------------
# UI helpers
# -----------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-bg-top: #081524;
            --app-bg-bottom: #123a63;
            --panel-bg: rgba(11, 31, 58, 0.92);
            --panel-border: rgba(255,255,255,0.16);
            --card-bg: rgba(255,255,255,0.06);
            --text-primary: #f7fbff;
            --text-secondary: #c7d7ee;
            --input-bg: rgba(255,255,255,0.10);
            --input-border: rgba(255,255,255,0.18);
        }
        html, body, [data-testid="stAppViewContainer"], .stApp {
            background: linear-gradient(180deg, var(--app-bg-top) 0%, var(--app-bg-bottom) 100%);
            color: var(--text-primary);
        }
        .stApp {
            color: var(--text-primary);
        }
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-primary) !important;
            letter-spacing: 0.01em;
        }
        p, label, .stMarkdown, .stMarkdown p, .stMarkdown li, .stCaption {
            color: var(--text-secondary) !important;
        }
        .stMarkdown strong {
            color: var(--text-primary) !important;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1.2rem;
            max-width: 1400px;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1d33 0%, #13304f 100%);
            border-right: 1px solid rgba(255,255,255,0.10);
        }
        [data-testid="stSidebar"] * {
            color: var(--text-primary) !important;
        }
        .panel {
            border: 1px solid var(--panel-border);
            border-radius: 16px;
            background: var(--panel-bg);
            color: var(--text-primary);
            padding: 14px 16px;
            box-shadow: 0 10px 24px rgba(0,0,0,0.22);
            margin-bottom: 12px;
        }
        .panel-title {
            color: var(--text-primary);
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.8rem;
        }
        .metric-card {
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 14px;
            padding: 12px 14px;
            background: var(--card-bg);
            color: var(--text-primary);
        }
        .student-card {
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 14px;
            padding: 10px 12px;
            background: var(--card-bg);
            color: var(--text-primary);
            min-height: 102px;
        }
        .risk-badge-low, .risk-badge-moderate, .risk-badge-high, .risk-badge-unknown {
            display: inline-block;
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 0.78rem;
            font-weight: 700;
            margin-top: 6px;
            letter-spacing: 0.01em;
        }
        .risk-badge-low { background: rgba(37, 181, 78, 0.24); color: #b8f3b7; }
        .risk-badge-moderate { background: rgba(244, 191, 62, 0.26); color: #ffe08a; }
        .risk-badge-high { background: rgba(234, 90, 90, 0.28); color: #ffb1b1; }
        .risk-badge-unknown { background: rgba(170,170,170,0.22); color: #ececec; }
        .small-muted {
            color: var(--text-secondary) !important;
            font-size: 0.85rem;
        }

        div[data-testid="stMetricLabel"] p,
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricDelta"] {
            color: var(--text-primary) !important;
        }

        [data-testid="stSelectbox"] label,
        [data-testid="stTextInput"] label,
        [data-testid="stSlider"] label,
        [data-testid="stCheckbox"] label {
            color: var(--text-secondary) !important;
            font-weight: 600;
        }

        .stTextInput input,
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div {
            background: var(--input-bg);
            color: var(--text-primary) !important;
            border: 1px solid var(--input-border);
        }
        
                /* Fix sidebar text input readability */
        [data-testid="stSidebar"] .stTextInput input {
            background: rgba(255,255,255,0.96) !important;
            color: #10233f !important;
            -webkit-text-fill-color: #10233f !important;
            caret-color: #10233f !important;
            border: 1px solid rgba(255,255,255,0.18) !important;
        }
        
        [data-testid="stSidebar"] .stTextInput input::placeholder {
            color: rgba(16, 35, 63, 0.55) !important;
            -webkit-text-fill-color: rgba(16, 35, 63, 0.55) !important;
        }
        
        [data-testid="stSidebar"] .stTextInput input:focus {
            color: #10233f !important;
            -webkit-text-fill-color: #10233f !important;
        }

        .stSelectbox div[data-baseweb="select"] *,
        .stMultiSelect div[data-baseweb="select"] * {
            color: var(--text-primary) !important;
        }

        div[data-baseweb="popover"] * {
            color: #10233f !important;
        }

        div[data-testid="stAlertContainer"] p,
        div[data-testid="stAlertContainer"] span,
        div[data-testid="stAlertContainer"] li {
            color: inherit !important;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.12);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
def risk_badge_html(level: str) -> str:
    klass = f"risk-badge-{str(level).lower()}"
    if str(level).lower() not in {"low", "moderate", "high"}:
        klass = "risk-badge-unknown"
    return f'<span class="{klass}">{level}</span>'


def make_gauge(value: float):
    v = 0.0 if not np.isfinite(value) else float(np.clip(value, 0, 100))
    if not PLOTLY_OK:
        return None
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=v,
            number={"suffix": "", "font": {"size": 42, "color": "#f7fbff"}},
            title={"text": "ARI", "font": {"size": 18, "color": "#f7fbff"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#d9e8ff", "tickwidth": 1},
                "bar": {"color": "#ff7a7a", "thickness": 0.28},
                "bgcolor": "rgba(255,255,255,0.05)",
                "borderwidth": 1,
                "bordercolor": "rgba(255,255,255,0.10)",
                "steps": [
                    {"range": [0, 30], "color": "rgba(46, 204, 113, 0.55)"},
                    {"range": [30, 60], "color": "rgba(241, 196, 15, 0.55)"},
                    {"range": [60, 100], "color": "rgba(231, 76, 60, 0.55)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f7fbff"},
    )
    return fig
def make_trend_chart(df: pd.DataFrame):
    if df.empty:
        return None
    plot_df = df.copy()
    if not PLOTLY_OK:
        tmp = plot_df[["event_dt", "ari_ma"]].set_index("event_dt")
        return tmp

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["event_dt"],
            y=plot_df["ari_ma"],
            mode="lines+markers",
            name="ARI",
            line={"color": "#7ec8ff", "width": 3},
            marker={"size": 6, "color": "#bde6ff"},
        )
    )
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(46,204,113,0.15)", line_width=0)
    fig.add_hrect(y0=30, y1=60, fillcolor="rgba(241,196,15,0.12)", line_width=0)
    fig.add_hrect(y0=60, y1=100, fillcolor="rgba(231,76,60,0.12)", line_width=0)
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis_title="ARI (risk)",
        xaxis_title="Time",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.05)",
        legend_title_text="",
        font={"color": "#f7fbff"},
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    fig.update_yaxes(range=[0, 100], gridcolor="rgba(255,255,255,0.10)", zerolinecolor="rgba(255,255,255,0.10)")
    return fig
def make_distribution_chart(df: pd.DataFrame):
    if df.empty:
        return None
    counts = (
        df["risk_level"]
        .fillna("Unknown")
        .value_counts()
        .reindex(["Low", "Moderate", "High", "Unknown"], fill_value=0)
        .reset_index()
    )
    counts.columns = ["risk_level", "count"]
    if not PLOTLY_OK:
        return counts

    fig = go.Figure(
        go.Bar(
            x=counts["risk_level"],
            y=counts["count"],
            marker_color=["#67d38b", "#f4c542", "#ff7a7a", "#b8c2d1"],
        )
    )
    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.05)",
        showlegend=False,
        xaxis_title="",
        yaxis_title="Frames / ticks",
        font={"color": "#f7fbff"},
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.10)", zerolinecolor="rgba(255,255,255,0.10)")
    return fig
def most_distracted_period(df: pd.DataFrame) -> str:
    if df.empty:
        return "N/A"
    tmp = df.copy()
    tmp["minute_bin"] = tmp["event_dt"].dt.floor("min")
    agg = tmp.groupby("minute_bin")["ari_ma"].mean().sort_values(ascending=False)
    if agg.empty:
        return "N/A"
    return agg.index[0].strftime("%H:%M")


def latest_student_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    latest = (
        df.sort_values("event_dt")
        .groupby("student_id", as_index=False)
        .tail(1)
        .sort_values(["risk_level", "student_id"], ascending=[True, True], kind="stable")
    )
    return latest


def session_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = (
        df.assign(date=df["event_dt"].dt.date)
        .groupby(["date", "student_id"], as_index=False)
        .agg(
            avg_ari=("ari_ma", "mean"),
            avg_attention=("attention_pct", "mean"),
            high_risk_events=("alert_emitted", "sum"),
            max_severity=("severity_final", "max"),
        )
        .sort_values(["date", "student_id"], ascending=[False, True])
    )
    out["avg_ari"] = out["avg_ari"].round(1)
    out["avg_attention"] = out["avg_attention"].round(1)
    return out


def explain_row(row: pd.Series) -> str:
    cause = str(row.get("cause_final", "unknown"))
    sev = int(row.get("severity_final", 0))
    drift_prob = row.get("drift_prob", np.nan)

    explanation = {
        "drowsy/eyes": "Eye closure or drowsiness signal increased.",
        "yawning": "Yawning signal was detected.",
        "looking_away/head_pose": "Head pose suggests the student is looking away.",
        "fidgeting/motion": "Body motion increased beyond the stable threshold.",
        "unknown": "Multiple weak signals contributed to the attention drop.",
    }.get(cause, "Multiple weak signals contributed to the attention drop.")

    if np.isfinite(drift_prob):
        explanation += f" Drift probability: {float(drift_prob):.2f}."
    explanation += f" Severity: {sev}."
    return explanation


# -----------------------------
# Main app
# -----------------------------
def main() -> None:
    st.set_page_config(page_title="Instructor Dashboard", layout="wide")
    inject_css()

    st.title("Instructor Dashboard")
    st.caption("Real-time overview of attention risk, alerts, trends, and session summaries.")

    with st.sidebar:
        st.header("Data source")
        data_path = st.text_input(
            "SQLite / CSV / JSONL path",
            value=DEFAULT_DB,
            help="Put streamlit_app.py in the same project root and point this to the event log file.",
        )
        table_name = st.text_input("SQLite table", value=DEFAULT_TABLE)
        refresh_sec = st.slider("Auto refresh (sec)", min_value=0, max_value=10, value=2)
        show_raw = st.checkbox("Show raw table", value=False)

        if refresh_sec > 0:
            if AUTOREFRESH_OK:
                st_autorefresh(interval=refresh_sec * 1000, key="dashboard_refresh")
            else:
                st.info("Install streamlit-autorefresh for automatic updates, or refresh manually.")

    raw_df = load_events(data_path, table_name)
    df = normalize_events(raw_df)

    if df.empty:
        st.warning("No event data found yet.")
        st.markdown(
            """
            Put `streamlit_app.py` in the **same project root** as your live script and point it to the same event file.

            **Recommended layout**
            ```text
            your-project/
            ├─ attentiontrack/
            ├─ live_attentiontrack_langgraph_llm_record_blur.py
            ├─ streamlit_app.py
            ├─ attention_events.db
            └─ runs/
            ```

            **Minimal event fields the dashboard can read**
            - `event_time` or `timestamp`
            - `student_id`
            - `session_id`
            - `t`
            - `rule_ready`, `att_score`
            - `eye_closed`, `eye_drowsy`, `yawning`, `head_forward`, `body_stable`
            - `drift_ready`, `drift_prob`, `drift_state`
            - optional: `ari_raw`, `ari_ma`, `cause`, `severity`, `alert_emitted`

            If formal ARI is not logged yet, the dashboard will derive an MVP ARI from `att_score`.
            """
        )
        return

    sessions = sorted(df["session_id"].dropna().astype(str).unique().tolist())
    students = sorted(df["student_id"].dropna().astype(str).unique().tolist())

    filt_col1, filt_col2 = st.columns([1.2, 1.2])
    with filt_col1:
        selected_session = st.selectbox("Session", options=sessions, index=max(len(sessions) - 1, 0))
    with filt_col2:
        selected_student = st.selectbox("Student detail", options=students, index=0)

    session_df = df[df["session_id"].astype(str) == str(selected_session)].copy()
    student_df = session_df[session_df["student_id"].astype(str) == str(selected_student)].copy()
    latest_all = latest_student_snapshot(session_df)
    latest_student = student_df.sort_values("event_dt").tail(1)
    latest_row = latest_student.iloc[0] if not latest_student.empty else None

    st.markdown('<div class="panel"><div class="panel-title">A. Classroom Overview</div>', unsafe_allow_html=True)
    if latest_all.empty:
        st.info("No student snapshots for the selected session.")
    else:
        cols = st.columns(min(4, max(1, len(latest_all))))
        for idx, (_, row) in enumerate(latest_all.iterrows()):
            with cols[idx % len(cols)]:
                st.markdown(
                    f'''
                    <div class="student-card">
                        <div style="font-size:1rem;font-weight:700;">{row['student_id']}</div>
                        <div class="small-muted">ARI: {row['ari_ma']:.1f} | Attention: {row['attention_pct']:.1f}%</div>
                        {risk_badge_html(str(row['risk_level']))}
                        <div class="small-muted" style="margin-top:8px;">Drift: {str(row['drift_state'])} | Cause: {str(row['cause_final'])}</div>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )
    st.markdown('</div>', unsafe_allow_html=True)

    left, right = st.columns([1.05, 1.35])

    with left:
        st.markdown('<div class="panel"><div class="panel-title">B. Student Profile & Current Session</div>', unsafe_allow_html=True)
        if latest_row is None:
            st.info("No data for the selected student.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Current ARI", f"{latest_row['ari_ma']:.1f}")
            m2.metric("Risk", str(latest_row["risk_level"]))
            m3.metric("Drift prob", f"{float(latest_row['drift_prob']) if np.isfinite(latest_row['drift_prob']) else 0.0:.2f}")

            gauge = make_gauge(float(latest_row["ari_ma"]))
            if gauge is not None:
                st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
            else:
                st.progress(int(np.clip(latest_row["ari_ma"], 0, 100)))

            avg_ari = float(student_df["ari_ma"].mean()) if not student_df.empty else np.nan
            avg_att = float(student_df["attention_pct"].mean()) if not student_df.empty else np.nan
            high_risk_ticks = int((student_df["risk_level"] == "High").sum())
            total_alerts = int(student_df["alert_emitted"].sum())

            st.markdown(
                f"""
                - **Average ARI:** {avg_ari:.1f} ({_risk_label(avg_ari)})
                - **Average Attention:** {avg_att:.1f}%
                - **High Risk Events:** {total_alerts}
                - **High Risk Ticks:** {high_risk_ticks}
                - **Most Distracted Period:** {most_distracted_period(student_df)}
                """
            )
            st.info(explain_row(latest_row))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel"><div class="panel-title">Summary</div>', unsafe_allow_html=True)
        dist_fig = make_distribution_chart(student_df)
        if dist_fig is not None:
            if PLOTLY_OK:
                st.plotly_chart(dist_fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.bar_chart(dist_fig.set_index("risk_level"))
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel"><div class="panel-title">C. Session History & Trends</div>', unsafe_allow_html=True)
        trend_fig = make_trend_chart(student_df)
        if trend_fig is not None:
            if PLOTLY_OK:
                st.plotly_chart(trend_fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.line_chart(trend_fig)

        event_cols = [
            "event_dt", "student_id", "ari_ma", "risk_level", "att_score", "drift_prob", "drift_state",
            "issue_final", "cause_final", "severity_final", "banner", "message", "micro_task", "check_question",
        ]
        event_cols = [c for c in event_cols if c in student_df.columns]
        recent_events = student_df.sort_values("event_dt", ascending=False).head(15)[event_cols].copy()
        if "ari_ma" in recent_events.columns:
            recent_events["ari_ma"] = recent_events["ari_ma"].round(1)
        if "drift_prob" in recent_events.columns:
            recent_events["drift_prob"] = recent_events["drift_prob"].round(2)
        st.dataframe(recent_events, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">D. Session Summary Table</div>', unsafe_allow_html=True)
    hist = session_history(df)
    st.dataframe(hist, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if show_raw:
        st.markdown('<div class="panel"><div class="panel-title">Raw events</div>', unsafe_allow_html=True)
        st.dataframe(df.sort_values("event_dt", ascending=False), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()