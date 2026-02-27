"""
Varian Trajectory Log Analyzer  ·  Department Edition
=======================================================
Two modes:
  1. 📂 Manual Upload  – drag-and-drop .bin files (single or batch)
  2. 📡 Folder Scan    – point to network log folders per machine;
                         auto-scans for new files, caches results,
                         shows per-machine trends and per-patient
                         fraction-by-fraction logs.
"""

import json
import os
import re
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from pylinac import TrajectoryLog

# ─────────────────────────────────────────────────────────────────────────────
# Paths – stored next to app.py
# ─────────────────────────────────────────────────────────────────────────────
APP_DIR       = Path(__file__).parent
MACHINES_JSON = APP_DIR / "machines.json"
CACHE_JSON    = APP_DIR / "results_cache.json"

# ─────────────────────────────────────────────────────────────────────────────
# Page config & CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Trajectory Log Analyzer", page_icon="⚛️", layout="wide")
st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"]{background:#0d1117;color:#e6edf3}
[data-testid="stSidebar"]{background:#161b22}
[data-testid="stSidebarContent"] hr{border-color:#30363d}
.metric-card{background:#161b22;border:1px solid #30363d;border-radius:12px;
             padding:16px 20px;text-align:center;margin-bottom:6px}
.metric-card h2{color:#58a6ff;margin:0;font-size:1.7rem}
.metric-card p{color:#8b949e;margin:0;font-size:.8rem}
h1,h2,h3{color:#58a6ff}
.stTabs [data-baseweb="tab"]{color:#8b949e}
.stTabs [aria-selected="true"]{color:#58a6ff;border-bottom:2px solid #58a6ff}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Persistent config helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_machines() -> list[dict]:
    if MACHINES_JSON.exists():
        return json.loads(MACHINES_JSON.read_text())
    return []


def save_machines(machines: list[dict]):
    MACHINES_JSON.write_text(json.dumps(machines, indent=2))


def load_cache() -> dict:
    if CACHE_JSON.exists():
        try:
            return json.loads(CACHE_JSON.read_text())
        except Exception:
            return {}
    return {}


def save_cache(cache: dict):
    CACHE_JSON.write_text(json.dumps(cache, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Filename parser
# Varian format: {PatientID}_{MachTag}_{Plan}_{Arc(s)}_{YYYYMMDDHHMMSS}.bin
# ─────────────────────────────────────────────────────────────────────────────
_LOG_RE = re.compile(
    r"^(?P<patient_id>\d+)_"
    r"(?P<mach_tag>[^_]+)_"
    r"(?P<plan>[^_]+)_"
    r"(?P<arc>[^_]+)_"
    r"(?P<dt>\d{14})\.bin$",
    re.IGNORECASE,
)

def parse_filename(fname: str) -> dict | None:
    m = _LOG_RE.match(Path(fname).name)
    if not m:
        return None
    dt_str = m.group("dt")
    try:
        dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
    except ValueError:
        dt = None
    return {
        "patient_id":  m.group("patient_id"),
        "mach_tag":    m.group("mach_tag"),
        "plan":        m.group("plan"),
        "arc":         m.group("arc"),
        "datetime":    dt.isoformat() if dt else dt_str,
        "date":        dt.strftime("%Y-%m-%d") if dt else dt_str[:8],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core: load & analyse a .bin file
# ─────────────────────────────────────────────────────────────────────────────
def load_log(path: str | Path) -> TrajectoryLog | None:
    tmp_path = None
    try:
        with open(path, "rb") as fh:
            data = fh.read()
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        return TrajectoryLog(tmp_path)
    except Exception:
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def load_log_from_upload(uploaded_file) -> TrajectoryLog | None:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        return TrajectoryLog(tmp_path)
    except Exception as exc:
        st.warning(f"⚠️ Could not parse **{uploaded_file.name}**: {exc}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def extract_arrays(tlog: TrajectoryLog):
    ad  = tlog.axis_data
    la  = ad.mlc.leaf_axes
    keys = sorted(la.keys())
    actual_full   = np.array([la[k].actual   for k in keys]).T
    expected_full = np.array([la[k].expected for k in keys]).T
    gantry_full   = np.array(ad.gantry.actual)
    n = min(actual_full.shape[0], expected_full.shape[0], len(gantry_full))
    return actual_full[:n], expected_full[:n], gantry_full[:n], getattr(tlog, "subbeams", [])


def compute_metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    err          = actual - expected
    ptp          = np.ptp(actual, axis=0)
    moving       = ptp > 0.5
    rms_per_leaf = np.sqrt(np.mean(err ** 2, axis=0))
    overall_rms  = (float(np.sqrt(np.mean(err[:, moving] ** 2)))
                    if moving.sum() > 0 else float(np.sqrt(np.mean(err ** 2))))
    worst_rms    = float(rms_per_leaf.max())
    worst_leaf   = int(np.argmax(rms_per_leaf))

    half = actual.shape[1] // 2
    diffs = np.abs(np.diff(actual, axis=0))
    mt = diffs.max() if diffs.max() > 0 else 1.0
    lsv = float(np.clip((1 - diffs.mean(axis=1) / mt).mean(), 0, 1))
    aps = np.abs(actual[:, half:] - actual[:, :half]).mean(axis=1)
    mu, sig = aps.mean(), aps.std()
    aav = float(np.clip(1 - sig / mu, 0, 1)) if mu > 0 else 0.0

    return {
        "overall_rms":  round(overall_rms, 6),
        "worst_rms":    round(worst_rms, 6),
        "worst_leaf":   worst_leaf,
        "n_moving":     int(moving.sum()),
        "n_leaves":     actual.shape[1],
        "n_snaps":      actual.shape[0],
        "MCS": round(lsv * aav, 4),
        "LSV": round(lsv, 4),
        "AAV": round(aav, 4),
        # heavy arrays kept in memory only during upload mode / deep-dive
        "_rms_per_leaf": rms_per_leaf.tolist(),
        "_leaf_error":   err.tolist(),     # only stored in RAM, not in cache
        "_ptp":          ptp.tolist(),
        "_moving":       moving.tolist(),
    }


def analyse_file(path: str | Path) -> dict | None:
    """Parse + compute metrics. Returns cacheable dict (no large arrays)."""
    tlog = load_log(path)
    if tlog is None:
        return None
    try:
        actual, expected, gantry, subbeams = extract_arrays(tlog)
        m = compute_metrics(actual, expected)
        return {
            "overall_rms": m["overall_rms"],
            "worst_rms":   m["worst_rms"],
            "worst_leaf":  m["worst_leaf"],
            "n_moving":    m["n_moving"],
            "n_leaves":    m["n_leaves"],
            "n_snaps":     m["n_snaps"],
            "MCS":         m["MCS"],
            "LSV":         m["LSV"],
            "AAV":         m["AAV"],
            "n_arcs":      len(subbeams),
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Folder scanner with caching
# ─────────────────────────────────────────────────────────────────────────────
def scan_folder(machine_name: str, folder_path: str,
                cache: dict, progress_bar=None) -> tuple[list[dict], dict]:
    """
    Walk folder_path recursively for .bin files. Skip cached (unchanged) files.
    Returns (rows, updated_cache).
    """
    rows = []
    folder = Path(folder_path)
    if not folder.exists():
        return rows, cache

    bin_files = list(folder.rglob("*.bin"))
    total = len(bin_files)

    for idx, fpath in enumerate(bin_files):
        if progress_bar:
            progress_bar.progress((idx + 1) / max(total, 1),
                                  text=f"Scanning {machine_name}: {fpath.name}")

        meta = parse_filename(fpath.name)
        if meta is None:
            continue

        cache_key  = str(fpath)
        file_mtime = fpath.stat().st_mtime

        if cache_key in cache and cache[cache_key].get("mtime") == file_mtime:
            # Use cached result
            entry = cache[cache_key]
        else:
            metrics = analyse_file(fpath)
            if metrics is None:
                continue
            entry = {**metrics, **meta, "mtime": file_mtime, "file": cache_key}
            cache[cache_key] = entry

        rows.append({
            "machine":     machine_name,
            "file":        cache_key,
            "patient_id":  meta["patient_id"],
            "mach_tag":    meta["mach_tag"],
            "plan":        meta["plan"],
            "arc":         meta["arc"],
            "date":        meta["date"],
            "datetime":    meta["datetime"],
            "overall_rms": entry["overall_rms"],
            "worst_rms":   entry["worst_rms"],
            "worst_leaf":  entry["worst_leaf"],
            "n_leaves":    entry["n_leaves"],
            "n_snaps":     entry["n_snaps"],
            "n_arcs":      entry.get("n_arcs", 1),
            "MCS":         entry["MCS"],
            "LSV":         entry["LSV"],
            "AAV":         entry["AAV"],
            "status":      "✅ PASS" if entry["overall_rms"] < 1.0 else "❌ REVIEW",
        })

    return rows, cache


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────
def metric_card(label, value):
    return f'<div class="metric-card"><h2>{value}</h2><p>{label}</p></div>'


def dark_layout(fig, title=""):
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3"),
        title=dict(text=title, font=dict(color="#58a6ff", size=14)),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        xaxis=dict(gridcolor="#30363d"),
        yaxis=dict(gridcolor="#30363d"),
    )
    return fig


def rms_trend_chart(df: pd.DataFrame, group_col: str, title: str) -> go.Figure:
    """Line chart: date on X, avg RMS on Y, one trace per group_col value."""
    grouped = (df.groupby([group_col, "date"])["overall_rms"]
               .mean().reset_index()
               .sort_values("date"))
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, grp in enumerate(grouped[group_col].unique()):
        sub = grouped[grouped[group_col] == grp]
        fig.add_trace(go.Scatter(
            x=sub["date"], y=sub["overall_rms"],
            mode="lines+markers", name=str(grp),
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=6)))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#f78166",
                  annotation_text="1 mm tolerance", annotation_font_color="#f78166")
    dark_layout(fig, title)
    fig.update_layout(xaxis_title="Date", yaxis_title="Avg Overall RMS (mm)")
    return fig


def fraction_trend_chart(df: pd.DataFrame, patient_id: str) -> go.Figure:
    """Per-fraction RMS chart for one patient across all machines."""
    sub = df[df["patient_id"] == patient_id].sort_values("datetime")
    fig = go.Figure()
    for mach in sub["machine"].unique():
        msub = sub[sub["machine"] == mach]
        fig.add_trace(go.Scatter(
            x=msub["datetime"], y=msub["overall_rms"],
            mode="lines+markers", name=mach,
            marker=dict(size=8),
            text=msub["arc"], hovertemplate=(
                "<b>%{text}</b><br>RMS: %{y:.4f} mm<br>%{x}<extra></extra>")))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#f78166",
                  annotation_text="1 mm tolerance", annotation_font_color="#f78166")
    dark_layout(fig, f"Fraction-by-Fraction RMS — Patient {patient_id}")
    fig.update_layout(xaxis_title="Treatment DateTime", yaxis_title="Overall RMS (mm)")
    return fig


def mcs_bar(df: pd.DataFrame, x_col: str, title: str) -> go.Figure:
    avg = df.groupby(x_col)["MCS"].mean().reset_index().sort_values("MCS")
    colors = ["#f78166" if v < 0.2 else "#d29922" if v < 0.4 else "#3fb950"
              for v in avg["MCS"]]
    fig = go.Figure(go.Bar(x=avg[x_col], y=avg["MCS"], marker_color=colors,
                           text=avg["MCS"].round(3), textposition="outside"))
    dark_layout(fig, title)
    fig.update_layout(xaxis_title=x_col, yaxis_title="Mean MCS",
                      yaxis_range=[0, 1.05])
    return fig


def worst_leaf_heatmap(df: pd.DataFrame, machine: str) -> go.Figure:
    """Heatmap: date × leaf index, colour = RMS error value."""
    sub = df[df["machine"] == machine].copy()
    if sub.empty:
        return go.Figure()
    pivot = sub.pivot_table(index="date", columns="worst_leaf",
                            values="worst_rms", aggfunc="mean")
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.astype(str),
        y=pivot.index.astype(str),
        colorscale="YlOrRd", colorbar=dict(title="Worst RMS (mm)")))
    dark_layout(fig, f"Worst Leaf Heatmap — {machine}")
    fig.update_layout(xaxis_title="Leaf Index", yaxis_title="Date", height=320)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Deep-dive (shared by both modes)
# ─────────────────────────────────────────────────────────────────────────────
def gravity_sag_polar(gantry_a, mlc_e):
    bins = np.arange(0, 361, 5)
    centres = (bins[:-1] + bins[1:]) / 2
    binned = [mlc_e[(gantry_a >= lo) & (gantry_a < hi)].mean()
              if ((gantry_a >= lo) & (gantry_a < hi)).sum() > 0 else np.nan
              for lo, hi in zip(bins[:-1], bins[1:])]
    binned = np.array(binned)
    mean_e = np.nanmean(binned)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=np.full(361, mean_e), theta=np.linspace(0,360,361),
        mode="lines", line=dict(color="#8b949e", width=1, dash="dash"),
        name=f"Mean ({mean_e:.4f} mm)"))
    fig.add_trace(go.Scatterpolar(
        r=np.where(np.isnan(binned), 0, binned), theta=centres,
        mode="lines+markers",
        line=dict(color="#58a6ff", width=2),
        marker=dict(size=5, color=np.where(np.isnan(binned), 0, binned),
                    colorscale="Plasma", showscale=True, colorbar=dict(title="mm", x=1.1)),
        fill="toself", fillcolor="rgba(88,166,255,0.15)", name="Avg MLC Error"))
    for ang, lbl in [(90,"90° Gravity↓"),(270,"270° Gravity↑")]:
        fig.add_trace(go.Scatterpolar(
            r=[max(np.nanmax(binned)*1.15, mean_e*1.5)], theta=[ang],
            mode="text", text=[lbl], textfont=dict(color="#f78166",size=11),
            showlegend=False))
    fig.update_layout(
        polar=dict(bgcolor="#0d1117",
                   radialaxis=dict(tickfont=dict(color="#8b949e"),gridcolor="#30363d"),
                   angularaxis=dict(tickfont=dict(color="#8b949e"),gridcolor="#30363d",
                                    direction="clockwise", rotation=90)),
        paper_bgcolor="#0d1117", font=dict(color="#e6edf3"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        title=dict(text="🌍 Gravity Sag — MLC Error vs. Gantry Angle",
                   font=dict(color="#58a6ff", size=14)))
    return fig


def gravity_sag_cart(gantry_a, mlc_e):
    df = pd.DataFrame({"a": gantry_a, "e": mlc_e}).sort_values("a")
    df["roll"] = df["e"].rolling(max(1, len(df)//60), center=True).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["a"], y=df["e"], mode="markers",
        marker=dict(size=3, color=df["e"], colorscale="Plasma", showscale=True,
                    colorbar=dict(title="mm"), opacity=0.5), name="Per-snapshot"))
    fig.add_trace(go.Scatter(x=df["a"], y=df["roll"], mode="lines",
        line=dict(color="#3fb950", width=2), name="Rolling mean"))
    for ang in [90, 270]:
        fig.add_vline(x=ang, line_dash="dash", line_color="#f78166",
                      annotation_text=f"{ang}°", annotation_font_color="#f78166")
    dark_layout(fig, "MLC Error vs. Gantry Angle")
    fig.update_layout(xaxis=dict(title="Gantry Angle (°)", tickvals=list(range(0,361,45))),
                      yaxis_title="Avg MLC Error (mm)")
    return fig


def render_deep_dive(tlog_path: str | None = None, uploaded_file=None,
                     label: str = ""):
    st.markdown(f"---\n### 🔬 Deep Dive — `{label}`")

    # Load arrays
    if tlog_path:
        tlog = load_log(tlog_path)
    elif uploaded_file:
        tlog = load_log_from_upload(uploaded_file)
    else:
        st.info("No file selected.")
        return

    if tlog is None:
        st.error("Could not load this log file.")
        return

    try:
        actual, expected, gantry, subbeams = extract_arrays(tlog)
        m = compute_metrics(actual, expected)
    except Exception as exc:
        st.error(f"Extraction failed: {exc}")
        return

    n_leaves = actual.shape[1]

    # Arc selector
    arc_ops = ["All Arcs (full log)"]
    arc_ops += [f"Arc {i+1}: {getattr(sb,'beam_name',f'Subbeam {i}')} ({len(sb._snapshots)} snaps)"
                for i, sb in enumerate(subbeams)]
    sel = st.selectbox("🔄 Select Arc", arc_ops, key=f"arc_{label}")
    arc_idx = arc_ops.index(sel)

    if arc_idx > 0 and subbeams:
        sb  = subbeams[arc_idx - 1]
        si  = np.array(sb._snapshots)
        si  = si[si < actual.shape[0]]
        actual_s, expected_s, gantry_s = actual[si], expected[si], gantry[si]
        ms = compute_metrics(actual_s, expected_s)
    else:
        actual_s, expected_s, gantry_s, ms = actual, expected, gantry, m

    le_s  = np.array(ms["_leaf_error"])
    rms_s = np.array(ms["_rms_per_leaf"])
    mov_s = np.array(ms["_moving"])
    snap_err = (np.abs(le_s[:, mov_s]).mean(axis=1)
                if mov_s.sum() > 0 else np.abs(le_s).mean(axis=1))

    # Summary cards
    cols = st.columns(5)
    cards = [("Overall RMS",   f"{ms['overall_rms']:.4f} mm"),
             ("Worst Leaf RMS",f"{ms['worst_rms']:.4f} mm"),
             ("Worst Leaf #",  f"#{ms['worst_leaf']}"),
             ("Moving Leaves", str(ms["n_moving"])),
             ("Snapshots",     str(actual_s.shape[0]))]
    for col, (lbl, val) in zip(cols, cards):
        col.markdown(metric_card(lbl, val), unsafe_allow_html=True)

    # MCS
    st.markdown("#### 🔢 MCS")
    mc = st.columns(3)
    for col, k in zip(mc, ["MCS","LSV","AAV"]):
        col.markdown(metric_card(k, str(ms[k])), unsafe_allow_html=True)
    if ms["MCS"] < 0.20:
        st.warning(
            "⚠️ **Low MCS detected (MCS < 0.20)** — this plan has very high modulation. "
            "Please review the plan: highly modulated beams are associated with an increased "
            "risk of PSQA failure. Consider re-optimising or verifying with a secondary "
            "dose calculation before treatment.",
            icon="🔴")

    # RMS bar
    st.markdown("#### 📈 RMS by Leaf")
    q90 = np.quantile(rms_s, 0.9)
    fig_bar = go.Figure(go.Bar(
        x=list(range(n_leaves)), y=rms_s,
        marker_color=["#f78166" if v > q90 else "#58a6ff" for v in rms_s]))
    dark_layout(fig_bar, "Per-Leaf RMS Error")
    fig_bar.update_layout(xaxis_title="Leaf Index", yaxis_title="RMS (mm)")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Worst leaves table
    top_n = st.slider("Show top N worst leaves", 5, min(40, n_leaves), 10,
                      key=f"sl_{label}")
    leaf_df = pd.DataFrame({
        "Leaf Index":      np.arange(n_leaves),
        "Bank":            ["A" if i < n_leaves//2 else "B" for i in range(n_leaves)],
        "RMS Error (mm)":  rms_s.round(5),
        "Max Error (mm)":  np.abs(le_s).max(axis=0).round(5),
        "P2P Travel (mm)": np.array(ms["_ptp"]).round(2),
        "Moving":          mov_s,
    }).sort_values("RMS Error (mm)", ascending=False).head(top_n)
    st.dataframe(leaf_df.style.background_gradient(subset=["RMS Error (mm)"], cmap="Reds"),
                 use_container_width=True)

    # Gravity sag
    st.markdown("#### 🌍 Gravity Sag")
    tp, tc = st.tabs(["🧭 Polar", "📉 Cartesian"])
    with tp: st.plotly_chart(gravity_sag_polar(gantry_s, snap_err), use_container_width=True)
    with tc: st.plotly_chart(gravity_sag_cart(gantry_s, snap_err), use_container_width=True)

    # Heatmap
    st.markdown("#### 🗺️ MLC Error Heatmap")
    step = max(1, actual_s.shape[0] // 500)
    fig_h = go.Figure(go.Heatmap(z=le_s[::step,:].T, colorscale="RdBu", zmid=0,
                                  colorbar=dict(title="Error (mm)")))
    dark_layout(fig_h, "MLC Positional Error Heatmap")
    fig_h.update_layout(xaxis_title="Snapshot (downsampled)",
                        yaxis_title="Leaf Index", height=380)
    st.plotly_chart(fig_h, use_container_width=True)

    st.download_button(f"⬇️ Download CSV — {label}",
                       data=leaf_df.to_csv(index=False).encode(),
                       file_name=f"{label}_leaf_errors.csv",
                       mime="text/csv", key=f"dl_{label}")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def sidebar() -> str:
    with st.sidebar:
        st.markdown("## ⚛️ Trajectory Log Analyzer")
        st.divider()
        mode = st.radio("**Mode**", ["📂 Manual Upload", "📡 Folder Scan"], index=0)
        st.divider()

        if mode == "📡 Folder Scan":
            st.markdown("### 🏥 Machine Configuration")
            machines = load_machines()

            # Display existing machines
            to_remove = []
            for i, m in enumerate(machines):
                cols = st.columns([3, 1])
                cols[0].markdown(f"**{m['name']}**  \n`{m['path']}`")
                if cols[1].button("🗑️", key=f"del_{i}", help="Remove"):
                    to_remove.append(i)
            for i in reversed(to_remove):
                machines.pop(i)
                save_machines(machines)
                st.rerun()

            st.divider()
            st.markdown("**➕ Add Machine**")
            new_name = st.text_input("Machine name", placeholder="e.g. Halcyon 1")
            new_path = st.text_input("Log folder path",
                                     placeholder=r"\\Server\Halcyon1\Logs")
            if st.button("Add", use_container_width=True):
                if new_name and new_path:
                    machines.append({"name": new_name, "path": new_path})
                    save_machines(machines)
                    st.success(f"Added **{new_name}**")
                    st.rerun()

            st.divider()
            if st.button("🗑️ Clear Results Cache", use_container_width=True):
                if CACHE_JSON.exists():
                    CACHE_JSON.unlink()
                st.success("Cache cleared.")

        st.divider()
        st.caption("Powered by pylinac & Streamlit")

    return mode


# ─────────────────────────────────────────────────────────────────────────────
# MANUAL UPLOAD MODE
# ─────────────────────────────────────────────────────────────────────────────
def manual_upload_mode():
    uploaded_files = st.file_uploader(
        "📂 Upload Trajectory Logs (.bin)",
        type=["bin"], accept_multiple_files=True,
        help="Hold Ctrl/⌘ to select multiple files.")

    if not uploaded_files:
        st.info("👆 Upload one or more **.bin** trajectory log files to begin.", icon="ℹ️")
        st.stop()

    results = []
    with st.spinner(f"Parsing {len(uploaded_files)} file(s)…"):
        for uf in uploaded_files:
            tlog = load_log_from_upload(uf)
            if tlog is None:
                continue
            try:
                actual, expected, gantry, subbeams = extract_arrays(tlog)
                m = compute_metrics(actual, expected)
                results.append({"name": uf.name, "actual": actual,
                                 "expected": expected, "gantry": gantry,
                                 "subbeams": subbeams, "metrics": m})
            except Exception as exc:
                st.warning(f"⚠️ {uf.name}: {exc}")

    if not results:
        st.error("No files could be parsed.")
        st.stop()

    st.success(f"✅ Parsed {len(results)} of {len(uploaded_files)} file(s).")

    # Summary table
    st.markdown("## 📋 Batch Summary")
    rows = [{
        "File": r["name"],
        "Leaves": r["metrics"]["n_leaves"],
        "Snapshots": r["metrics"]["n_snaps"],
        "Moving Leaves": r["metrics"]["n_moving"],
        "Overall RMS (mm)": r["metrics"]["overall_rms"],
        "Worst Leaf RMS (mm)": r["metrics"]["worst_rms"],
        "Worst Leaf #": r["metrics"]["worst_leaf"],
        "MCS": r["metrics"]["MCS"],
        "Status": "✅ PASS" if r["metrics"]["overall_rms"] < 1.0 else "❌ REVIEW",
    } for r in results]
    sdf = pd.DataFrame(rows)
    st.dataframe(sdf.style.background_gradient(subset=["Overall RMS (mm)"], cmap="YlOrRd"),
                 use_container_width=True)
    st.download_button("⬇️ Download Batch CSV", sdf.to_csv(index=False).encode(),
                       "batch_report.csv", "text/csv")
    st.divider()

    # Comparison charts (multi-file)
    if len(results) > 1:
        st.markdown("## 📊 Cross-File Comparison")
        c1, c2 = st.columns(2)
        names = [r["name"] for r in results]
        rms_v = [r["metrics"]["overall_rms"] for r in results]
        mcs_v = [r["metrics"]["MCS"] for r in results]
        with c1:
            fr = go.Figure(go.Bar(x=names, y=rms_v,
                marker_color=["#f78166" if v >= 1 else "#58a6ff" for v in rms_v],
                text=[f"{v:.4f}" for v in rms_v], textposition="outside"))
            fr.add_hline(y=1.0, line_dash="dash", line_color="#f78166")
            dark_layout(fr, "Overall RMS per File")
            fr.update_layout(xaxis=dict(tickangle=-30), yaxis_title="RMS (mm)")
            st.plotly_chart(fr, use_container_width=True)
        with c2:
            fm = go.Figure(go.Bar(x=names, y=mcs_v,
                marker_color=["#3fb950" if v >= 0.4 else "#d29922" if v >= 0.2 else "#f78166"
                              for v in mcs_v],
                text=[f"{v:.3f}" for v in mcs_v], textposition="outside"))
            dark_layout(fm, "MCS per File")
            fm.update_layout(xaxis=dict(tickangle=-30), yaxis=dict(title="MCS", range=[0,1.1]))
            st.plotly_chart(fm, use_container_width=True)
        st.divider()

    # Deep-dive
    st.markdown("## 🔬 Per-File Deep Dive")
    sel = st.selectbox("Select file", [r["name"] for r in results])
    r = next(x for x in results if x["name"] == sel)
    # Re-upload isn't available here — we pass None for tlog_path and use trick:
    # we already have arrays in memory, so call a lightweight version directly
    _inline_deep_dive(r["name"], r["actual"], r["expected"],
                      r["gantry"], r["metrics"], r["subbeams"])


def _inline_deep_dive(name, actual, expected, gantry, m, subbeams):
    """Deep-dive when arrays are already in RAM (upload mode)."""
    st.markdown(f"---\n### 🔬 Deep Dive — `{name}`")
    n_leaves = actual.shape[1]
    arc_ops = ["All Arcs (full log)"]
    arc_ops += [f"Arc {i+1}: {getattr(sb,'beam_name',f'Subbeam {i}')} ({len(sb._snapshots)} snaps)"
                for i, sb in enumerate(subbeams)]
    sel = st.selectbox("🔄 Select Arc", arc_ops, key=f"arc_{name}")
    arc_idx = arc_ops.index(sel)
    if arc_idx > 0 and subbeams:
        sb = subbeams[arc_idx-1]
        si = np.array(sb._snapshots); si = si[si < actual.shape[0]]
        actual_s, expected_s, gantry_s = actual[si], expected[si], gantry[si]
        ms = compute_metrics(actual_s, expected_s)
    else:
        actual_s, expected_s, gantry_s, ms = actual, expected, gantry, m

    le_s  = np.array(ms["_leaf_error"])
    rms_s = np.array(ms["_rms_per_leaf"])
    mov_s = np.array(ms["_moving"])
    snap_err = (np.abs(le_s[:, mov_s]).mean(axis=1)
                if mov_s.sum() > 0 else np.abs(le_s).mean(axis=1))

    cols = st.columns(5)
    for col,(lbl,val) in zip(cols, [
        ("Overall RMS",   f"{ms['overall_rms']:.4f} mm"),
        ("Worst Leaf RMS",f"{ms['worst_rms']:.4f} mm"),
        ("Worst Leaf #",  f"#{ms['worst_leaf']}"),
        ("Moving Leaves", str(ms["n_moving"])),
        ("Snapshots",     str(actual_s.shape[0]))]):
        col.markdown(metric_card(lbl, val), unsafe_allow_html=True)

    mc = st.columns(3)
    for col,k in zip(mc,["MCS","LSV","AAV"]):
        col.markdown(metric_card(k, str(ms[k])), unsafe_allow_html=True)
    if ms["MCS"] < 0.20:
        st.warning(
            "⚠️ **Low MCS detected (MCS < 0.20)** — this plan has very high modulation. "
            "Please review the plan: highly modulated beams are associated with an increased "
            "risk of PSQA failure. Consider re-optimising or verifying with a secondary "
            "dose calculation before treatment.",
            icon="🔴")

    q90 = np.quantile(rms_s, 0.9)
    fig_bar = go.Figure(go.Bar(x=list(range(n_leaves)), y=rms_s,
        marker_color=["#f78166" if v>q90 else "#58a6ff" for v in rms_s]))
    dark_layout(fig_bar, "Per-Leaf RMS Error")
    st.plotly_chart(fig_bar, use_container_width=True)

    top_n = st.slider("Top N worst leaves", 5, min(40, n_leaves), 10, key=f"sl_{name}")
    leaf_df = pd.DataFrame({
        "Leaf Index": np.arange(n_leaves),
        "Bank": ["A" if i < n_leaves//2 else "B" for i in range(n_leaves)],
        "RMS Error (mm)": rms_s.round(5),
        "Max Error (mm)": np.abs(le_s).max(axis=0).round(5),
        "P2P Travel (mm)": np.array(ms["_ptp"]).round(2),
        "Moving": mov_s,
    }).sort_values("RMS Error (mm)", ascending=False).head(top_n)
    st.dataframe(leaf_df.style.background_gradient(subset=["RMS Error (mm)"], cmap="Reds"),
                 use_container_width=True)

    st.markdown("#### 🌍 Gravity Sag")
    tp, tc = st.tabs(["🧭 Polar", "📉 Cartesian"])
    with tp: st.plotly_chart(gravity_sag_polar(gantry_s, snap_err), use_container_width=True)
    with tc: st.plotly_chart(gravity_sag_cart(gantry_s, snap_err), use_container_width=True)

    step = max(1, actual_s.shape[0]//500)
    fig_h = go.Figure(go.Heatmap(z=le_s[::step,:].T, colorscale="RdBu", zmid=0,
                                  colorbar=dict(title="Error (mm)")))
    dark_layout(fig_h, "MLC Error Heatmap")
    fig_h.update_layout(xaxis_title="Snapshot (downsampled)",
                        yaxis_title="Leaf Index", height=380)
    st.plotly_chart(fig_h, use_container_width=True)
    st.download_button(f"⬇️ Download CSV — {name}", leaf_df.to_csv(index=False).encode(),
                       f"{name}_errors.csv", "text/csv", key=f"dl_{name}")


# ─────────────────────────────────────────────────────────────────────────────
# FOLDER SCAN MODE
# ─────────────────────────────────────────────────────────────────────────────
def folder_scan_mode():
    machines = load_machines()

    if not machines:
        st.info("👈 Add at least one machine in the sidebar to start scanning.", icon="ℹ️")
        st.stop()

    # Scan button
    col_btn, col_auto = st.columns([2, 4])
    do_scan = col_btn.button("🔍 Scan Now", use_container_width=True, type="primary")
    col_auto.caption("Results are cached — only new/changed files are re-processed.")

    cache = load_cache()
    all_rows: list[dict] = []

    if do_scan:
        prog = st.progress(0, "Starting scan…")
        for i, mach in enumerate(machines):
            rows, cache = scan_folder(mach["name"], mach["path"], cache, prog)
            all_rows.extend(rows)
            prog.progress((i+1)/len(machines), f"Done: {mach['name']}")
        save_cache(cache)
        prog.empty()
        # Store in session so we keep results between reruns
        st.session_state["scan_rows"] = all_rows
        if all_rows:
            st.success(f"✅ Found **{len(all_rows)}** logs across **{len(machines)}** machine(s).")
        else:
            st.warning("No valid trajectory log files found in the configured folders.")
            st.stop()
    elif "scan_rows" in st.session_state:
        all_rows = st.session_state["scan_rows"]
    else:
        st.info("Press **Scan Now** to load logs from the configured folders.", icon="📡")
        st.stop()

    if not all_rows:
        st.stop()

    df = pd.DataFrame(all_rows)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["date"]     = pd.to_datetime(df["date"], errors="coerce")

    # ── Date range filter ─────────────────────────────────────────────
    st.markdown("---")
    min_d, max_d = df["date"].min(), df["date"].max()
    d1, d2 = st.date_input("📅 Date range",
                            value=[min_d, max_d],
                            min_value=min_d, max_value=max_d)
    df = df[(df["date"] >= pd.Timestamp(d1)) & (df["date"] <= pd.Timestamp(d2))]
    if df.empty:
        st.warning("No logs in the selected date range.")
        st.stop()

    # ═══════════════════════════════════════════════════════════════════
    # TABS: Machine Dashboard | Patient Log Book
    # ═══════════════════════════════════════════════════════════════════
    tab_mach, tab_pat = st.tabs(["📡 Machine Dashboard", "👤 Patient Log Book"])

    # ─────────────────────────────────────────────────────────────────
    # TAB 1 — MACHINE DASHBOARD
    # ─────────────────────────────────────────────────────────────────
    with tab_mach:
        # Department-level summary cards
        pass_rate = (df["status"] == "✅ PASS").mean() * 100
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.markdown(metric_card("Total Fractions",    str(len(df))), unsafe_allow_html=True)
        sc2.markdown(metric_card("Unique Patients",    str(df["patient_id"].nunique())), unsafe_allow_html=True)
        sc3.markdown(metric_card("Dept Pass Rate",     f"{pass_rate:.1f}%"), unsafe_allow_html=True)
        sc4.markdown(metric_card("Machines Monitored", str(df["machine"].nunique())), unsafe_allow_html=True)

        st.markdown("---")

        # Machine filter
        mach_list = ["All Machines"] + sorted(df["machine"].unique().tolist())
        sel_mach  = st.selectbox("Filter by Machine", mach_list)
        df_m = df if sel_mach == "All Machines" else df[df["machine"] == sel_mach]

        # RMS trend
        st.markdown("#### 📈 RMS Trend Over Time")
        st.plotly_chart(rms_trend_chart(df_m, "machine", "Average Overall RMS — by Machine"),
                        use_container_width=True)

        # MCS comparison
        st.markdown("#### 🔢 Mean MCS by Machine")
        st.plotly_chart(mcs_bar(df_m, "machine", "Mean MCS per Machine"),
                        use_container_width=True)

        # Worst-leaf heatmap per machine
        if sel_mach != "All Machines":
            st.markdown(f"#### 🗺️ Worst-Leaf Heatmap — {sel_mach}")
            st.caption("Shows which leaf is the worst performer on each treatment day.")
            st.plotly_chart(worst_leaf_heatmap(df_m, sel_mach), use_container_width=True)

        # Log table for this machine
        st.markdown("#### 📋 Log Table")
        disp = df_m[["machine","patient_id","date","arc","plan",
                      "overall_rms","worst_rms","worst_leaf","MCS","status"]].copy()
        disp = disp.sort_values("date", ascending=False)
        disp.columns = ["Machine","Patient ID","Date","Arc","Plan",
                         "Overall RMS (mm)","Worst RMS (mm)","Worst Leaf","MCS","Status"]
        st.dataframe(disp.style.background_gradient(subset=["Overall RMS (mm)"], cmap="YlOrRd"),
                     use_container_width=True, height=400)
        st.download_button("⬇️ Download Machine Report CSV",
                           disp.to_csv(index=False).encode(),
                           "machine_report.csv", "text/csv", key="dl_mach")

        # Deep-dive from machine view
        st.markdown("#### 🔬 Deep Dive — Single Log")
        row_files = df_m["file"].tolist()
        row_labels = (df_m["patient_id"] + " | " + df_m["date"].astype(str)
                      + " | " + df_m["arc"]).tolist()
        if row_files:
            sel_log = st.selectbox("Select log to inspect", row_labels, key="mach_dd")
            chosen_file = row_files[row_labels.index(sel_log)]
            if st.button("Load Deep Dive", key="mach_dd_btn"):
                render_deep_dive(tlog_path=chosen_file, label=sel_log)

    # ─────────────────────────────────────────────────────────────────
    # TAB 2 — PATIENT LOG BOOK
    # ─────────────────────────────────────────────────────────────────
    with tab_pat:
        st.markdown("### 👤 Patient Log Book")

        # Search / filter by patient ID
        all_pids = sorted(df["patient_id"].unique().tolist())
        search   = st.text_input("🔍 Search Patient ID", placeholder="e.g. 291492")
        if search:
            all_pids = [p for p in all_pids if search in p]
        if not all_pids:
            st.warning("No patients match the search.")
            st.stop()

        sel_pid = st.selectbox("Select Patient", all_pids)
        pat_df  = df[df["patient_id"] == sel_pid].sort_values("datetime")

        # Patient summary cards
        n_fracs = len(pat_df)
        n_mach  = pat_df["machine"].nunique()
        avg_rms = pat_df["overall_rms"].mean()
        pat_pass = (pat_df["status"] == "✅ PASS").mean() * 100
        p1, p2, p3, p4 = st.columns(4)
        p1.markdown(metric_card("Fractions Logged", str(n_fracs)),    unsafe_allow_html=True)
        p2.markdown(metric_card("Machines Used",    str(n_mach)),     unsafe_allow_html=True)
        p3.markdown(metric_card("Mean RMS",         f"{avg_rms:.4f} mm"), unsafe_allow_html=True)
        p4.markdown(metric_card("Pass Rate",        f"{pat_pass:.0f}%"), unsafe_allow_html=True)

        # Fraction-by-fraction RMS trend
        st.markdown("#### 📈 Fraction RMS History")
        st.plotly_chart(fraction_trend_chart(df, sel_pid), use_container_width=True)

        # Per-fraction log table
        st.markdown("#### 📋 Fraction Log")
        frac_disp = pat_df[["date","machine","arc","plan",
                              "overall_rms","worst_rms","worst_leaf","MCS","status"]].copy()
        frac_disp = frac_disp.rename(columns={
            "date":"Date","machine":"Machine","arc":"Arc","plan":"Plan",
            "overall_rms":"Overall RMS (mm)","worst_rms":"Worst RMS (mm)",
            "worst_leaf":"Worst Leaf","MCS":"MCS","status":"Status"})
        st.dataframe(
            frac_disp.style.background_gradient(subset=["Overall RMS (mm)"], cmap="YlOrRd")
                           .applymap(lambda v: "color:#f78166" if v == "❌ REVIEW" else "",
                                     subset=["Status"]),
            use_container_width=True)
        st.download_button(f"⬇️ Download Patient {sel_pid} Report CSV",
                           frac_disp.to_csv(index=False).encode(),
                           f"patient_{sel_pid}_report.csv", "text/csv",
                           key="dl_pat")

        # Deep-dive for a specific fraction
        st.markdown("#### 🔬 Deep Dive — Fraction")
        frac_labels = (pat_df["date"].astype(str) + " | "
                       + pat_df["machine"] + " | " + pat_df["arc"]).tolist()
        frac_files  = pat_df["file"].tolist()
        sel_frac = st.selectbox("Select fraction", frac_labels, key="pat_dd")
        chosen_frac_file = frac_files[frac_labels.index(sel_frac)]
        if st.button("Load Fraction Deep Dive", key="pat_dd_btn"):
            render_deep_dive(tlog_path=chosen_frac_file, label=f"Patient {sel_pid} – {sel_frac}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown(
        "<h1 style='text-align:center'>⚛️ Varian Trajectory Log Analyzer</h1>"
        "<p style='text-align:center;color:#8b949e'>Department-wide MLC performance "
        "monitoring — manual upload or automatic folder scan</p>",
        unsafe_allow_html=True)
    st.divider()

    mode = sidebar()

    if mode == "📂 Manual Upload":
        manual_upload_mode()
    else:
        folder_scan_mode()


if __name__ == "__main__":
    main()
