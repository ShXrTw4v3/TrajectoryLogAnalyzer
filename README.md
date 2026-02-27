# ⚛️ Varian Trajectory Log Analyzer

A Streamlit web app for department-wide analysis of Varian DMLC trajectory logs.  
Supports **Halcyon**, **TrueBeam**, **Ethos**, and standard C-arm linacs.

---

## Features

- **Manual Upload** — drag-and-drop one or many `.bin` files for immediate analysis
- **Folder Scan** — point to network log directories per machine; auto-scans for new files and caches results
- **Per-leaf RMS error** analysis with bar chart and worst-leaf table
- **Modulation Complexity Score (MCS)** — custom LSV × AAV implementation with clinical warning when MCS < 0.20
- **Gravity Sag (Antigravity) Analysis** — polar + cartesian plot of MLC error vs. gantry angle to detect gravitational mechanical drift at 90°/270°
- **MLC error heatmap** (leaf × snapshot)
- **Machine Dashboard** — per-machine RMS trend, MCS comparison, worst-leaf heatmap over time
- **Patient Log Book** — fraction-by-fraction RMS history per patient, grouped by machine
- **CSV export** at every level

## Quick Start

```bash
# 1. Install dependencies (once)
pip install -r requirements.txt

# 2. Run
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## TBox / Network Deployment

1. Run **`install.bat`** once — installs dependencies and creates a desktop shortcut
2. Double-click **`launch.bat`** (or the shortcut) to start
3. Optional: run **`setup_autostart.bat`** as Administrator to auto-start on boot

Other PCs on the network can access the app at `http://<TBox-hostname>:8501`.

## Project Structure

```
LogAnalyzer/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── install.bat            # One-time TBox installer
├── launch.bat             # Day-to-day launcher
├── setup_autostart.bat    # Windows Task Scheduler auto-start (optional)
├── machines.json          # Machine config (auto-created, gitignored)
└── results_cache.json     # Analysis cache (auto-created, gitignored)
```

## Dependencies

- [pylinac](https://pylinac.readthedocs.io/) — trajectory log parsing
- [Streamlit](https://streamlit.io/) — web UI
- [Plotly](https://plotly.com/python/) — interactive charts
- NumPy, Pandas

## MCS Methodology

`MCS = LSV × AAV` — adapted from McNiven et al., *Med. Phys.* 2010.

- **LSV** (Leaf Sequence Variability): normalised inter-snapshot leaf displacement. Lower = more modulated.
- **AAV** (Aperture Area Variation): coefficient of variation of the open aperture width. Lower = more varying apertures.
- MCS → 1.0: simple open field. MCS → 0: highly modulated IMRT/VMAT.
- **Clinical threshold: MCS < 0.20 triggers a PSQA review warning.**

## License

MIT
