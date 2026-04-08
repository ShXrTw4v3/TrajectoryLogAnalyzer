# ⚛️ Linac QA Suite

A Streamlit web app for department-wide radiotherapy machine QA — built on **pylinac**.  
Supports **Halcyon**, **TrueBeam**, **Ethos, and all standard Varian C-arm linacs.

---

## Modules

| # | Module | Input | Key outputs |
|---|--------|-------|-------------|
| 1 | 📊 **Trajectory Log Analyzer** | `.bin` DMLC logs | Per-leaf RMS error, MCS, gravity sag, patient log book |
| 2 | 🔲 **Picket Fence** | `.dcm` EPID image | Pass/fail overlay, worst-leaf table, % passing |
| 3 | 🎯 **Winston-Lutz** | `.zip` of EPID DICOMs | 3D isocenter diameter, 2D offsets, scatter plot |
| 4 | 🔬 **CBCT (CatPhan)** | `.zip` of CBCT slices | HU linearity, uniformity index, MTF 50%, phantom images |

All three QA modules include a **🧪 Use pylinac demo** checkbox — tick it to run a full analysis with no file upload (demo files are downloaded once then cached).

---

## Features

### Trajectory Log Analyzer
- **Manual Upload** — drag-and-drop one or many `.bin` files for batch analysis
- **Folder Scan** — configure network log directories per machine; auto-scans for new files and caches results between sessions
- **Per-leaf RMS error** bar chart + worst-leaf dataframe
- **Modulation Complexity Score (MCS = LSV × AAV)** — ⚠️ warning banner when MCS < 0.20 (elevated PSQA failure risk)
- **Gravity Sag (Antigravity) analysis** — polar + cartesian plots of MLC error vs. gantry angle
- **MLC error heatmap** (leaf × snapshot)
- **📡 Machine Dashboard** — per-machine RMS trend, MCS comparison, worst-leaf heatmap over time
- **👤 Patient Log Book** — fraction-by-fraction RMS history per patient, grouped by machine
- **CSV export** at every level

### Picket Fence
- Adjustable tolerance and action tolerance sliders (0.1–0.5 mm)
- Analysed image with pass/fail overlay
- Worst-performing leaf table (sorted by error, colour-coded)

### Winston-Lutz
- 3D isocenter diameter with ≤ 1 mm clinical tolerance check
- Pylinac summary plot + interactive Plotly 3D scatter of CAX→BB vectors

### CBCT (CatPhan)
- Supports CatPhan 503, 504, 600, 604
- HU linearity table (measured vs. nominal, ΔHU per ROI)
- Summary cards: HU R², uniformity index, MTF 50%, geometric mean, low-contrast ROIs
- Phantom analysis image grid

---

## Quick Start

```bash
# 1. Install dependencies (once)
pip install -r requirements.txt

# 2. Run
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## TBox / Network Deployment

1. Run **`install.bat`** once — installs dependencies and creates a desktop shortcut
2. Double-click **`launch.bat`** (or the shortcut) to start the server
3. *(Optional)* Run **`setup_autostart.bat`** as Administrator to start automatically on boot

Other PCs on the same network can reach the app at `http://<TBox-hostname>:8501`.

---

## Project Structure

```
LogAnalyzer/
├── app.py                 # Main application (all 4 modules)
├── requirements.txt       # Python dependencies
├── install.bat            # One-time TBox installer
├── launch.bat             # Day-to-day launcher
├── setup_autostart.bat    # Windows Task Scheduler auto-start (optional, run as Admin)
├── machines.json          # Machine config — auto-created, gitignored
└── results_cache.json     # Analysis cache — auto-created, gitignored
```

---

## Dependencies

- [pylinac](https://pylinac.readthedocs.io/) — log parsing, PF, WL, CatPhan analysis
- [Streamlit](https://streamlit.io/) — web UI framework
- [Plotly](https://plotly.com/python/) — interactive charts
- [Matplotlib](https://matplotlib.org/) — pylinac image rendering
- NumPy, Pandas

---

## MCS Methodology

`MCS = LSV × AAV` — adapted from McNiven et al., *Med. Phys.* 2010.

| Component | Definition | Low value means… |
|-----------|-----------|-----------------|
| **LSV** (Leaf Sequence Variability) | Normalised inter-snapshot leaf displacement | Highly modulated |
| **AAV** (Aperture Area Variation) | CoV of open aperture width | Rapidly changing apertures |

- MCS → 1.0: simple open field. MCS → 0: highly complex IMRT/VMAT.
- **Clinical threshold: MCS < 0.20 triggers a ⚠️ PSQA review warning.**

---

## License

MIT
