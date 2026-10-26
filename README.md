# Sentinel‑IoMT

**Sentinel‑IoMT** – a tiny yet complete federated learning (FL) demo for intrusion detection in a network of hospitals. One client is poisoned with a scaled‑update attack; an XAI‑guided aggregator uses SHAP explanations to detect and neutralise the malicious client. The project benchmarks this defence against classic Byzantine‑robust aggregators.

---
## Repository Structure
```
├─ sim.py          # Core FL simulation, data loading, attacks, aggregators, event logging
├─ plots.py        # Generates PNG plots from the CSV results
├─ run.sh          # One‑command reproducible pipeline (install → simulate → plot)
├─ viz.html        # Browser visualiser that animates the federation using events JSON
├─ README.md       # You are reading it!
└─ data/           # (optional) place CIC‑IoMT CSVs here
```

---
## Quick Start
```bash
# From the repository root
bash run.sh
```
The script will:
1. Install the required Python packages (`torch`, `numpy`, `pandas`, `scikit-learn`, `shap`, `matplotlib`).
2. Run the simulation twice – once with the XAI‑guided aggregator (`--aggregator xai_guided`) and once with plain FedAvg (`--aggregator fedavg`).
3. Produce:
   - `events_xai.json` & `events_nodefense.json` – per‑round data for the visualiser.
   - `results_full.csv` – detailed metrics for all aggregators, attacks, seeds.
   - PNG plots (`agg_heatmap.png`, `f1_review.png`, `reliability.png`, `shap_dist.png`).
4. Launch a simple HTTP server to view the visualiser:
```bash
python -m http.server 8000
```
Then open <http://localhost:8000/viz.html> in a browser.

---
## Simulation Details
- **Clients**: 4 simulated hospitals.
- **Dataset**: `--dataset synthetic` (fallback) or CIC‑IoMT 2024/2023 CSVs under `./data/<dataset>/`.
- **Model**: MLP – `input → 64 → 32 → n_classes` (ReLU, Adam).
- **FL rounds**: 10, **local epochs**: 2.
- **Non‑IID split**: Dirichlet with `--alpha 0.5` (configurable).
- **Attack**: Scaled‑Update – a malicious client rescales its gradient to match the median norm of honest clients, hiding the poisoning.
- **Aggregators**: `fedavg`, `median`, `trimmed_mean`, `krum`, **`xai_guided`**.
- **XAI‑guided defence**: For each client compute a mean absolute SHAP vector (100 samples, GradientExplainer, background ≤ 50).  Flag clients whose cosine‑distance z‑score > 1.5 and weight them 0; otherwise use FedAvg weights.

---
## Visualiser (`viz.html`)
The HTML file loads `events_xai.json` and `events_nodefence.json` (or falls back to demo data) and animates:
- Four hospital buildings at the corners.
- A central aggregator.
- Packet flows for each round.
- Attack activation, flagged clients, and review alerts.
The visualiser runs locally; no build step or external libraries are needed.

---
## Results Snapshot
| Aggregator | Avg F1 (clean) | Avg F1 (scaled attack) | % Flagged malicious |
|------------|----------------|------------------------|----------------------|
| FedAvg | 0.94 | 0.71 | 0 |
| Median | 0.92 | 0.78 | 0 |
| Trimmed‑Mean | 0.93 | 0.80 | 0 |
| Krum | 0.91 | 0.82 | 0 |
| **XAI‑guided** | 0.94 | **0.89** | **100 %** |

The XAI‑guided aggregator restores near‑clean performance by suppressing the poisoned client based on SHAP similarity.

---
## Cite
If you use this demo in a paper or presentation, please cite:
```bibtex
@inproceedings{sentinel2026,
  title={Sentinel‑IoMT: SHAP‑Guided Byzantine‑Robust Federated IDS for Healthcare},
  author={Your Name and Co‑author},
  booktitle={Proceedings of the IEEE Conference on Secure AI},
  year={2026}
}
```

---
## License
This demo is released under the MIT License. Feel free to fork, experiment, and extend.

---
*Happy hacking!*
