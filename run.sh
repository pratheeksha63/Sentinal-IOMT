#!/bin/bash
pip install torch numpy pandas scikit-learn shap matplotlib
python sim.py --visual
python sim.py --grid
python plots.py
