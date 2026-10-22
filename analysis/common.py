import copy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import load_config
from main import (
    MUON_ONLY,
    build_windows,
    classify,
    load_templates,
    load_test_values,
)

FIGURES = ROOT / "report" / "figures"
GENERATED = ROOT / "report" / "generated"

COLORS = {
    "cosmic-muons": "#2a78d6",
    "Bi-207": "#eb6834",
    "Cs-137": "#1baf7a",
    "Co-60": "#eda100",
    "Na-22": "#e87ba4",
}
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": INK_SOFT,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "lines.linewidth": 1.2,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)

COLUMN = 3.4
PAGE = 7.0


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf")
    plt.close(fig)
    print(f"figure -> {FIGURES / name}.pdf")


def with_classifier(config, **overrides):
    config = copy.deepcopy(config)
    config["classifier"].update(overrides)
    return config


def run_windows(templates, test_values, config):
    clf = config["classifier"]
    rows = []
    for truth, kind, values in build_windows(test_values, clf["muon_isotope"], config):
        counts = np.histogram(values, bins=config["_edges"])[0]
        result = classify(counts, templates, clf["muon_isotope"], clf["detection_threshold"])
        truth_label = MUON_ONLY if truth == clf["muon_isotope"] else truth
        rows.append((truth_label, kind, result, counts))
    return rows


def setup():
    config = load_config()
    templates, edges = load_templates(config)
    config["_edges"] = edges
    return config, templates, edges, load_test_values(config)
