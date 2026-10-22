"""Figure: accuracy of the template fit versus window size and isotope fraction."""

import numpy as np

from common import COLORS, MUON_ONLY, PAGE, plt, run_windows, save, setup, with_classifier

config, templates, edges, test_values = setup()
SIZES = [50, 100, 200, 500, 1000, 2000, 5000]
FRACTIONS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
PER_CLASS = 100


def accuracies(**overrides):
    scan_config = with_classifier(config, windows_per_class=PER_CLASS, **overrides)
    hits = {}
    for truth, kind, result, _ in run_windows(templates, test_values, scan_config):
        if kind != "pure" or truth == MUON_ONLY:
            hits.setdefault(truth, {}).setdefault(kind, []).append(result["label"] == truth)
    return {
        truth: np.mean(kinds["mixed"] if "mixed" in kinds else kinds["pure"])
        for truth, kinds in hits.items()
    }


by_size = [accuracies(window_size=size) for size in SIZES]
by_fraction = [accuracies(mixed_fraction=fraction) for fraction in FRACTIONS]

fig, axes = plt.subplots(1, 2, figsize=(PAGE, 2.4), sharey=True)
panels = (
    (axes[0], SIZES, by_size, "pairs per window", "log"),
    (axes[1], [100 * f for f in FRACTIONS], by_fraction, "isotope share of the window [%]", "log"),
)
for ax, xs, results, xlabel, scale in panels:
    for truth in [MUON_ONLY] + [i for i in COLORS if i != "cosmic-muons"]:
        color = COLORS["cosmic-muons" if truth == MUON_ONLY else truth]
        ax.plot(xs, [100 * r[truth] for r in results], marker="o", ms=3, color=color, label=truth)
    ax.set_xscale(scale)
    ax.set_xticks(xs, [f"{x:g}" for x in xs])
    ax.minorticks_off()
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, 102)
axes[0].set_ylabel("accuracy [%]")
axes[0].text(0.97, 0.04, f"isotope share {100 * config['classifier']['mixed_fraction']:.0f}%", transform=axes[0].transAxes, fontsize=7, ha="right")
axes[1].text(0.97, 0.04, f"{config['classifier']['window_size']} pairs per window", transform=axes[1].transAxes, fontsize=7, ha="right")
axes[1].legend(loc="center right", ncol=1)
save(fig, "scan")
for size, result in zip(SIZES, by_size):
    print("size", size, {k: round(float(v), 3) for k, v in result.items()})
for fraction, result in zip(FRACTIONS, by_fraction):
    print("fraction", fraction, {k: round(float(v), 3) for k, v in result.items()})
