"""Figures and numbers: confusion matrix, Delta distributions, accuracy table."""

import numpy as np

from common import (
    COLORS,
    COLUMN,
    GENERATED,
    INK,
    INK_SOFT,
    MUON_ONLY,
    plt,
    run_windows,
    save,
    setup,
)

config, templates, edges, test_values = setup()
clf = config["classifier"]
rows = run_windows(templates, test_values, config)

labels = [MUON_ONLY] + [i for i in templates if i != clf["muon_isotope"]]
index = {name: k for k, name in enumerate(labels)}
confusion = np.zeros((len(labels), len(labels)), dtype=int)
for truth, _, result, _ in rows:
    confusion[index[truth], index[result["label"]]] += 1

fig, ax = plt.subplots(figsize=(COLUMN, 2.7))
shares = confusion / confusion.sum(axis=1, keepdims=True)
ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
ax.grid(False)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(
            j, i, confusion[i, j], ha="center", va="center", fontsize=7,
            color="white" if shares[i, j] > 0.55 else INK,
        )
ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
ax.set_yticks(range(len(labels)), labels)
ax.set_xlabel("predicted")
ax.set_ylabel("true")
ax.spines[:].set_visible(False)
ax.tick_params(length=0)
save(fig, "confusion")

groups = {
    "muon windows": [r[2]["delta"] for r in rows if r[0] == MUON_ONLY],
    "isotope + muons": [r[2]["delta"] for r in rows if r[0] != MUON_ONLY and r[1] == "mixed"],
    "isotope only": [r[2]["delta"] for r in rows if r[0] != MUON_ONLY and r[1] == "pure"],
}
styles = {
    "muon windows": (COLORS["cosmic-muons"], "-"),
    "isotope + muons": (INK_SOFT, "--"),
    "isotope only": (INK, "-"),
}
fig, ax = plt.subplots(figsize=(COLUMN, 2.4))
for name, deltas in groups.items():
    color, style = styles[name]
    values = np.sort(np.clip(deltas, 0.1, None))
    ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", color=color, ls=style, label=name)
ax.axvline(clf["detection_threshold"], color=INK_SOFT, lw=0.8, ls=":")
ax.text(clf["detection_threshold"] * 1.15, 0.9, "threshold", fontsize=7, color=INK_SOFT)
ax.set_xscale("log")
ax.set_xlim(0.1, 1e5)
ax.set_ylim(0, 1.02)
ax.set_xlabel(r"$\Delta = 2\,(\mathrm{NLL}_{\mathrm{muon}} - \mathrm{NLL}_{\mathrm{best}})$")
ax.set_ylabel("fraction of windows with $\\Delta$ below")
ax.legend(loc="lower left")
save(fig, "delta")

kinds = {}
for truth, kind, result, _ in rows:
    kinds.setdefault((truth, kind), []).append(result["label"] == truth)

lines = []
for name in labels:
    pure = np.mean(kinds[(name, "pure")])
    mixed = np.mean(kinds[(name, "mixed")]) if (name, "mixed") in kinds else None
    mixed_text = "--" if mixed is None else f"{100 * mixed:.1f}"
    lines.append(f"{name} & {100 * pure:.1f} & {mixed_text} \\\\")

muon_deltas = np.array(groups["muon windows"])
macros = {
    "AccOverall": f"{100 * confusion.trace() / confusion.sum():.1f}",
    "NWindows": str(len(rows)),
    "WindowSize": str(clf["window_size"]),
    "WindowsPerClass": str(clf["windows_per_class"]),
    "MixedPercent": f"{100 * clf['mixed_fraction']:.0f}",
    "Threshold": f"{clf['detection_threshold']:g}",
    "MuonDeltaMean": f"{muon_deltas.mean():.1f}",
    "MuonDeltaNinetyFive": f"{np.percentile(muon_deltas, 95):.1f}",
    "MuonDeltaMax": f"{muon_deltas.max():.1f}",
    "MixedDeltaMin": f"{min(groups['isotope + muons']):.0f}",
    "PureDeltaMin": f"{min(groups['isotope only']):.0f}",
    "FalseAlarms": str(int(np.sum(muon_deltas > clf["detection_threshold"]))),
}
GENERATED.mkdir(parents=True, exist_ok=True)
(GENERATED / "accuracy_table.tex").write_text("\n".join(lines) + "\n")
(GENERATED / "macros.tex").write_text(
    "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items())
)
print(macros)
print("\n".join(lines))
