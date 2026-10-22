"""Figure: one 2000-pair window (20% Cs-137, 80% muons) with its template fit."""

import numpy as np

from common import (
    COLORS,
    COLUMN,
    build_windows,
    classify,
    plt,
    save,
    setup,
    with_classifier,
)

config, templates, edges, test_values = setup()
clf = config["classifier"]
config = with_classifier(config, windows_per_class=1)
isotope = "Cs-137"

window = next(
    values
    for truth, kind, values in build_windows(test_values, clf["muon_isotope"], config)
    if truth == isotope and kind == "mixed"
)
counts = np.histogram(window, bins=edges)[0]
result = classify(counts, templates, clf["muon_isotope"], clf["detection_threshold"])
f, total = result["fraction"], counts.sum()
muon = templates[clf["muon_isotope"]]
fitted = total * ((1 - f) * muon + f * templates[result["best"]])
centers = 0.5 * (edges[1:] + edges[:-1])

fig, ax = plt.subplots(figsize=(COLUMN, 2.4))
ax.step(centers, counts, where="mid", color="#9a9992", lw=0.8, label="window (data)")
ax.plot(centers, total * muon, color=COLORS["cosmic-muons"], ls="--", label="muons only (f = 0)")
ax.plot(centers, fitted, color=COLORS[result["best"]], label=f"fit: {result['best']}, f = {f:.2f}")
ax.set_yscale("log")
ax.set_ylim(0.05, 1000)
ax.set_xlim(edges[0], edges[-1])
ax.set_xlabel(r"pair value $x$ [ADC counts]")
ax.set_ylabel("pairs per bin")
ax.legend(loc="upper right")
save(fig, "fit_example")
print(f"example window: truth {isotope}, label {result['label']}, f={f:.3f}, delta={result['delta']:.1f}")
