"""Figure: the reference templates (normalized spectra) of every source."""

import numpy as np

from common import COLORS, COLUMN, plt, save, setup

config, templates, edges, _ = setup()
centers = 0.5 * (edges[1:] + edges[:-1])

fig, ax = plt.subplots(figsize=(COLUMN, 2.4))
for isotope, template in templates.items():
    ax.step(centers, template, where="mid", color=COLORS[isotope], label=isotope)
ax.set_yscale("log")
ax.set_xlim(edges[0], edges[-1])
ax.set_ylim(1e-6, 1e-1)
ax.set_xlabel(r"pair value $x=\sqrt{(U-P_U)(D-P_D)}$ [ADC counts]")
ax.set_ylabel("fraction of pairs per bin")
ax.legend(ncol=2, loc="upper right")
save(fig, "spectra")
