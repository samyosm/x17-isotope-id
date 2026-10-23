"""The scintillator hits of a simulation file: bar, deposited energy and z.

The simulation writes two trees: hit_truth (one row per bar hit: bar, energy, z) and vf48 (two rows
per hit, with the simulated photoelectrons). Only the truth is used here: the simulated optics
attenuate the light much faster than the real bars (see x17-data-analysis/SIMULATION.md), so the
detector response is rebuilt from the truth by response.py.
"""

import numpy as np


def read_hits(path, max_hits=None, seed=0):
    """Return {"bar": scintillator index 0..15, "edep": MeV, "z": cm} of a simulation file.

    A random subset of max_hits hits is kept if the file has more.
    """
    import uproot

    with uproot.open(path) as file:
        hits = file["hit_truth"].arrays(["bar", "edep_MeV", "z_cm"], library="np")
    bar = hits["bar"].astype(np.int16) - 1  # the simulation numbers the bars from 1
    edep = hits["edep_MeV"].astype(np.float32)
    z = hits["z_cm"].astype(np.float32)
    if max_hits is not None and len(bar) > max_hits:
        keep = np.random.default_rng(seed).choice(len(bar), size=max_hits, replace=False)
        bar, edep, z = bar[keep], edep[keep], z[keep]
    return {"bar": bar, "edep": edep, "z": z}
