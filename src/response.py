"""The detector response applied to the truth of a simulation, and the source shapes built with it.

Per hit (bar b, deposited energy E in MeV, position z in cm):
  - light: N = n0 * E * exp(-+z / lambda_b) photoelectrons for the D and the U PMT (Poisson), plus a
    single-photoelectron spread;
  - amplitude: ADC = adc_scale * f * g_b * N, with g_b = 1 / (gamma_b * n0 / 1000) so that
    sqrt(D * U) = E / gamma_b at z = 0, and the U/D ratio alpha0_b;
  - f, the recorded fraction of the pulse (kernel): the stored value is not always the full amplitude;
  - the noise of a real noise-only event (the D and U readings of one event are added);
  - the pair selection of the analysis: both PMTs at or above the threshold.
All parameters are in config.toml ([calibration.gamma], [response], [excess]).
"""

import numpy as np

CLASSES = 9  # distance classes from the source: 0 (in front of it) to 8 (the opposite bar)


def recorded_fraction(kernel, rng, n):
    """n recorded fractions f in (0, 1] for the kernel named like "full", "uniform", "exp:X" or "mix:A:X".

    full: 1. uniform: U(0, 1). exp:X: exp(-X u), u ~ U(0, 1). mix:A:X: with probability A the full
    amplitude (3 % spread), otherwise exp:X.
    """
    kind, *args = kernel.split(":")
    if kind == "full":
        return np.ones(n)
    if kind == "uniform":
        return rng.random(n)
    if kind == "exp":
        return np.exp(-float(args[0]) * rng.random(n))
    if kind == "mix":
        head = rng.random(n) < float(args[0])
        tail = np.exp(-float(args[1]) * rng.random(n))
        return np.where(head, np.clip(rng.normal(1.0, 0.03, n), 0.5, 1.2), tail)
    raise ValueError(f"unknown kernel {kernel!r}")


def distance_class(bar, centre):
    """Circular distance, in bars, between two of the 16 bars."""
    delta = (np.asarray(bar) - centre) % 16
    return np.minimum(delta, 16 - delta)


class Response:
    def __init__(self, config):
        r = config["response"]
        self.n0 = float(r["n0"])
        self.spe = float(r["spe_relative_sigma"])
        self.kernel = r["kernel"]
        self.adc_scale = float(r["adc_scale"])
        self.gamma = {int(k): float(v) for k, v in config["calibration"]["gamma"].items()}
        self.alpha0 = {int(k): float(v) for k, v in r["alpha0"].items()}
        self.lambda_cm = {int(k): float(v) for k, v in r["lambda_cm"].items()}

    def signal(self, bar, edep, z, rng):
        """The D and U amplitudes (ADC, no noise) of hits in the real bar `bar`."""
        e, zz = np.asarray(edep, float), np.asarray(z, float)
        n_d = rng.poisson(self.n0 * e * np.exp(-zz / self.lambda_cm[bar]))
        n_u = rng.poisson(self.n0 * e * np.exp(+zz / self.lambda_cm[bar]))
        n_d = n_d + self.spe * np.sqrt(n_d) * rng.standard_normal(len(e))
        n_u = n_u + self.spe * np.sqrt(n_u) * rng.standard_normal(len(e))
        f = recorded_fraction(self.kernel, rng, len(e))
        g = self.adc_scale * f / (self.gamma[bar] * self.n0 / 1000)
        return g * n_d / np.sqrt(self.alpha0[bar]), g * n_u * np.sqrt(self.alpha0[bar])


def pair_values(d, u, pedestal_d, pedestal_u):
    """sqrt((U - PU)(D - PD)), negative differences counting as 0."""
    return np.sqrt(np.maximum(d - pedestal_d, 0) * np.maximum(u - pedestal_u, 0))


def build_templates(config, hits, noise, columns, bars, per_class=200_000, seed=3):
    """The shape of a source's pair spectrum for every distance class and every real bar.

    hits: {"bar", "edep", "z"} of one simulated source, the source in front of bar index 0.
    noise: array (events, channels) of noise-only events; columns: the channel name of each column.
    bars: {bar: (D channel, U channel)} of the bars to build.
    Returns {"shapes": {class: {bar: [bins]}}, "pairs_per_hit": {class: {bar: float}}}, each shape
    normalised to a sum of 1 (the amplitude is fitted on the data).
    """
    ex, resp = config["excess"], Response(config)
    edges = np.linspace(0.0, float(ex["max"]), int(ex["bins"]) + 1)
    threshold = float(ex["threshold"])
    pedestal = {k: float(v) for k, v in ex["pedestals"].items()}
    column = {name: i for i, name in enumerate(columns)}
    rng = np.random.default_rng(seed)
    classes = distance_class(hits["bar"], 0)
    shapes, yields = {}, {}
    for c in range(CLASSES):
        index = np.flatnonzero((classes == c) & (hits["edep"] > 0.02))
        if len(index) > per_class:
            index = rng.choice(index, per_class, replace=False)
        shapes[str(c)], yields[str(c)] = {}, {}
        for bar, (d_channel, u_channel) in bars.items():
            e, z = hits["edep"][index], hits["z"][index]
            sig_d, sig_u = resp.signal(bar, e, z, rng)
            pick = rng.integers(0, len(noise), len(index))
            raw_d = sig_d + noise[pick, column[d_channel]]
            raw_u = sig_u + noise[pick, column[u_channel]]
            keep = (raw_d >= threshold) & (raw_u >= threshold)
            v = pair_values(raw_d, raw_u, pedestal[d_channel], pedestal[u_channel])[keep]
            histogram = np.histogram(v, bins=edges)[0].astype(float)
            total = histogram.sum()
            shapes[str(c)][str(bar)] = (histogram / total).tolist() if total > 0 else [0.0] * (len(edges) - 1)
            yields[str(c)][str(bar)] = float(keep.mean()) if len(index) else 0.0
    return {"shapes": shapes, "pairs_per_hit": yields}
