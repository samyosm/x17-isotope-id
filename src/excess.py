"""Source identification by excess over a reference period.

The input is what the registry counts in ClickHouse for two periods, a window and a reference:
the number of events (random triggers: the exposure) and, per bar, the histogram of
sqrt((U-PU)(D-PD)) of the pairs with both PMTs at or above a low threshold. A source shows as an
excess in the window over the reference scaled by the exposure.

1. Per bar: pairs, expected from the reference, excess, significance. Only bars with two good PMTs.
2. Which source: for each source shape (from the simulation's truth through the measured response,
   see response.py) and each candidate position of the source (a bar), the expected window counts are
   background + amplitude(distance) * shape(distance, bar). The amplitude is a free number per
   distance class, constrained not to grow with the distance from the source. The real source
   placement is wider than the simulated one, so the geometry (the amplitudes) is a nuisance and only
   the SHAPE separates the isotopes. The background of every bin is a free nuisance parameter fixed by
   both the window and the reference (the "on/off" likelihood: the reference is a measurement with its
   own Poisson uncertainty, not a known number), profiled out in closed form. The test statistic
   TS = 2 ln [L(best) / L(no source)] of the best position is the score of the source.
3. Verdict: "none" below detection_ts; "identified" if the best source beats every other by margin_ts;
   otherwise "detected" with the candidates within the margin.
"""

import json
from pathlib import Path

import numpy as np

from response import CLASSES, distance_class

BARS = 16


class NotEnoughEvents(ValueError):
    pass


def _xlogy(n, x):
    """n * ln(x), taken as 0 where n is 0."""
    return np.where(n > 0, n * np.log(np.maximum(x, 1e-300)), 0.0)


def profile_loglike(n_window, n_reference, alpha, signal):
    """ln L of the window and reference counts with the background of every bin profiled out.

    n_window ~ Poisson(b + signal), n_reference ~ Poisson(b / alpha), with alpha the ratio of the
    window to the reference exposure. Returns (ln L, d ln L / d signal) arrays-summed and per bin.
    """
    k = 1.0 + 1.0 / alpha
    t = n_window + n_reference - k * signal
    b = (t + np.sqrt(t * t + 4.0 * k * n_reference * signal)) / (2.0 * k)
    lnl = (_xlogy(n_window, b + signal) - (b + signal) + _xlogy(n_reference, b / alpha) - b / alpha).sum()
    gradient = np.where(b + signal > 0, n_window / np.maximum(b + signal, 1e-300), 0.0) - 1.0
    return float(lnl), gradient


class ExcessAnalyzer:
    def __init__(self, config, templates=None):
        ex = config["excess"]
        self.threshold = float(ex["threshold"])
        self.bins = int(ex["bins"])
        self.high = float(ex["max"])
        self.edges = np.linspace(0.0, self.high, self.bins + 1)
        self.detection_ts = float(ex["detection_ts"])
        self.margin_ts = float(ex["margin_ts"])
        self.min_events = int(ex["min_events"])
        self.bad_channels = set(ex.get("bad_channels", []))
        self.pedestals = {k: float(v) for k, v in ex["pedestals"].items()}
        self.activity_bq = {k: float(v) for k, v in ex.get("activity_bq", {}).items()}
        self.min_activity_bq = float(ex.get("min_plausible_activity_bq", 0.0))
        self.templates = templates

    @classmethod
    def from_config(cls, config):
        """The analyzer of a config with an [excess] section (None without); templates are read if present."""
        if "excess" not in config:
            return None
        path = Path(config.get("base_dir", ".")) / config["excess"]["templates"]
        templates = json.loads(path.read_text()) if path.exists() else None
        return cls(config, templates)

    def pedestal(self, field):
        return self.pedestals.get(field, 0.0)

    def _shapes(self, source, centre, bars):
        """(classes, shapes) of the given bars for a source at the bar `centre`."""
        table = self.templates["sources"][source]["shapes"]
        classes = distance_class(np.array(bars), centre)
        shapes = np.array([table[str(c)][str(b)] for c, b in zip(classes, bars)], dtype=float)
        return classes, shapes

    def _fit(self, n_window, n_reference, alpha, classes, shapes):
        """Best amplitudes (nonincreasing with the distance class) and the test statistic."""
        from scipy.optimize import minimize

        present = np.bincount(classes, minlength=CLASSES) > 0
        zero_lnl, _ = profile_loglike(n_window, n_reference, alpha, np.zeros_like(shapes))
        # amplitude of class o = sum of the steps delta_j for j >= o, every step >= 0
        triangle = np.triu(np.ones((CLASSES, CLASSES)))  # triangle[o, j] = 1 for j >= o

        def objective(delta):
            a = triangle @ delta
            lnl, gradient = profile_loglike(n_window, n_reference, alpha, a[classes][:, None] * shapes)
            per_class = np.bincount(classes, weights=(gradient * shapes).sum(axis=1), minlength=CLASSES)
            return -lnl, -(triangle.T @ per_class)

        start = np.full(CLASSES, 0.5)
        result = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=[(0, None)] * CLASSES)
        a = np.where(present, triangle @ result.x, 0.0)
        return a, max(2.0 * (-result.fun - zero_lnl), 0.0)

    def analyze(self, groups, window, reference):
        """Analyse a window against a reference.

        groups: {bar name: (D channel, U channel)}. window and reference: {"events": n, "histograms":
        {bar name: counts per bin}}. Returns a JSON-serialisable dict (bars, detection, sources, verdict, spectrum).
        """
        n_w, n_r = int(window["events"]), int(reference["events"])
        if min(n_w, n_r) < self.min_events:
            raise NotEnoughEvents(
                f"Too few events: {n_w} in the window and {n_r} in the reference (at least {self.min_events} each)"
            )
        scale = n_w / n_r
        bars, rows, usable = sorted(int(b) for b in groups), [], []
        counts, expected, ref_counts = {}, {}, {}
        for b in bars:
            d_field, u_field = groups[str(b)]
            bad = [f for f in (d_field, u_field) if f in self.bad_channels]
            hw = np.asarray(window["histograms"][str(b)], float)
            hr = np.asarray(reference["histograms"][str(b)], float)
            mu = hr * scale
            pairs, exp_total = float(hw.sum()), float(mu.sum())
            sigma = float(np.sqrt(max(exp_total + hr.sum() * scale**2, 1.0)))
            rows.append({
                "bar": b,
                "usable": not bad,
                "reason": None if not bad else "PMT " + ", ".join(bad) + " is dead or too noisy",
                "pairs": int(pairs),
                "expected": exp_total,
                "excess": pairs - exp_total,
                "z": (pairs - exp_total) / sigma,
            })
            if not bad:
                usable.append(b)
                counts[b], expected[b], ref_counts[b] = hw, mu, hr
        good = [r for r in rows if r["usable"]]
        total_excess = sum(r["excess"] for r in good)
        # variance of the total: window Poisson (= expected under no source) plus the scaled reference
        variance = sum(max(r["expected"], 0.0) + (r["expected"] / scale) * scale**2 for r in good)
        rate_z = total_excess / np.sqrt(max(variance, 1.0))
        result = {
            "bars": rows,
            "events": {"window": n_w, "reference": n_r, "scale": scale},
            "detection": {
                "pairs": int(sum(r["pairs"] for r in good)),
                "expected": sum(r["expected"] for r in good),
                "excess": total_excess,
                "rate_z": float(rate_z),
                "bars_above_3": int(sum(r["z"] > 3 for r in good)),
            },
            "sources": [],
            "edges": self.edges.tolist(),
        }
        if self.templates is None:
            detected = rate_z >= 5.0
            result["verdict"] = {
                "code": "detected" if detected else "none",
                "source": None, "candidates": [], "centre": None,
                "warning": "No source shapes are installed: only the pair rate was tested.",
            }
            result["spectrum"] = self._spectrum(usable, counts, expected, None)
            return result

        stack_w = np.array([counts[b] for b in usable])
        stack_r = np.array([ref_counts[b] for b in usable])
        for name in self.templates["sources"]:
            best = None
            for centre in range(BARS):
                classes, shapes = self._shapes(name, centre, usable)
                a, ts = self._fit(stack_w, stack_r, scale, classes, shapes)
                if best is None or ts > best["ts"]:
                    best = {"name": name, "ts": float(ts), "centre": centre,
                            "amplitudes": {str(k): float(v) for k, v in enumerate(a) if v > 0},
                            "plausible": self.activity_bq.get(name, np.inf) >= self.min_activity_bq,
                            "activity_bq": self.activity_bq.get(name)}
            result["sources"].append(best)
        result["sources"].sort(key=lambda s: -s["ts"])
        result["verdict"] = self._verdict(result["sources"])
        centre = result["verdict"]["centre"]
        result["spectrum"] = self._spectrum(usable, counts, expected, centre)
        return result

    def _verdict(self, ranked):
        best = ranked[0]
        if best["ts"] < self.detection_ts:
            return {"code": "none", "source": None, "candidates": [], "centre": None, "warning": None}
        close = [s for s in ranked if best["ts"] - s["ts"] < self.margin_ts]
        plausible = [s for s in close if s["plausible"]]
        warning = None
        if len(close) == 1 and best["plausible"]:
            code, source = "identified", best["name"]
        else:
            code, source = "detected", None
            if not best["plausible"]:
                warning = (f"{best['name']} fits best but its activity today is only {best['activity_bq']:.0f} Bq: "
                           "unlikely to be seen here")
        return {"code": code, "source": source,
                "candidates": [s["name"] for s in (plausible or close)],
                "centre": best["centre"], "warning": warning}

    def _spectrum(self, usable, counts, expected, centre):
        """Window and expected spectrum summed over the bars within two bars of the source (all bars if unknown)."""
        if centre is None:
            chosen = usable
        else:
            chosen = [b for b in usable if distance_class(b, centre) <= 2]
        window = np.sum([counts[b] for b in chosen], axis=0) if chosen else np.zeros(self.bins)
        exp = np.sum([expected[b] for b in chosen], axis=0) if chosen else np.zeros(self.bins)
        return {"bars": chosen, "window": window.tolist(), "expected": exp.tolist()}
