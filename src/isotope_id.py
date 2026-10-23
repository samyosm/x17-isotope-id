"""Library entry point: identify the source near each scintillator from detector readings.

Readings are the rows of the database, one per ADC value: a time, a channel name
(_field, e.g. "FPGA0_ADCA") and the ADC value. A group is one scintillator, given as the
channel of its D-end PMT and the channel of its U-end PMT. For each group the function

  1. keeps the readings of the two channels,
  2. pairs the U and D readings that share the same time (src/pairing.py),
  3. drops the pairs rejected by the acquisition threshold (config filter.*, as in the
     simulation: the stored readings must reach the threshold, "both" or "either"),
  4. subtracts the channel pedestals (config defaults / channels.*) and histograms
     sqrt((U - PU) * (D - PD)) on the template bins (config histogram.*),
  5. classifies that single histogram with the template fit (src/main.py).

Usage:
  identifier = Identifier.from_config()          # loads the templates once
  results = identifier.identify(times, fields, adc, {"3": ("FPGA1_ADCE", "FPGA3_ADCE")})

No gain is applied: the values are used as stored, so the spectra only match the templates
if the detector data has the same scale as the templates.
"""

import math

import numpy as np

from config import load_config
from main import classify, load_templates
from pairing import D_END, U_END, match_pairs, pair_variable

MIN_PAIRS = 100
RAW_START = -10  # raw readings are histogrammed from here, in bins of 1 ADC,
RAW_BINS = 120   # to show where the baseline noise ends


def finite_or_none(value):
    return None if value is None or not math.isfinite(value) else float(value)


def summary(values):
    """Count and percentiles of a 1-d array (None entries when empty)."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"n": 0, "min": None, "p1": None, "p50": None, "p99": None, "max": None,
                "negative": None}
    p1, p50, p99 = np.percentile(values, [1, 50, 99])
    return {
        "n": int(len(values)),
        "min": float(values.min()),
        "p1": float(p1),
        "p50": float(p50),
        "p99": float(p99),
        "max": float(values.max()),
        "negative": int(np.count_nonzero(values < 0)),
    }


def config_defaults(config):
    return config.get("defaults", {})


class Identifier:
    def __init__(self, config):
        self.config = config
        self.templates, self.edges = load_templates(config)
        clf = config["classifier"]
        self.muon_isotope = clf["muon_isotope"]
        self.threshold = clf["detection_threshold"]
        self.min_pairs = MIN_PAIRS
        self.filter = config["filter"]
        self.raw_start = RAW_START
        self.raw_bins = RAW_BINS

    def pedestal(self, field):
        params = config_defaults(self.config) | self.config["channels"].get(field, {})
        return float(params.get("pedestal", 0.0))

    def accepted(self, d_raw, u_raw):
        """Mask of the pairs that pass the acquisition threshold (all, if it is disabled)."""
        if not self.filter["enabled"]:
            return np.ones(len(d_raw), dtype=bool)
        threshold = float(self.filter["threshold"])
        d_ok, u_ok = d_raw >= threshold, u_raw >= threshold
        return d_ok & u_ok if self.filter["mode"] == "both" else d_ok | u_ok

    @classmethod
    def from_config(cls, path=None):
        return cls(load_config(path) if path else load_config())

    def matched_pairs(self, times, fields, adc, d_field, u_field):
        """Stored (U, D) readings with equal times for one scintillator."""
        is_d, is_u = fields == d_field, fields == u_field
        used = is_d | is_u
        end = np.where(is_u[used], U_END, D_END)
        slot = np.zeros(len(end), dtype=np.int64)
        return match_pairs(times[used], end, slot, adc[used])

    def pair_values(self, times, fields, adc, d_field, u_field):
        """sqrt((U - PU) * (D - PD)) of the accepted pairs of one scintillator.

        Returns (values, matched): the values and the number of pairs before the threshold.
        """
        u_raw, d_raw = self.matched_pairs(times, fields, adc, d_field, u_field)
        keep = self.accepted(d_raw, u_raw)
        u_charge = u_raw[keep] - self.pedestal(u_field)
        d_charge = d_raw[keep] - self.pedestal(d_field)
        return pair_variable(u_charge, d_charge), len(u_raw)

    def bin_layout(self):
        """(low, high, bins) of the uniform histogram bins; the database bins on the same layout."""
        edges = np.asarray(self.edges, dtype=np.float64)
        widths = np.diff(edges)
        if not np.allclose(widths, widths[0]):
            raise ValueError("The templates must use uniform bins to be binned in the database")
        return float(edges[0]), float(edges[-1]), len(edges) - 1

    def identify_histograms(self, histograms, min_pairs=None):
        """Classify one histogram per group.

        histograms: {group name: (counts per template bin, accepted pairs)}; the pairs also
        count those outside the histogram range.
        Returns a list, in the order of histograms, of dicts with: group, pairs, in_range
        (pairs inside the histogram range), label (isotope, "muon-only", or None when fewer
        than min_pairs pairs are in range; default self.min_pairs), best, fraction, delta,
        margin and nll (see main.classify; None when label is None).
        """
        min_pairs = self.min_pairs if min_pairs is None else min_pairs
        results = []
        for name, (counts, pairs) in histograms.items():
            counts = np.asarray(counts, dtype=np.float64)
            row = {
                "group": name,
                "pairs": int(pairs),
                "in_range": int(counts.sum()),
                "label": None,
                "best": None,
                "fraction": None,
                "delta": None,
                "margin": None,
                "nll": None,
            }
            if counts.sum() >= max(min_pairs, 1):
                result = classify(counts, self.templates, self.muon_isotope, self.threshold)
                row |= {
                    "label": result["label"],
                    "best": result["best"],
                    "fraction": finite_or_none(result["fraction"]),
                    "delta": finite_or_none(result["delta"]),
                    "margin": finite_or_none(result["margin"]),
                    "nll": {key: finite_or_none(value) for key, value in result["nll"].items()},
                }
            results.append(row)
        return results

    def identify(self, times, fields, adc, groups, min_pairs=None):
        """Classify one histogram per group from raw readings.

        times, fields, adc: equally long arrays (integer times, channel names, ADC values).
        groups: {group name: (D-end channel, U-end channel)}.
        Returns what identify_histograms returns.
        """
        times = np.asarray(times)
        fields = np.asarray(fields)
        adc = np.asarray(adc, dtype=np.float64)
        if not len(times) == len(fields) == len(adc):
            raise ValueError("times, fields and adc must have the same length")
        histograms = {}
        for name, (d_field, u_field) in groups.items():
            values, _ = self.pair_values(times, fields, adc, d_field, u_field)
            histograms[name] = (np.histogram(values, bins=self.edges)[0], len(values))
        return self.identify_histograms(histograms, min_pairs)

    def diagnose_histogram(self, counts, pair_summary, d, u, matched, below, above, zero, raw=None):
        """Why a scintillator is classified as it is: data scale against the templates.

        counts: accepted pairs per template bin; pair_summary: summary() of the accepted
        sqrt((U - PU) * (D - PD)) values; d / u: summary() of the raw readings of each channel;
        matched: pairs of equal time before the acquisition threshold; below / above / zero:
        pairs under the histogram range, over it, and exactly 0 after clipping negatives;
        raw: optional histogram of the raw readings (see RAW_START and RAW_BINS).

        Returns a dict with d, u, pairs, matched, filter (the threshold settings), unpaired
        (readings without a partner of equal time), values (= pair_summary), below, above, zero,
        edges, observed (counts per bin), muon (the muon template scaled to the observed
        in-range total), fit (the best isotope and its fitted fraction, with the fitted expected
        counts), template_median (median of the muon template), scale (observed median /
        template median: the factor by which the data is larger than the templates if the data
        is muons only) and raw.
        """
        counts = np.asarray(counts, dtype=np.float64)
        total = float(counts.sum())
        muon = self.templates[self.muon_isotope]
        cumulative = np.concatenate([[0.0], np.cumsum(muon)])
        template_median = float(np.interp(0.5, cumulative, self.edges))
        median = pair_summary["p50"]

        fit = None
        if total > 0:
            result = classify(counts, self.templates, self.muon_isotope, self.threshold)
            template = self.templates[result["best"]]
            fraction = result["fraction"]
            fit = {
                "isotope": result["best"],
                "fraction": float(fraction),
                "delta": finite_or_none(result["delta"]),
                "expected": (total * ((1 - fraction) * muon + fraction * template)).tolist(),
            }
        return {
            "d": d,
            "u": u,
            "pairs": int(pair_summary["n"]),
            "matched": int(matched),
            "unpaired": int(d["n"] + u["n"] - 2 * matched),
            "filter": {
                "enabled": bool(self.filter["enabled"]),
                "mode": self.filter["mode"],
                "threshold": float(self.filter["threshold"]),
            },
            "values": pair_summary,
            "below": int(below),
            "above": int(above),
            "zero": int(zero),
            "edges": np.asarray(self.edges).tolist(),
            "observed": counts.tolist(),
            "muon": (total * muon).tolist(),
            "fit": fit,
            "template_median": template_median,
            "scale": median / template_median if median and template_median > 0 else None,
            "raw": raw,
        }

    def diagnose(self, times, fields, adc, d_field, u_field):
        """diagnose_histogram from raw readings."""
        times = np.asarray(times)
        fields = np.asarray(fields)
        adc = np.asarray(adc, dtype=np.float64)
        d_adc, u_adc = adc[fields == d_field], adc[fields == u_field]
        values, matched = self.pair_values(times, fields, adc, d_field, u_field)
        edges = np.asarray(self.edges)
        raw_edges = RAW_START + np.arange(RAW_BINS + 1)
        return self.diagnose_histogram(
            np.histogram(values, bins=edges)[0],
            summary(values),
            summary(d_adc),
            summary(u_adc),
            matched,
            np.count_nonzero(values < edges[0]),
            np.count_nonzero(values > edges[-1]),
            np.count_nonzero(values == 0),
            {
                "start": RAW_START,
                "d": np.histogram(d_adc, bins=raw_edges)[0].tolist(),
                "u": np.histogram(u_adc, bins=raw_edges)[0].tolist(),
            },
        )
