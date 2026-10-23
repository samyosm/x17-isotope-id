"""Library entry point: identify the source near each scintillator from detector readings.

Readings are the rows of the database, one per ADC value: a time, a channel name
(_field, e.g. "FPGA0_ADCA") and the ADC value. A group is one scintillator, given as the
channel of its D-end PMT and the channel of its U-end PMT. For each group the function

  1. keeps the readings of the two channels,
  2. pairs the U and D readings that share the same time (src/pairing.py),
  3. histograms sqrt(U * D) of the pairs on the template bins (config histogram.*),
  4. classifies that single histogram with the template fit (src/main.py).

Usage:
  identifier = Identifier.from_config()          # loads the templates once
  results = identifier.identify(times, fields, adc, {"3": ("FPGA1_ADCE", "FPGA3_ADCE")})

No calibration is applied: the values are used as stored, so the spectra only match the
templates if the detector data has the same scale as the templates.
"""

import math

import numpy as np

from config import load_config
from main import classify, load_templates
from pairing import D_END, U_END, match_pairs, pair_variable

MIN_PAIRS = 100


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


class Identifier:
    def __init__(self, config):
        self.config = config
        self.templates, self.edges = load_templates(config)
        clf = config["classifier"]
        self.muon_isotope = clf["muon_isotope"]
        self.threshold = clf["detection_threshold"]
        self.min_pairs = MIN_PAIRS

    @classmethod
    def from_config(cls, path=None):
        return cls(load_config(path) if path else load_config())

    def pair_values(self, times, fields, adc, d_field, u_field):
        """sqrt(U * D) of the U and D readings with equal times for one scintillator."""
        is_d, is_u = fields == d_field, fields == u_field
        used = is_d | is_u
        end = np.where(is_u[used], U_END, D_END)
        slot = np.zeros(len(end), dtype=np.int64)
        u_charge, d_charge = match_pairs(times[used], end, slot, adc[used])
        return pair_variable(u_charge, d_charge)

    def identify(self, times, fields, adc, groups, min_pairs=None):
        """Classify one histogram per group.

        times, fields, adc: equally long arrays (integer times, channel names, ADC values).
        groups: {group name: (D-end channel, U-end channel)}.
        Returns a list, in the order of groups, of dicts with: group, pairs (matched U/D
        pairs), in_range (pairs inside the histogram range), label (isotope, "muon-only", or
        None when fewer than min_pairs pairs are in range; default self.min_pairs), best,
        fraction, delta, margin and nll (see main.classify; None when label is None).
        """
        min_pairs = self.min_pairs if min_pairs is None else min_pairs
        times = np.asarray(times)
        fields = np.asarray(fields)
        adc = np.asarray(adc, dtype=np.float64)
        if not len(times) == len(fields) == len(adc):
            raise ValueError("times, fields and adc must have the same length")

        results = []
        for name, (d_field, u_field) in groups.items():
            values = self.pair_values(times, fields, adc, d_field, u_field)
            counts = np.histogram(values, bins=self.edges)[0]
            row = {
                "group": name,
                "pairs": int(len(values)),
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

    def diagnose(self, times, fields, adc, d_field, u_field):
        """Why a scintillator is classified as it is: data scale against the templates.

        Returns a dict with: d / u (summary of the raw readings of each channel), pairs
        (number of matched U/D pairs), unpaired (readings without a partner of equal time),
        values (summary of sqrt(U * D)), below / above / zero (pairs under the histogram range,
        over it, and exactly 0 after clipping negatives), edges, observed (counts per bin),
        muon (the muon template scaled to the observed in-range total), fit (the best isotope
        and its fitted fraction, with the fitted expected counts), template_median (median of
        the muon template) and scale (observed median / template median: the factor by which
        the data is larger than the templates if the data is muons only).
        """
        times = np.asarray(times)
        fields = np.asarray(fields)
        adc = np.asarray(adc, dtype=np.float64)
        d_adc, u_adc = adc[fields == d_field], adc[fields == u_field]
        values = self.pair_values(times, fields, adc, d_field, u_field)
        counts = np.histogram(values, bins=self.edges)[0]
        total = float(counts.sum())

        muon = self.templates[self.muon_isotope]
        cumulative = np.concatenate([[0.0], np.cumsum(muon)])
        template_median = float(np.interp(0.5, cumulative, self.edges))
        median = float(np.median(values)) if len(values) else None

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
            "d": summary(d_adc),
            "u": summary(u_adc),
            "pairs": int(len(values)),
            "unpaired": int(len(d_adc) + len(u_adc) - 2 * len(values)),
            "values": summary(values),
            "below": int(np.count_nonzero(values < self.edges[0])),
            "above": int(np.count_nonzero(values > self.edges[-1])),
            "zero": int(np.count_nonzero(values == 0)),
            "edges": self.edges.tolist(),
            "observed": counts.tolist(),
            "muon": (total * muon).tolist(),
            "fit": fit,
            "template_median": template_median,
            "scale": median / template_median if median and template_median > 0 else None,
        }
