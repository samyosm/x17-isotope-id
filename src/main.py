"""Template-fit isotope classifier and its evaluation.

Usage:
  python src/main.py HISTOGRAM [--name sqrt_UD]   classify one histogram
  python src/main.py --evaluate                   classify windows of held-out measurements

Templates are the reference histograms written by scripts/initialize.py (config
io.output_dir), smoothed with a pseudocount alpha and normalized:
    T_i = (H_i + alpha) / (sum_j H_j + alpha * B)
For each isotope s, with N the observed total and T_mu the muon template, the observed
counts n_i are Poisson with mean
    lambda_i(f) = N * ((1 - f) * T_mu_i + f * T_s_i),   0 <= f <= 1,
and f is fitted by minimizing the Poisson negative log-likelihood (NLL). Isotopes are fitted
independently (at most one source is assumed present). The muon-only hypothesis is f = 0.
With Delta = 2 * (NLL_muon_only - NLL_best_isotope), the observation is "muon-only" if
Delta <= classifier.detection_threshold, else the best isotope. Delta and f are not
probabilities and the threshold is not calibrated by any chi-square assumption.

NLLs omit the constant sum(ln n_i!), which is the same for every hypothesis.
"""

import argparse
from pathlib import Path

import numpy as np
import uproot
from scipy.optimize import minimize_scalar

from config import load_config

HISTOGRAM_NAME = "sqrt_UD"
TEST_VALUES_TREE = "test_values"
MUON_ONLY = "muon-only"


# --- loading and validation -------------------------------------------------


def validate_counts(counts, label):
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError(f"{label}: counts must be finite and nonnegative")


def validate_bins(edges, reference_edges, label):
    if len(edges) != len(reference_edges) or not np.allclose(edges, reference_edges):
        raise ValueError(f"{label}: bin edges differ from the reference templates")


def load_histogram(path, name=HISTOGRAM_NAME):
    """Return (counts, edges) of a ROOT histogram."""
    with uproot.open(path) as file:
        counts, edges = file[name].to_numpy()
    counts = counts.astype(np.float64)
    validate_counts(counts, str(path))
    return counts, edges


def source_file(source, config, suffix):
    stem = source["path"].stem
    return config["io"]["output_dir"] / f"{stem}_digitized_histogram_{suffix}.root"


def normalize(counts, alpha):
    return (counts + alpha) / (counts.sum() + alpha * len(counts))


def load_templates(config):
    """Return ({isotope: template}, edges); sources of one isotope are summed."""
    counts, edges = {}, None
    for source in config["sources"]:
        hist, hist_edges = load_histogram(source_file(source, config, "ref"))
        if edges is None:
            edges = hist_edges
        validate_bins(hist_edges, edges, str(source["path"]))
        isotope = source["isotope"]
        counts[isotope] = counts.get(isotope, 0) + hist
    alpha = config["classifier"]["pseudocount"]
    return {isotope: normalize(hist, alpha) for isotope, hist in counts.items()}, edges


def load_observed(path, name, edges):
    if path.suffix == ".root":
        counts, observed_edges = load_histogram(path, name)
        validate_bins(observed_edges, edges, str(path))
        return counts
    if path.suffix == ".npy":
        counts = np.load(path).astype(np.float64)
    else:
        counts = np.loadtxt(path, delimiter="," if path.suffix == ".csv" else None)
    validate_counts(counts, str(path))
    if len(counts) != len(edges) - 1:
        raise ValueError(f"{path}: {len(counts)} bins, templates have {len(edges) - 1}")
    return counts


# --- fitting and classification ---------------------------------------------


def poisson_nll(f, observed, muon, isotope):
    expected = observed.sum() * ((1 - f) * muon + f * isotope)
    return float(np.sum(expected - observed * np.log(expected)))


def fit_isotope(observed, muon, isotope):
    """Best-fit isotope fraction f in [0, 1] and the NLL there."""
    result = minimize_scalar(
        poisson_nll,
        bounds=(0.0, 1.0),
        args=(observed, muon, isotope),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"Optimizer failed: {result.message}")
    # The bounded search never evaluates the endpoints exactly; check them too.
    candidates = [(result.x, result.fun)]
    candidates += [(f, poisson_nll(f, observed, muon, isotope)) for f in (0.0, 1.0)]
    f, nll = min(candidates, key=lambda item: item[1])
    if not np.isfinite(nll):
        raise RuntimeError("Non-finite negative log-likelihood")
    return float(f), nll


def classify(observed, templates, muon_isotope, threshold):
    """Classify one observed histogram.

    Returns a dict with: label (isotope or MUON_ONLY), best (best-fitting isotope),
    fraction (its fitted f), delta (2 * (NLL_muon_only - NLL_best)),
    margin (second-best minus best isotope NLL, nan with one isotope), and nll
    ({hypothesis: NLL}, including MUON_ONLY).
    """
    observed = np.asarray(observed, dtype=np.float64)
    if observed.sum() <= 0:
        raise ValueError("Observed histogram is empty")
    muon = templates[muon_isotope]
    fits = {
        isotope: fit_isotope(observed, muon, template)
        for isotope, template in templates.items()
        if isotope != muon_isotope
    }
    if not fits:
        raise ValueError("No non-muon isotope templates to classify against")

    nll = {isotope: value for isotope, (_, value) in fits.items()}
    nll[MUON_ONLY] = poisson_nll(0.0, observed, muon, muon)
    if not np.isfinite(nll[MUON_ONLY]):
        raise RuntimeError("Non-finite negative log-likelihood")

    ranked = sorted(fits, key=lambda isotope: nll[isotope])
    best = ranked[0]
    delta = 2 * (nll[MUON_ONLY] - nll[best])
    margin = nll[ranked[1]] - nll[best] if len(ranked) > 1 else np.nan
    return {
        "label": MUON_ONLY if delta <= threshold else best,
        "best": best,
        "fraction": fits[best][0],
        "delta": delta,
        "margin": margin,
        "nll": nll,
    }


def print_result(result):
    print(f"  prediction: {result['label']}")
    print(f"  best isotope: {result['best']}  f = {result['fraction']:.4f}")
    print(f"  Delta = 2*(NLL_muon-only - NLL_best) = {result['delta']:.2f}")
    print(f"  NLL margin (second best - best isotope) = {result['margin']:.2f}")
    for hypothesis, value in sorted(result["nll"].items(), key=lambda item: item[1]):
        print(f"    {hypothesis:<20} NLL = {value:.2f}")


# --- evaluation ---------------------------------------------------------------


def load_test_values(config):
    """Return {isotope: held-out per-pair values}; sources of one isotope are joined."""
    values = {}
    for source in config["sources"]:
        path = source_file(source, config, "test_values")
        with uproot.open(path) as file:
            held_out = file[TEST_VALUES_TREE]["value"].array(library="np")
        values.setdefault(source["isotope"], []).append(held_out)
    return {isotope: np.concatenate(parts) for isotope, parts in values.items()}


def build_windows(test_values, muon_isotope, config):
    """Yield (truth, kind, values) windows of disjoint held-out measurements.

    kinds: "pure" (muons or isotope only) and "mixed" (isotope plus muons, with
    classifier.mixed_fraction of the measurements from the isotope).
    """
    clf = config["classifier"]
    size, count, fraction = clf["window_size"], clf["windows_per_class"], clf["mixed_fraction"]
    rng = np.random.default_rng(clf["seed"])
    pools = {isotope: rng.permutation(values) for isotope, values in test_values.items()}
    used = dict.fromkeys(pools, 0)

    def take(isotope, number):
        start = used[isotope]
        if start + number > len(pools[isotope]):
            raise ValueError(
                f"Not enough held-out measurements for {isotope}; "
                "reduce classifier.window_size or windows_per_class"
            )
        used[isotope] = start + number
        return pools[isotope][start : start + number]

    for _ in range(count):
        yield muon_isotope, "pure", take(muon_isotope, size)
    n_source = int(round(fraction * size))
    for isotope in pools:
        if isotope == muon_isotope:
            continue
        for _ in range(count):
            yield isotope, "pure", take(isotope, size)
        for _ in range(count):
            mixed = np.concatenate([take(isotope, n_source), take(muon_isotope, size - n_source)])
            yield isotope, "mixed", mixed


def evaluate(templates, edges, test_values, config):
    clf = config["classifier"]
    muon_isotope = clf["muon_isotope"]
    labels = [MUON_ONLY] + [i for i in templates if i != muon_isotope]
    index = {name: k for k, name in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)), dtype=int)
    by_kind = {}
    muon_deltas = []

    for truth, kind, values in build_windows(test_values, muon_isotope, config):
        counts = np.histogram(values, bins=edges)[0]
        result = classify(counts, templates, muon_isotope, clf["detection_threshold"])
        truth_label = MUON_ONLY if truth == muon_isotope else truth
        confusion[index[truth_label], index[result["label"]]] += 1
        by_kind.setdefault((truth_label, kind), []).append(result["label"] == truth_label)
        if truth == muon_isotope:
            muon_deltas.append(result["delta"])

    print(
        f"{clf['windows_per_class']} windows per class and kind, {clf['window_size']} "
        f"measurements per window, mixed isotope fraction {clf['mixed_fraction']}, "
        f"threshold {clf['detection_threshold']}"
    )
    print(f"\noverall accuracy: {confusion.trace() / confusion.sum():.3f}")
    print("\nper-source accuracy:")
    for (truth, kind), hits in by_kind.items():
        print(f"  {truth:<12} {kind:<6} {np.mean(hits):.3f}  ({len(hits)} windows)")
    print("\nconfusion matrix (rows: truth, columns: prediction; mixed windows count as their isotope)")
    print(" " * 12 + "".join(f"{name:>12}" for name in labels))
    for name, row in zip(labels, confusion):
        print(f"{name:<12}" + "".join(f"{value:>12d}" for value in row))
    print(
        "\nDelta of pure-muon windows (for calibrating the threshold): "
        f"mean {np.mean(muon_deltas):.2f}, 95% {np.percentile(muon_deltas, 95):.2f}, "
        f"max {np.max(muon_deltas):.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Classify histograms by template fit.")
    parser.add_argument("input", type=Path, nargs="?", help="ROOT file, or .npy/.txt/.csv of bin counts")
    parser.add_argument("--name", default=HISTOGRAM_NAME, help="histogram name in a ROOT file")
    parser.add_argument("--evaluate", action="store_true", help="evaluate on held-out windows")
    args = parser.parse_args()
    if args.evaluate == (args.input is not None):
        parser.error("give either a histogram file or --evaluate")

    config = load_config()
    clf = config["classifier"]
    templates, edges = load_templates(config)
    if args.evaluate:
        evaluate(templates, edges, load_test_values(config), config)
        return

    observed = load_observed(args.input, args.name, edges)
    print(f"{args.input}: {int(observed.sum()):,} entries")
    result = classify(observed, templates, clf["muon_isotope"], clf["detection_threshold"])
    print_result(result)


if __name__ == "__main__":
    main()
