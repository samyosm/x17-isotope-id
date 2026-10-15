"""Histogram sqrt((U - PU)(D - PD)) of matched PMT pairs into reference/test files.

Usage: python scripts/histogram.py INPUT.root
Outputs: INPUT_histogram_ref.root and INPUT_histogram_test.root, next to INPUT.
INPUT is the output of scripts/digitize.py. Settings come from config.toml.

Steps (per batch of rows):
 1. read_batches     read _time, _field, ADC
 2. complete_rows    hold back the rows of the last _time so no pair is split across batches
 3. channel_table    map each channel to (end, pair slot) and look up its pedestal
 4. match_pairs      pair the U and D rows that share the same exact _time and pair slot
 5. pair_variable    sqrt(max(0, U - PU) * max(0, D - PD))
 6. Spectra.fill     randomly assign each pair to reference or test and histogram it

A pair slot is the ADC letter and the position of the FPGA in the d_fpgas/u_fpgas
lists of config.toml. Rows sharing a _time are assumed to be adjacent. The in-range test
values are also saved one per pair (INPUT_histogram_test_values.root) for building
evaluation windows. Values outside
[min, max] are counted as underflow/overflow and reported as a warning.
"""

import argparse
import sys
import warnings
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import load_config
from pairing import channel_table, match_pairs, pair_variable
from rootio import BRANCHES, require_branches

HISTOGRAM_NAME = "sqrt_UD"
TEST_VALUES_TREE = "test_values"


class Spectra:
    def __init__(self, config):
        hist = config["histogram"]
        self.edges = np.linspace(hist["min"], hist["max"], int(hist["bins"]) + 1)
        self.ref_fraction = hist["ref_fraction"]
        self.rng = np.random.default_rng(hist["seed"])
        self.counts = {
            "ref": np.zeros(int(hist["bins"]), dtype=np.int64),
            "test": np.zeros(int(hist["bins"]), dtype=np.int64),
        }
        self.test_values = []
        self.underflow = self.overflow = 0

    def fill(self, values):
        self.underflow += int(np.count_nonzero(values < self.edges[0]))
        self.overflow += int(np.count_nonzero(values > self.edges[-1]))
        to_ref = self.rng.random(len(values)) < self.ref_fraction
        self.counts["ref"] += np.histogram(values[to_ref], bins=self.edges)[0]
        self.counts["test"] += np.histogram(values[~to_ref], bins=self.edges)[0]
        held_out = values[~to_ref]
        self.test_values.append(
            held_out[(held_out >= self.edges[0]) & (held_out <= self.edges[-1])]
        )

    def save(self, input_path):
        for label, counts in self.counts.items():
            path = input_path.with_name(
                f"{input_path.stem}_histogram_{label}{input_path.suffix}"
            )
            with uproot.recreate(path) as target:
                target[HISTOGRAM_NAME] = (counts, self.edges)
            tqdm.write(f"{label}: {int(counts.sum()):,} pairs -> {path}")
        path = input_path.with_name(
            f"{input_path.stem}_histogram_test_values{input_path.suffix}"
        )
        with uproot.recreate(path) as target:
            target[TEST_VALUES_TREE] = {"value": np.concatenate(self.test_values)}
        tqdm.write(f"test values -> {path}")

    def warn_out_of_range(self):
        if self.underflow:
            warnings.warn(
                f"{self.underflow:,} pairs below histogram.min={self.edges[0]:g}"
                " (underflow, not histogrammed)"
            )
        if self.overflow:
            warnings.warn(
                f"{self.overflow:,} pairs above histogram.max={self.edges[-1]:g}"
                " (overflow, not histogrammed)"
            )


def read_batches(tree, chunk_size, bar):
    for batch in tree.iterate(BRANCHES, step_size=chunk_size, library="ak"):
        bar.update(len(batch))
        yield (
            ak.to_numpy(batch["_time"]),
            ak.to_numpy(batch["_field"]),
            ak.to_numpy(batch["ADC"]).astype(np.float64),
        )


def complete_rows(batches):
    carry = None
    for times, fields, adc in batches:
        if carry is not None:
            times = np.concatenate([carry[0], times])
            fields = np.concatenate([carry[1], fields])
            adc = np.concatenate([carry[2], adc])
        held = times == times[-1]
        carry = (times[held], fields[held], adc[held])
        yield times[~held], fields[~held], adc[~held]
    if carry is not None:
        yield carry


def histogram_file(input_path, config, desc=None, position=None):
    spectra = Spectra(config)

    with uproot.open(input_path) as source:
        tree = source[config["io"]["tree"]]
        require_branches(tree)
        with tqdm(
            total=tree.num_entries,
            unit="row",
            unit_scale=True,
            desc=desc,
            position=position,
        ) as bar:
            batches = read_batches(tree, config["io"]["chunk_size"], bar)
            for times, fields, adc in complete_rows(batches):
                end, slot, pedestal = channel_table(fields, config)
                u_charge, d_charge = match_pairs(times, end, slot, adc - pedestal)
                spectra.fill(pair_variable(u_charge, d_charge))

    spectra.save(input_path)
    spectra.warn_out_of_range()


def main():
    parser = argparse.ArgumentParser(description="Histogram sqrt((U-PU)(D-PD)).")
    parser.add_argument("input", type=Path, help="output of digitize.py")
    args = parser.parse_args()
    histogram_file(args.input, load_config())


if __name__ == "__main__":
    main()
