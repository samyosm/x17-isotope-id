"""Digitize simulated PMT photoelectron counts.

Usage: python scripts/digitize.py INPUT.root [OUTPUT.root]
Default output: INPUT_digitized.root. Settings come from config.toml.

Steps (per batch of rows, with per-channel settings applied row-wise):
 1. read_batch              read _time, _field, ADC (photoelectrons) and validate
 2. apply_gain              charge = gain * photoelectrons
 3. apply_spe_fluctuations  add single-photoelectron gain spread
 4. add_electronic_noise    add gaussian electronic noise
 5. add_pedestal            add the channel pedestal
 6. convert_to_adc          round and clip to [0, adc_max]
 7. acquisition_mask        optionally keep only records passing the threshold
 8. write_batch             append the kept rows to the output tree

Steps 2-6 are chained by digitize_batch (src/digitization.py).
"""

import argparse
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import load_config
from digitization import acquisition_mask, digitize_batch
from rootio import BRANCHES, require_branches


def read_batch(batch):
    times = ak.to_numpy(batch["_time"])
    fields = batch["_field"]
    photoelectrons = ak.to_numpy(batch["ADC"]).astype(np.float64)
    if not np.all(np.isfinite(photoelectrons)) or np.any(photoelectrons < 0):
        raise ValueError("Photoelectron counts must be finite and nonnegative")
    return times, fields, photoelectrons


def write_batch(output_tree, times, fields, adc, keep):
    if not np.any(keep):
        return
    output_tree.extend(
        {
            "_time": times[keep],
            "_field": fields[keep],
            "ADC": adc[keep],
        }
    )


def default_output_path(input_path):
    return input_path.with_name(f"{input_path.stem}_digitized{input_path.suffix}")


def digitize_file(input_path, output_path, config, desc=None, position=None):
    tree_name = config["io"]["tree"]
    rng = np.random.default_rng(config["digitization"]["seed"])
    total_input = total_output = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.open(input_path) as source, uproot.recreate(output_path) as target:
        input_tree = source[tree_name]
        require_branches(input_tree)

        output_tree = target.mktree(
            tree_name,
            {
                "_time": np.dtype(input_tree["_time"].interpretation.numpy_dtype),
                "_field": "string",
                "ADC": np.dtype("int32"),
            },
        )

        batches = input_tree.iterate(
            BRANCHES, step_size=config["io"]["chunk_size"], library="ak"
        )
        with tqdm(
            total=input_tree.num_entries,
            unit="row",
            unit_scale=True,
            desc=desc,
            position=position,
        ) as bar:
            for batch in batches:
                times, fields, photoelectrons = read_batch(batch)
                adc = digitize_batch(ak.to_numpy(fields), photoelectrons, config, rng)
                keep = acquisition_mask(times, adc, config["filter"])
                write_batch(output_tree, times, fields, adc, keep)
                total_input += len(adc)
                total_output += int(np.count_nonzero(keep))
                bar.update(len(adc))

    tqdm.write(f"Digitized {total_input:,} rows -> {total_output:,} rows: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Digitize PMT photoelectron counts.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()

    output_path = args.output or default_output_path(args.input)
    if args.input.resolve() == output_path.resolve():
        parser.error("input and output files must be different")

    digitize_file(args.input, output_path, load_config())


if __name__ == "__main__":
    main()
