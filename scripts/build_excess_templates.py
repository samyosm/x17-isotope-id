"""Build the source shapes used by the excess identification (templates/excess_templates.json).

Usage: uv run python scripts/build_excess_templates.py --simulations DIR --noise noise_events.npy \
           --columns noise_events_columns.txt [--out templates/excess_templates.json]

DIR holds the simulation files (312=Cs-137-*.root, 308=Bi-207-*.root, 319=Co-60-*.root, 332=Na-22-*.root).
noise_events.npy is an array (events, 32 PMT channels) of noise-only events of the real detector and
columns lists the channel name of each column (x17-data-analysis/scripts/05_noise.py writes both).
The detector response and the selection come from config.toml ([response], [excess]). Only the
truth of the simulation (bar, energy, z) is used.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import load_config
from response import build_templates
from simhits import read_hits

SOURCES = {"Cs-137": "312=Cs-137-*.root", "Bi-207": "308=Bi-207-*.root", "Co-60": "319=Co-60-*.root",
           "Na-22": "332=Na-22-*.root"}


def main():
    parser = argparse.ArgumentParser(description="Build the source shapes of the excess identification.")
    parser.add_argument("--simulations", type=Path, required=True)
    parser.add_argument("--noise", type=Path, required=True)
    parser.add_argument("--columns", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("templates/excess_templates.json"))
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    noise = np.load(args.noise)
    columns = args.columns.read_text().split()
    if noise.shape[1] != len(columns):
        raise SystemExit(f"{args.noise} has {noise.shape[1]} columns but {args.columns} lists {len(columns)}")
    # the bars that can be built: both PMTs among the noise columns
    bad = set(config["excess"].get("bad_channels", []))
    from_map = {}
    for position in range(16):  # D-end position p is scintillator (16 - p) % 16; U-end likewise on FPGA2-3
        scint = (16 - position) % 16
        d = f"FPGA{position // 8}_ADC{'ABCDEFGH'[position % 8]}"
        u = f"FPGA{2 + position // 8}_ADC{'ABCDEFGH'[position % 8]}"
        from_map[scint] = (d, u)
    bars = {b: p for b, p in sorted(from_map.items()) if p[0] in columns and p[1] in columns
            and p[0] not in bad and p[1] not in bad}
    print(f"{len(bars)} bars with two good PMTs: {sorted(bars)}")

    templates = {"version": 1, "built": datetime.now(UTC).isoformat(timespec="seconds"),
                 "threshold": config["excess"]["threshold"], "bins": config["excess"]["bins"],
                 "max": config["excess"]["max"], "kernel": config["response"]["kernel"],
                 "adc_scale": config["response"]["adc_scale"], "sources": {}}
    for name, pattern in SOURCES.items():
        matches = sorted(args.simulations.glob(pattern))
        if not matches:
            print(f"skipping {name}: no file {pattern} in {args.simulations}")
            continue
        print(f"{name}: reading {matches[0].name} ...")
        hits = read_hits(matches[0], max_hits=3_000_000)
        built = build_templates(config, hits, noise, columns, bars)
        built["file"] = matches[0].name
        templates["sources"][name] = built
        mid = built["shapes"]["1"][str(min(bars))]
        print(f"   class 1, bar {min(bars)}: pair yield per hit {built['pairs_per_hit']['1'][str(min(bars))]:.4f}, "
              f"mean v {np.dot(mid, (np.linspace(0, config['excess']['max'], config['excess']['bins'] + 1)[:-1] + 1)):.1f} ADC")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(templates, separators=(",", ":")))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
