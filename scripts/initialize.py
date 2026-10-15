"""Digitize and histogram every source file listed in config.toml.

Usage: python scripts/initialize.py [--jobs N]
Each [[sources]] entry (path, isotope) of config.toml produces, in io.output_dir:
  STEM_digitized.root, STEM_digitized_histogram_ref.root, STEM_digitized_histogram_test.root

Steps (per source, sources run in parallel processes):
 1. digitize_file    digitize the source file (scripts/digitize.py)
 2. histogram_file   histogram the digitized file into reference/test files (scripts/histogram.py)
By default one process per source is used, up to the number of CPU cores.
The isotope label is carried along but not used yet.
"""

import argparse
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import load_config
from digitize import digitize_file
from histogram import histogram_file


def initialize_source(source, config, position=None):
    input_path = source["path"]
    digitized_path = config["io"]["output_dir"] / f"{input_path.stem}_digitized.root"
    desc = source["isotope"]
    digitize_file(input_path, digitized_path, config, f"{desc} digitize", position)
    histogram_file(digitized_path, config, f"{desc} histogram", position)


def initialize_in_worker(source, config, jobs):
    position = (multiprocessing.current_process()._identity[0] - 1) % jobs
    initialize_source(source, config, position)


def run_parallel(sources, config, jobs):
    failures = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(initialize_in_worker, source, config, jobs): source
            for source in sources
        }
        for future in as_completed(futures):
            error = future.exception()
            if error:
                failures.append((futures[future]["isotope"], error))
    return failures


def main():
    parser = argparse.ArgumentParser(description="Digitize and histogram all sources.")
    parser.add_argument("--jobs", type=int, help="number of parallel processes")
    args = parser.parse_args()

    config = load_config()
    sources = config["sources"]
    if not sources:
        raise SystemExit("No [[sources]] defined in config.toml")

    jobs = args.jobs or min(len(sources), os.cpu_count() or 1)
    if jobs < 1:
        raise SystemExit("--jobs must be at least 1")

    failures = run_parallel(sources, config, min(jobs, len(sources)))
    print("\n" * min(jobs, len(sources)))
    for isotope, error in failures:
        print(f"FAILED {isotope}: {error!r}", file=sys.stderr)
    if failures:
        raise SystemExit(1)
    print(f"Processed {len(sources)} sources into {config['io']['output_dir']}")


if __name__ == "__main__":
    main()
