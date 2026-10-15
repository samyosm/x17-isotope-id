"""Digitize and histogram every source file listed in config.toml.

Usage: python scripts/initialize.py [--only {digitize,histogram}] [--jobs N]
Each [[sources]] entry (path, isotope) of config.toml produces, in io.output_dir:
  STEM_digitized.root, STEM_digitized_histogram_ref.root, STEM_digitized_histogram_test.root

Steps (per source, sources run in parallel processes):
 1. digitize_file    digitize the source file (scripts/digitize.py)
 2. histogram_file   histogram the digitized file into reference/test files (scripts/histogram.py)
--only digitize runs just step 1; --only histogram runs just step 2 on the
existing digitized files. By default both run and one process per source is used, up to the number of CPU cores.
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


def digitized_path(source, config):
    return config["io"]["output_dir"] / f"{source['path'].stem}_digitized.root"


def initialize_source(source, config, only, position=None):
    desc = source["isotope"]
    if only != "histogram":
        digitize_file(
            source["path"], digitized_path(source, config), config,
            f"{desc} digitize", position,
        )
    if only != "digitize":
        histogram_file(digitized_path(source, config), config, f"{desc} histogram", position)


def initialize_in_worker(source, config, only, jobs):
    position = (multiprocessing.current_process()._identity[0] - 1) % jobs
    initialize_source(source, config, only, position)


def require_digitized_files(sources, config):
    missing = [str(digitized_path(s, config)) for s in sources if not digitized_path(s, config).exists()]
    if missing:
        raise SystemExit("Digitized files missing (run without --only histogram):\n" + "\n".join(missing))


def run_parallel(sources, config, only, jobs):
    failures = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(initialize_in_worker, source, config, only, jobs): source
            for source in sources
        }
        for future in as_completed(futures):
            error = future.exception()
            if error:
                failures.append((futures[future]["isotope"], error))
    return failures


def main():
    parser = argparse.ArgumentParser(description="Digitize and histogram all sources.")
    parser.add_argument("--only", choices=("digitize", "histogram"), help="run a single step")
    parser.add_argument("--jobs", type=int, help="number of parallel processes")
    args = parser.parse_args()

    config = load_config()
    sources = config["sources"]
    if not sources:
        raise SystemExit("No [[sources]] defined in config.toml")

    jobs = args.jobs or min(len(sources), os.cpu_count() or 1)
    if jobs < 1:
        raise SystemExit("--jobs must be at least 1")

    if args.only == "histogram":
        require_digitized_files(sources, config)

    failures = run_parallel(sources, config, args.only, min(jobs, len(sources)))
    print("\n" * min(jobs, len(sources)))
    for isotope, error in failures:
        print(f"FAILED {isotope}: {error!r}", file=sys.stderr)
    if failures:
        raise SystemExit(1)
    print(f"Processed {len(sources)} sources into {config['io']['output_dir']}")


if __name__ == "__main__":
    main()
