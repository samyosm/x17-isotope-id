import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "config.toml"
ADC_MAX_LIMIT = np.iinfo(np.int32).max


def load_config(path=DEFAULT_PATH):
    with Path(path).open("rb") as file:
        config = tomllib.load(file)

    config.setdefault("io", {})
    config.setdefault("digitization", {})
    config.setdefault("channels", {})
    config.setdefault("filter", {})
    config.setdefault("histogram", {})
    config.setdefault("classifier", {})
    config.setdefault("sources", [])

    config["io"].setdefault("tree", "vf48")
    config["io"].setdefault("chunk_size", 100_000)
    config["io"].setdefault("output_dir", "data")
    config["digitization"].setdefault("seed", 1729)
    config["digitization"].setdefault("adc_max", 32767)
    config["filter"].setdefault("enabled", False)
    config["filter"].setdefault("mode", "both")
    config["filter"].setdefault("threshold", 0)

    config["histogram"].setdefault("ref_fraction", 0.75)
    config["histogram"].setdefault("seed", 1729)
    config["histogram"].setdefault("bins", 200)
    config["histogram"].setdefault("min", 0.0)
    config["histogram"].setdefault("max", float(config["digitization"]["adc_max"]))
    config["histogram"].setdefault("d_fpgas", [0, 1])
    config["histogram"].setdefault("u_fpgas", [2, 3])

    clf = config["classifier"]
    clf.setdefault("muon_isotope", "cosmic-muons")
    clf.setdefault("pseudocount", 0.5)
    clf.setdefault("detection_threshold", 25.0)
    clf.setdefault("window_size", 2000)
    clf.setdefault("windows_per_class", 100)
    clf.setdefault("mixed_fraction", 0.2)
    clf.setdefault("seed", 1729)

    config["io"]["output_dir"] = resolve(config["io"]["output_dir"])
    for source in config["sources"]:
        source["path"] = resolve(source["path"])

    validate(config)
    return config


def resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def validate(config):
    chunk_size = config["io"]["chunk_size"]
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if config["filter"]["enabled"] and chunk_size % 2:
        raise ValueError("When filtering, chunk_size must be even")
    if not 1 <= config["digitization"]["adc_max"] <= ADC_MAX_LIMIT:
        raise ValueError("adc_max must be a positive int32 value")
    if config["filter"]["mode"] not in ("both", "either"):
        raise ValueError("filter.mode must be 'both' or 'either'")
    hist = config["histogram"]
    if not 0 < hist["ref_fraction"] < 1:
        raise ValueError("histogram.ref_fraction must be between 0 and 1")
    if hist["bins"] <= 0:
        raise ValueError("histogram.bins must be positive")
    if not hist["min"] < hist["max"]:
        raise ValueError("histogram.min must be below histogram.max")
    if len(hist["d_fpgas"]) != len(hist["u_fpgas"]):
        raise ValueError("histogram.d_fpgas and u_fpgas must have the same length")
    clf = config["classifier"]
    if clf["pseudocount"] <= 0:
        raise ValueError("classifier.pseudocount must be positive")
    if clf["detection_threshold"] < 0:
        raise ValueError("classifier.detection_threshold must be nonnegative")
    if clf["window_size"] <= 0 or clf["windows_per_class"] <= 0:
        raise ValueError("classifier.window_size and windows_per_class must be positive")
    if not 0 < clf["mixed_fraction"] < 1:
        raise ValueError("classifier.mixed_fraction must be between 0 and 1")
    for source in config["sources"]:
        if "path" not in source or "isotope" not in source:
            raise ValueError("Each [[sources]] entry needs 'path' and 'isotope'")
    stems = [source["path"].stem for source in config["sources"]]
    if len(set(stems)) != len(stems):
        raise ValueError("Source files must have distinct filenames")


def channel_parameters(config, channel):
    params = config["defaults"] | config["channels"].get(channel, {})
    gain = float(params["gain"])
    spe_sigma = float(params["spe_relative_sigma"])
    pedestal = float(params["pedestal"])
    noise_sigma = float(params["electronic_noise_sigma"])
    if not all(np.isfinite(v) for v in (gain, spe_sigma, pedestal, noise_sigma)):
        raise ValueError(f"Nonfinite parameters for channel {channel}")
    if gain < 0 or spe_sigma < 0 or noise_sigma < 0:
        raise ValueError(f"Negative gain or noise for channel {channel}")
    return gain, spe_sigma, pedestal, noise_sigma
