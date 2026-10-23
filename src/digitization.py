import numpy as np

from config import channel_parameters


def apply_gain(photoelectrons, gain):
    return gain * photoelectrons


def apply_spe_fluctuations(charge, photoelectrons, gain, relative_sigma, rng):
    spread = gain * relative_sigma * np.sqrt(photoelectrons)
    return charge + spread * rng.standard_normal(len(photoelectrons))


def add_electronic_noise(charge, sigma, rng):
    return charge + sigma * rng.standard_normal(len(charge))


def add_pedestal(charge, pedestal):
    return charge + pedestal


def convert_to_adc(charge, adc_max):
    return np.clip(np.rint(charge), 0, adc_max).astype(np.int32)


def channel_parameter_table(channels, config):
    names, codes = np.unique(channels, return_inverse=True)
    table = np.array([channel_parameters(config, str(name)) for name in names])
    return table[codes].T


def digitize_batch(channels, photoelectrons, config, rng):
    gain, spe_sigma, pedestal, noise_sigma = channel_parameter_table(channels, config)
    charge = apply_gain(photoelectrons, gain)
    charge = apply_spe_fluctuations(charge, photoelectrons, gain, spe_sigma, rng)
    charge = add_electronic_noise(charge, noise_sigma, rng)
    charge = add_pedestal(charge, pedestal)
    return convert_to_adc(charge, config["digitization"]["adc_max"])


def acquisition_mask(times, adc, filter_config):
    if not filter_config["enabled"]:
        return np.ones(len(adc), dtype=bool)

    if len(adc) % 2 or np.any(times[::2] != times[1::2]):
        raise ValueError("Filter requires adjacent pairs with matching timestamps")

    threshold = float(filter_config.get("simulation_threshold", filter_config["threshold"]))
    left = adc[::2] >= threshold
    right = adc[1::2] >= threshold
    accepted = left & right if filter_config["mode"] == "both" else left | right
    return np.repeat(accepted, 2)
