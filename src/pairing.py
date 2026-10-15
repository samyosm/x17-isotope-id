import re

import numpy as np

from config import channel_parameters

CHANNEL_PATTERN = re.compile(r"FPGA(\d+)_ADC([A-Z])$")
D_END, U_END = 0, 1
LETTERS = 26


def channel_table(fields, config):
    hist = config["histogram"]
    ends = {fpga: (D_END, slot) for slot, fpga in enumerate(hist["d_fpgas"])}
    ends |= {fpga: (U_END, slot) for slot, fpga in enumerate(hist["u_fpgas"])}

    names, codes = np.unique(fields, return_inverse=True)
    end = np.full(len(names), -1)
    slot = np.zeros(len(names), dtype=np.int64)
    pedestal = np.zeros(len(names))
    for i, name in enumerate(names):
        match = CHANNEL_PATTERN.match(str(name))
        if not match or int(match[1]) not in ends:
            continue
        end[i], fpga_slot = ends[int(match[1])]
        slot[i] = fpga_slot * LETTERS + ord(match[2]) - ord("A")
        pedestal[i] = channel_parameters(config, str(name))[2]
    return end[codes], slot[codes], pedestal[codes]


def match_pairs(times, end, slot, charge):
    used = end >= 0
    times, end, slot, charge = times[used], end[used], slot[used], charge[used]
    order = np.lexsort((end, slot, times))
    times, end, slot, charge = times[order], end[order], slot[order], charge[order]

    new_group = np.ones(len(times), dtype=bool)
    new_group[1:] = (times[1:] != times[:-1]) | (slot[1:] != slot[:-1])
    starts = np.flatnonzero(new_group)
    sizes = np.diff(np.append(starts, len(times)))
    starts = starts[sizes == 2]
    starts = starts[(end[starts] == D_END) & (end[starts + 1] == U_END)]
    return charge[starts + 1], charge[starts]


def pair_variable(u_charge, d_charge):
    return np.sqrt(np.maximum(u_charge, 0) * np.maximum(d_charge, 0))
