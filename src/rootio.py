BRANCHES = ("_time", "_field", "ADC")


def require_branches(tree):
    missing = set(BRANCHES) - set(tree.keys())
    if missing:
        raise KeyError(f"Missing required branches: {sorted(missing)}")
