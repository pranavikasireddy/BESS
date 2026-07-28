import pandas as pd


def apply_grid_transport_cost(
    net_revenue_eur: float, dispatch: pd.DataFrame, config: dict
) -> float:
    """Deduct the fixed grid transport/connection cost from a dispatch
    result's net revenue.

    This is a capacity-based fixed cost (you pay for the connection whether
    or not you dispatch), not a per-MWh operating cost — so unlike cycling
    cost, it has no place inside the LP/heuristic objective, where it would
    incorrectly influence hour-by-hour decisions it has no real bearing on.
    It's applied once, after the fact, prorated by how many days the
    dispatch actually covers so the same function works on a one-week test
    slice or the full multi-year backtest without changing the annual rate.
    Proration uses row count / 24, not the timestamp span between the first
    and last row — the latter undercounts by exactly one row's duration
    (N equally-spaced hourly points span N-1 hours, not N), which mattered
    enough to get wrong once already: it silently reported 8.00 days for a
    169-row week that's actually 169/24 = 7.04 days, a ~14% overstatement.

    See config.yaml `grid.transport_eur_per_mw_year` for the rate and its
    provisional/incomplete status — this only covers the annual capacity
    charge, not the separate monthly max-demand charge TenneT also applies.
    """
    power_mw = config["battery"]["power_mw"]
    annual_rate = config["grid"]["transport_eur_per_mw_year"]
    days_covered = len(dispatch) / 24
    cost = annual_rate * power_mw * (days_covered / 365)
    return net_revenue_eur - cost
