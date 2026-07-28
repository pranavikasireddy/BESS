import logging

import pandas as pd
import pulp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ISP_HOURS = 0.25  # Dutch Imbalance Settlement Period = 15 minutes


def _group_into_price_blocks(df: pd.DataFrame, columns: list[str]) -> list[list[pd.Timestamp]]:
    """Group consecutive hours into TenneT's real bidding blocks for the
    given price column(s), detected directly from the data (runs of
    identical price) rather than a hardcoded cutover date. Self-adapts:
    24-hour blocks before ~2025-07, 4-hour after, and anything TenneT
    changes to in future data without needing this code updated. aFRR and
    FCR are checked separately since they're independent products that
    needn't share a block boundary.
    """
    blocks = [[df.index[0]]]
    for prev_t, t in zip(df.index, df.index[1:]):
        same_block = all(df.loc[t, c] == df.loc[prev_t, c] for c in columns)
        if same_block:
            blocks[-1].append(t)
        else:
            blocks.append([t])
    return blocks


def _solve_lp(
    df: pd.DataFrame, config: dict, initial_soc: float, terminal_soc: float | None
) -> pd.DataFrame:
    """Core LP solve shared by the full-horizon and rolling-horizon
    dispatchers — same objective and constraints either way, so the only
    real difference between them is how much of the price series `df`
    covers and what SoC boundary conditions bracket it. See
    run_lp_dispatch and run_rolling_horizon_lp for what each represents.
    """
    battery = config["battery"]
    power_mw = battery["power_mw"]
    energy_mwh = battery["energy_mwh"]
    efficiency = battery["round_trip_efficiency"]
    cycle_cost = battery["cycle_cost_eur_mwh"]

    hours = list(df.index)
    problem = pulp.LpProblem("bess_dispatch", pulp.LpMaximize)

    charge = pulp.LpVariable.dicts("charge", hours, lowBound=0)
    discharge = pulp.LpVariable.dicts("discharge", hours, lowBound=0)
    afrr_up_mw = pulp.LpVariable.dicts("afrr_up_mw", hours, lowBound=0)
    afrr_down_mw = pulp.LpVariable.dicts("afrr_down_mw", hours, lowBound=0)
    fcr_mw = pulp.LpVariable.dicts("fcr_mw", hours, lowBound=0)
    soc = pulp.LpVariable.dicts("soc", hours, lowBound=0, upBound=energy_mwh)

    for block in _group_into_price_blocks(df, ["afrr_up", "afrr_down"]):
        anchor = block[0]
        for t in block[1:]:
            problem += afrr_up_mw[t] == afrr_up_mw[anchor], f"block_up_{t}"
            problem += afrr_down_mw[t] == afrr_down_mw[anchor], f"block_down_{t}"

    for block in _group_into_price_blocks(df, ["fcr"]):
        anchor = block[0]
        for t in block[1:]:
            problem += fcr_mw[t] == fcr_mw[anchor], f"block_fcr_{t}"

    problem += pulp.lpSum(
        discharge[t] * df.loc[t, "day_ahead"]
        - charge[t] * df.loc[t, "day_ahead"]
        + afrr_up_mw[t] * df.loc[t, "afrr_up"]
        + afrr_down_mw[t] * df.loc[t, "afrr_down"]
        + fcr_mw[t] * df.loc[t, "fcr"]
        - (charge[t] + discharge[t]) * cycle_cost
        for t in hours
    )

    prev_soc = initial_soc
    for t in hours:
        problem += (
            discharge[t] + afrr_up_mw[t] + fcr_mw[t] <= power_mw,
            f"up_headroom_{t}",
        )
        problem += (
            charge[t] + afrr_down_mw[t] + fcr_mw[t] <= power_mw,
            f"down_headroom_{t}",
        )
        # LER rule: reservation must be backed by energy (Up/FCR) or headroom
        # (Down/FCR) actually available at the start of the hour, for a full ISP.
        problem += (afrr_up_mw[t] + fcr_mw[t]) * ISP_HOURS <= prev_soc, f"ler_up_{t}"
        problem += (
            ((afrr_down_mw[t] + fcr_mw[t]) * ISP_HOURS <= energy_mwh - prev_soc),
            f"ler_down_{t}",
        )
        problem += (
            soc[t] == prev_soc + charge[t] * efficiency - discharge[t],
            f"soc_{t}",
        )
        prev_soc = soc[t]

    if terminal_soc is not None:
        problem += soc[hours[-1]] == terminal_soc, "terminal_soc"

    problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[problem.status] != "Optimal":
        logger.warning("LP status: %s for window %s to %s", pulp.LpStatus[problem.status],
                        hours[0], hours[-1])

    records = []
    for t in hours:
        c, d = charge[t].value(), discharge[t].value()
        au, ad, f = afrr_up_mw[t].value(), afrr_down_mw[t].value(), fcr_mw[t].value()
        day_ahead_revenue = d * df.loc[t, "day_ahead"] - c * df.loc[t, "day_ahead"]
        afrr_revenue = au * df.loc[t, "afrr_up"] + ad * df.loc[t, "afrr_down"]
        fcr_revenue = f * df.loc[t, "fcr"]
        cycling_cost = (c + d) * cycle_cost
        records.append(
            {
                "timestamp": t,
                "charge_mwh": c,
                "discharge_mwh": d,
                "afrr_up_mw": au,
                "afrr_down_mw": ad,
                "fcr_mw": f,
                "soc_mwh": soc[t].value(),
                "day_ahead_revenue": day_ahead_revenue,
                "afrr_revenue": afrr_revenue,
                "fcr_revenue": fcr_revenue,
                "cycling_cost": cycling_cost,
                "net_revenue": day_ahead_revenue + afrr_revenue + fcr_revenue - cycling_cost,
            }
        )
    return pd.DataFrame.from_records(records).set_index("timestamp")


def _build_price_frame(
    day_ahead: pd.Series, afrr_up: pd.Series, afrr_down: pd.Series, fcr: pd.Series
) -> pd.DataFrame:
    return pd.DataFrame(
        {"day_ahead": day_ahead, "afrr_up": afrr_up, "afrr_down": afrr_down, "fcr": fcr}
    ).dropna()


def run_lp_dispatch(
    day_ahead: pd.Series,
    afrr_up: pd.Series,
    afrr_down: pd.Series,
    fcr: pd.Series,
    config: dict,
) -> pd.DataFrame:
    """Full-horizon LP: co-optimises day-ahead arbitrage against asymmetric
    aFRR capacity and symmetric FCR capacity reservation (PuLP + CBC), given
    the ENTIRE price series at once. This is a perfect-foresight ceiling —
    an upper bound on achievable revenue, not a realistic strategy, since a
    real operator would never see next month's prices while deciding
    today's dispatch. Report alongside run_rolling_horizon_lp, not alone.
    """
    df = _build_price_frame(day_ahead, afrr_up, afrr_down, fcr)
    result = _solve_lp(df, config, initial_soc=0.0, terminal_soc=None)
    logger.info(
        "Full-horizon LP: %d hours, total net revenue EUR %.0f",
        len(result),
        result["net_revenue"].sum(),
    )
    return result


def run_rolling_horizon_lp(
    day_ahead: pd.Series,
    afrr_up: pd.Series,
    afrr_down: pd.Series,
    fcr: pd.Series,
    config: dict,
) -> pd.DataFrame:
    """Rolling-horizon LP: solves one calendar day at a time, using only
    that day's own prices — genuinely known by the time of dispatch, since
    NL's day-ahead auction clears ~noon the day before delivery, and
    aFRR/FCR capacity auctions run daily on the same d-1 basis. This is the
    realistic, defensible headline number, unlike run_lp_dispatch's
    perfect-foresight ceiling.

    Solving each day in isolation makes an unconstrained LP want to end at
    0 SoC every day (nothing to gain by holding energy it can't see
    tomorrow's value for). Fixed by requiring every day to both start AND
    end at a fixed boundary SoC (config: rolling_horizon_boundary_soc_frac,
    default 50% of capacity) — except day one, which starts at 0 like every
    other method in this project, for a like-for-like comparison.
    """
    df = _build_price_frame(day_ahead, afrr_up, afrr_down, fcr)
    boundary_frac = config["market"].get("rolling_horizon_boundary_soc_frac", 0.5)
    boundary_soc = config["battery"]["energy_mwh"] * boundary_frac

    daily_results = []
    initial_soc = 0.0
    for _, day_df in df.groupby(df.index.date):
        day_result = _solve_lp(
            day_df, config, initial_soc=initial_soc, terminal_soc=boundary_soc
        )
        daily_results.append(day_result)
        initial_soc = boundary_soc

    result = pd.concat(daily_results)
    logger.info(
        "Rolling-horizon LP: %d hours across %d days, total net revenue EUR %.0f",
        len(result),
        len(daily_results),
        result["net_revenue"].sum(),
    )
    return result


if __name__ == "__main__":
    from src.data_pipeline import (
        fetch_afrr_capacity_prices,
        fetch_day_ahead_prices,
        fetch_fcr_capacity_prices,
        load_config,
    )

    config = load_config()
    start, end = "2024-01-01", "2024-01-08"

    day_ahead = fetch_day_ahead_prices(start, end)
    afrr = fetch_afrr_capacity_prices(start, end)
    fcr = fetch_fcr_capacity_prices(start, end)

    full = run_lp_dispatch(day_ahead, afrr["Up"], afrr["Down"], fcr, config)
    rolling = run_rolling_horizon_lp(day_ahead, afrr["Up"], afrr["Down"], fcr, config)

    print("Full-horizon (perfect foresight ceiling):")
    print(f"  EUR {full['net_revenue'].sum():,.0f}")
    print("Rolling-horizon (realistic, day-by-day):")
    print(f"  EUR {rolling['net_revenue'].sum():,.0f}")
    print(f"  Rolling captures {rolling['net_revenue'].sum() / full['net_revenue'].sum():.1%} "
          f"of the full-horizon ceiling")
