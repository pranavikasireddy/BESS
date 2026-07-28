import logging

import pandas as pd
import pulp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _group_into_price_blocks(df: pd.DataFrame) -> list[list[pd.Timestamp]]:
    """Group consecutive hours into TenneT's real aFRR bidding blocks.

    Detected directly from the price data (runs of identical Up/Down price)
    rather than a hardcoded cutover date, so it self-adapts: 24-hour blocks
    before ~2025-07, 4-hour blocks after, and anything TenneT changes to in
    future data without needing this code updated.
    """
    blocks = [[df.index[0]]]
    for prev_t, t in zip(df.index, df.index[1:]):
        same_block = (
            df.loc[t, "afrr_up"] == df.loc[prev_t, "afrr_up"]
            and df.loc[t, "afrr_down"] == df.loc[prev_t, "afrr_down"]
        )
        if same_block:
            blocks[-1].append(t)
        else:
            blocks.append([t])
    return blocks


def run_lp_dispatch(
    day_ahead: pd.Series, afrr_up: pd.Series, afrr_down: pd.Series, config: dict
) -> pd.DataFrame:
    """Full-horizon LP: co-optimises day-ahead arbitrage against asymmetric
    aFRR capacity reservation (PuLP + CBC). Sees the entire price series
    before deciding anything, so this is a perfect-foresight ceiling, not a
    realistic achievable strategy — report it alongside a rolling-horizon
    version, not on its own.

    Same capacity-coupling convention as heuristic.py, for a fair
    comparison: reserving Up capacity competes with discharge headroom,
    reserving Down capacity competes with charge headroom, each capped at
    the battery's power rating.

    Enforces TenneT's LER (Limited Energy Resource) rule: reserved capacity
    must be backed by enough real energy (Up) or headroom (Down) to sustain
    it for a full 15-minute ISP, not just fit under the power rating. Without
    this, the LP can reserve Up capacity from an empty battery for free.

    Enforces the real aFRR block-granularity rule: reservation must stay
    constant within each bidding block (24h before ~2025-07, 4h after,
    detected empirically from the price data — see _group_into_price_blocks).
    Without this, the LP gets an hour-by-hour flexibility that didn't
    actually exist for most of the backtest window, inflating pre-2025
    revenue relative to what was really achievable.
    """
    battery = config["battery"]
    power_mw = battery["power_mw"]
    energy_mwh = battery["energy_mwh"]
    efficiency = battery["round_trip_efficiency"]
    cycle_cost = battery["cycle_cost_eur_mwh"]
    isp_hours = 0.25  # Dutch Imbalance Settlement Period = 15 minutes

    df = pd.DataFrame(
        {"day_ahead": day_ahead, "afrr_up": afrr_up, "afrr_down": afrr_down}
    ).dropna()
    hours = list(df.index)

    problem = pulp.LpProblem("bess_dispatch", pulp.LpMaximize)

    charge = pulp.LpVariable.dicts("charge", hours, lowBound=0)
    discharge = pulp.LpVariable.dicts("discharge", hours, lowBound=0)
    afrr_up_mw = pulp.LpVariable.dicts("afrr_up_mw", hours, lowBound=0)
    afrr_down_mw = pulp.LpVariable.dicts("afrr_down_mw", hours, lowBound=0)
    soc = pulp.LpVariable.dicts("soc", hours, lowBound=0, upBound=energy_mwh)

    for block in _group_into_price_blocks(df):
        anchor = block[0]
        for t in block[1:]:
            problem += afrr_up_mw[t] == afrr_up_mw[anchor], f"block_up_{t}"
            problem += afrr_down_mw[t] == afrr_down_mw[anchor], f"block_down_{t}"

    problem += pulp.lpSum(
        discharge[t] * df.loc[t, "day_ahead"]
        - charge[t] * df.loc[t, "day_ahead"]
        + afrr_up_mw[t] * df.loc[t, "afrr_up"]
        + afrr_down_mw[t] * df.loc[t, "afrr_down"]
        - (charge[t] + discharge[t]) * cycle_cost
        for t in hours
    )

    prev_soc = 0.0
    for t in hours:
        problem += discharge[t] + afrr_up_mw[t] <= power_mw, f"up_headroom_{t}"
        problem += charge[t] + afrr_down_mw[t] <= power_mw, f"down_headroom_{t}"
        # LER rule: reservation must be backed by energy (Up) or headroom (Down)
        # actually available at the start of the hour, for a full ISP.
        problem += afrr_up_mw[t] * isp_hours <= prev_soc, f"ler_up_{t}"
        problem += afrr_down_mw[t] * isp_hours <= energy_mwh - prev_soc, f"ler_down_{t}"
        problem += soc[t] == prev_soc + charge[t] * efficiency - discharge[t], f"soc_{t}"
        prev_soc = soc[t]

    problem.solve(pulp.PULP_CBC_CMD(msg=False))
    logger.info("LP status: %s", pulp.LpStatus[problem.status])

    records = []
    for t in hours:
        c, d = charge[t].value(), discharge[t].value()
        au, ad = afrr_up_mw[t].value(), afrr_down_mw[t].value()
        day_ahead_revenue = d * df.loc[t, "day_ahead"] - c * df.loc[t, "day_ahead"]
        afrr_revenue = au * df.loc[t, "afrr_up"] + ad * df.loc[t, "afrr_down"]
        cycling_cost = (c + d) * cycle_cost
        records.append(
            {
                "timestamp": t,
                "charge_mwh": c,
                "discharge_mwh": d,
                "afrr_up_mw": au,
                "afrr_down_mw": ad,
                "soc_mwh": soc[t].value(),
                "day_ahead_revenue": day_ahead_revenue,
                "afrr_revenue": afrr_revenue,
                "cycling_cost": cycling_cost,
                "net_revenue": day_ahead_revenue + afrr_revenue - cycling_cost,
            }
        )

    result = pd.DataFrame.from_records(records).set_index("timestamp")
    logger.info(
        "LP dispatch: %d hours, total net revenue EUR %.0f",
        len(result),
        result["net_revenue"].sum(),
    )
    return result


if __name__ == "__main__":
    from src.data_pipeline import (
        fetch_afrr_capacity_prices,
        fetch_day_ahead_prices,
        load_config,
    )

    config = load_config()
    start, end = "2024-01-01", "2024-01-08"

    day_ahead = fetch_day_ahead_prices(start, end)
    afrr = fetch_afrr_capacity_prices(start, end)

    result = run_lp_dispatch(day_ahead, afrr["Up"], afrr["Down"], config)
    print(result.head(10))
    print(result[["day_ahead_revenue", "afrr_revenue", "cycling_cost", "net_revenue"]].sum())
    both_active = ((result["charge_mwh"] > 1e-6) & (result["discharge_mwh"] > 1e-6)).sum()
    print(f"Hours with simultaneous charge AND discharge: {both_active}")
