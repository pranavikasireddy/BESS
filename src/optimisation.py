import logging

import pandas as pd
import pulp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _group_into_price_blocks(
    df: pd.DataFrame, columns: list[str]
) -> list[list[pd.Timestamp]]:
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


def run_lp_dispatch(
    day_ahead: pd.Series,
    afrr_up: pd.Series,
    afrr_down: pd.Series,
    fcr: pd.Series,
    config: dict,
) -> pd.DataFrame:
    """Full-horizon LP: co-optimises day-ahead arbitrage against asymmetric
    aFRR capacity and symmetric FCR capacity reservation (PuLP + CBC). Sees
    the entire price series before deciding anything, so this is a
    perfect-foresight ceiling, not a realistic achievable strategy — report
    it alongside a rolling-horizon version, not on its own.

    FCR is symmetric (one price, one MW figure) rather than Up/Down like
    aFRR, because it responds automatically to frequency deviations in
    either direction — so a reserved FCR MW competes for headroom on BOTH
    the charge and discharge side simultaneously, unlike aFRR's Up/Down
    which each only compete on one side.

    Enforces TenneT's LER (Limited Energy Resource) rule: reserved capacity
    (aFRR + FCR together) must be backed by enough real energy (Up/FCR) or
    headroom (Down/FCR) to sustain it for a full 15-minute ISP, not just fit
    under the power rating. Without this, the LP can reserve capacity from
    an empty battery for free.

    Enforces block-granularity separately for aFRR and FCR: each must stay
    constant within its own real bidding block (detected empirically from
    the price data — see _group_into_price_blocks). Without this, the LP
    gets hour-by-hour flexibility that didn't actually exist for most of
    the backtest window, inflating revenue relative to what was achievable.
    """
    battery = config["battery"]
    power_mw = battery["power_mw"]
    energy_mwh = battery["energy_mwh"]
    efficiency = battery["round_trip_efficiency"]
    cycle_cost = battery["cycle_cost_eur_mwh"]
    isp_hours = 0.25  # Dutch Imbalance Settlement Period = 15 minutes

    df = pd.DataFrame(
        {"day_ahead": day_ahead, "afrr_up": afrr_up, "afrr_down": afrr_down, "fcr": fcr}
    ).dropna()
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

    prev_soc = 0.0
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
        problem += (afrr_up_mw[t] + fcr_mw[t]) * isp_hours <= prev_soc, f"ler_up_{t}"
        problem += (
            ((afrr_down_mw[t] + fcr_mw[t]) * isp_hours <= energy_mwh - prev_soc),
            f"ler_down_{t}",
        )
        problem += (
            soc[t] == prev_soc + charge[t] * efficiency - discharge[t],
            f"soc_{t}",
        )
        prev_soc = soc[t]

    problem.solve(pulp.PULP_CBC_CMD(msg=False))
    logger.info("LP status: %s", pulp.LpStatus[problem.status])

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
                "net_revenue": day_ahead_revenue
                + afrr_revenue
                + fcr_revenue
                - cycling_cost,
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
        fetch_fcr_capacity_prices,
        load_config,
    )

    config = load_config()
    start, end = "2024-01-01", "2024-01-08"

    day_ahead = fetch_day_ahead_prices(start, end)
    afrr = fetch_afrr_capacity_prices(start, end)
    fcr = fetch_fcr_capacity_prices(start, end)

    result = run_lp_dispatch(day_ahead, afrr["Up"], afrr["Down"], fcr, config)
    print(result.head(10))
    print(
        result[
            [
                "day_ahead_revenue",
                "afrr_revenue",
                "fcr_revenue",
                "cycling_cost",
                "net_revenue",
            ]
        ].sum()
    )
    both_active = (
        (result["charge_mwh"] > 1e-6) & (result["discharge_mwh"] > 1e-6)
    ).sum()
    print(f"Hours with simultaneous charge AND discharge: {both_active}")
