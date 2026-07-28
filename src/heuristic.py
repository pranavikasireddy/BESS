import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHARGE_HOURS_PER_DAY = 4
DISCHARGE_HOURS_PER_DAY = 4


def run_heuristic_dispatch(
    day_ahead: pd.Series, afrr_up: pd.Series, config: dict
) -> pd.DataFrame:
    """Fixed-rule dispatch: charge the 4 cheapest hours/day, discharge the 4
    priciest, reserve a fixed aFRR block every hour. This is the naive
    benchmark the LP gets measured against — a mechanical rule, no
    optimisation.

    Deliberate simplifications versus the LP (not oversights):
    - aFRR reservation is Up-direction only, fixed MW, every hour. A fixed
      rule has no basis to choose an Up/Down split the way the LP can.
    - Enforces the power-rating headroom for aFRR (available charge/discharge
      power is reduced by the reserved MW) but not TenneT's LER rule that
      reserved capacity must also be deliverable as sustained energy for a
      full ISP. Left for the LP.
    """
    battery = config["battery"]
    power_mw = battery["power_mw"]
    energy_mwh = battery["energy_mwh"]
    efficiency = battery["round_trip_efficiency"]
    cycle_cost = battery["cycle_cost_eur_mwh"]
    afrr_reserve_mw = config["market"]["aFRR_reserve_mw"]

    available_power_mw = power_mw - afrr_reserve_mw

    df = pd.DataFrame({"day_ahead_price": day_ahead, "afrr_up_price": afrr_up}).dropna()

    records = []
    soc = 0.0

    for _, day_df in df.groupby(df.index.date):
        cheapest = set(day_df["day_ahead_price"].nsmallest(CHARGE_HOURS_PER_DAY).index)
        priciest = set(day_df["day_ahead_price"].nlargest(DISCHARGE_HOURS_PER_DAY).index)

        for ts, row in day_df.iterrows():
            charge_mwh = 0.0
            discharge_mwh = 0.0

            if ts in cheapest:
                charge_mwh = max(0.0, min(available_power_mw, (energy_mwh - soc) / efficiency))
                soc += charge_mwh * efficiency
            elif ts in priciest:
                discharge_mwh = max(0.0, min(available_power_mw, soc))
                soc -= discharge_mwh

            day_ahead_revenue = (
                discharge_mwh * row["day_ahead_price"] - charge_mwh * row["day_ahead_price"]
            )
            afrr_revenue = afrr_reserve_mw * row["afrr_up_price"]
            cycling_cost = (charge_mwh + discharge_mwh) * cycle_cost

            records.append(
                {
                    "timestamp": ts,
                    "charge_mwh": charge_mwh,
                    "discharge_mwh": discharge_mwh,
                    "soc_mwh": soc,
                    "day_ahead_revenue": day_ahead_revenue,
                    "afrr_revenue": afrr_revenue,
                    "cycling_cost": cycling_cost,
                    "net_revenue": day_ahead_revenue + afrr_revenue - cycling_cost,
                }
            )

    result = pd.DataFrame.from_records(records).set_index("timestamp")
    logger.info(
        "Heuristic dispatch: %d hours, total net revenue EUR %.0f",
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

    result = run_heuristic_dispatch(day_ahead, afrr["Up"], config)
    print(result.head(10))
    print(result[["day_ahead_revenue", "afrr_revenue", "cycling_cost", "net_revenue"]].sum())
