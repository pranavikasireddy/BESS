import logging

import pandas as pd

logger = logging.getLogger(__name__)

ISP_HOURS = 0.25  # Dutch Imbalance Settlement Period = 15 minutes


def apply_imbalance_overlay(
    dispatch: pd.DataFrame, day_ahead: pd.Series, imbalance: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """Reactive imbalance rule layered on top of an already-computed hourly
    dispatch. Only reads charge_mwh/discharge_mwh/soc_mwh from `dispatch`,
    so it's identical whether that came from heuristic.py or
    optimisation.py — any revenue difference stays attributable to the
    scheduling method, not to this layer.

    Decides using only the PREVIOUS quarter-hour's imbalance price, not the
    current one. TenneT's own documentation confirms the final settled
    price for a quarter is only known once that quarter closes — the live
    signal during the quarter is a delayed, indicative estimate, not
    something this project has real archived data for. Using the prior
    quarter's already-final price is the most recent genuinely-known
    information a real decision could be based on. Settlement still uses
    the ACTUAL current quarter's price, though — the decision is made on
    stale information, but real money settles at the real (possibly
    different) rate, which is exactly the execution risk a live operator
    faces and a same-quarter rule would hide.

    Deliberately skips TenneT's dual-pricing ("regulation state 2")
    periods (judged from the prior quarter, same as everything else here)
    — those exist specifically to penalise both long and short positions,
    so there's no edge to react to (confirmed against TenneT's Imbalance
    Pricing System documentation, not assumed).

    During single-priced periods, deviates from the baseline only when the
    prior quarter's imbalance price cleared a margin over/under the
    day-ahead price the baseline energy was already sold/bought at:
    - Long price attractive (> day-ahead + margin) -> discharge extra.
    - Short price attractive (< day-ahead - margin, often negative) ->
      charge extra.

    The hourly baseline is assumed spread evenly across its four quarters
    (day-ahead has no sub-hourly granularity to do otherwise). Bounded by
    the battery's power rating and SoC limits, net of the baseline's own
    quarter-share of both — NOT net of any aFRR/FCR reservation. That's a
    stated simplification: this overlay could in principle encroach on
    capacity already sold to aFRR/FCR. Left as a further refinement.
    """
    battery = config["battery"]
    power_mw = battery["power_mw"]
    energy_mwh = battery["energy_mwh"]
    efficiency = battery["round_trip_efficiency"]
    cycle_cost = battery["cycle_cost_eur_mwh"]
    margin = config["market"].get("imbalance_margin_eur_mwh", 5.0)

    hours = imbalance.index.floor("h")
    baseline_charge = (dispatch["charge_mwh"].reindex(hours).to_numpy()) / 4
    baseline_discharge = (dispatch["discharge_mwh"].reindex(hours).to_numpy()) / 4
    day_ahead_price = day_ahead.reindex(hours).to_numpy()
    long_price = imbalance["Long"].to_numpy()
    short_price = imbalance["Short"].to_numpy()
    long_price_prev = imbalance["Long"].shift(1).to_numpy()
    short_price_prev = imbalance["Short"].shift(1).to_numpy()
    single_priced_prev = long_price_prev == short_price_prev  # False for the first quarter (NaN)

    quarter_power_cap = power_mw * ISP_HOURS
    soc = 0.0  # battery starts empty, consistent with heuristic.py and optimisation.py

    records = []
    for i, t in enumerate(imbalance.index):
        remaining_power = max(0.0, quarter_power_cap - baseline_charge[i] - baseline_discharge[i])

        extra_charge = 0.0
        extra_discharge = 0.0
        if single_priced_prev[i]:
            if long_price_prev[i] > day_ahead_price[i] + margin:
                extra_discharge = max(0.0, min(remaining_power, soc))
            elif short_price_prev[i] < day_ahead_price[i] - margin:
                extra_charge = max(0.0, min(remaining_power, (energy_mwh - soc) / efficiency))

        soc += (
            baseline_charge[i] * efficiency
            - baseline_discharge[i]
            + extra_charge * efficiency
            - extra_discharge
        )
        soc = min(max(soc, 0.0), energy_mwh)  # guard against float drift at the bounds

        imbalance_revenue = extra_discharge * long_price[i] - extra_charge * short_price[i]
        cycling_cost = (extra_charge + extra_discharge) * cycle_cost

        records.append(
            {
                "timestamp": t,
                "extra_charge_mwh": extra_charge,
                "extra_discharge_mwh": extra_discharge,
                "soc_mwh": soc,
                "imbalance_revenue": imbalance_revenue,
                "imbalance_cycling_cost": cycling_cost,
                "imbalance_net_revenue": imbalance_revenue - cycling_cost,
            }
        )

    result = pd.DataFrame.from_records(records).set_index("timestamp")
    logger.info(
        "Imbalance overlay: %d quarters, %d reactive, total net revenue EUR %.0f",
        len(result),
        int(((result["extra_charge_mwh"] > 0) | (result["extra_discharge_mwh"] > 0)).sum()),
        result["imbalance_net_revenue"].sum(),
    )
    return result


if __name__ == "__main__":
    from src.data_pipeline import (
        fetch_afrr_capacity_prices,
        fetch_day_ahead_prices,
        fetch_imbalance_prices,
        load_config,
    )
    from src.heuristic import run_heuristic_dispatch

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = load_config()
    start, end = "2024-01-01", "2024-01-08"

    day_ahead = fetch_day_ahead_prices(start, end)
    afrr = fetch_afrr_capacity_prices(start, end)
    imbalance = fetch_imbalance_prices(start, end)

    dispatch = run_heuristic_dispatch(day_ahead, afrr["Up"], config)
    overlay = apply_imbalance_overlay(dispatch, day_ahead, imbalance, config)

    print(overlay.head(10))
    print(overlay[["imbalance_revenue", "imbalance_cycling_cost", "imbalance_net_revenue"]].sum())
    print(f"Base heuristic net revenue: EUR {dispatch['net_revenue'].sum():.0f}")
    print(
        f"Combined (heuristic + imbalance overlay): EUR "
        f"{dispatch['net_revenue'].sum() + overlay['imbalance_net_revenue'].sum():.0f}"
    )
