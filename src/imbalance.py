import logging

import pandas as pd

logger = logging.getLogger(__name__)

ISP_HOURS = 0.25  # Dutch Imbalance Settlement Period = 15 minutes


def apply_imbalance_overlay(
    dispatch: pd.DataFrame, day_ahead: pd.Series, imbalance: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """Reactive imbalance rule layered on top of an already-computed hourly
    dispatch. Only reads charge_mwh/discharge_mwh/soc_mwh from `dispatch`,
    so it works identically regardless of whether that came from
    heuristic.py or optimisation.py.

    Decisions use only the PREVIOUS quarter-hour's imbalance price, not
    the current one: the settled price for a given quarter isn't known
    until that quarter closes, so reacting to a quarter's own price would
    assume information not actually available at decision time. Actual
    settlement still uses the real current-quarter price, so a decision
    made on the prior quarter's signal can turn out better or worse than
    expected — the same execution risk a live operator faces.

    Skips TenneT's dual-pricing ("regulation state 2") periods, which
    apply when both upward and downward regulation were activated within
    the same settlement period. Dual pricing penalises both long and
    short positions in those periods, so there's no profitable direction
    to react to.

    During single-priced periods, the rule deviates from the baseline
    schedule only when the prior quarter's imbalance price clears both
    the margin AND the cycling cost over/under the day-ahead price the
    baseline energy was already sold/bought at — the trade must be worth
    doing net of wear, not just bigger than the noise-filter margin:
    - Long price attractive (> day-ahead + margin + cycle_cost) ->
      discharge extra.
    - Short price attractive (< day-ahead - margin - cycle_cost, often
      negative) -> charge extra.

    The hourly baseline is spread evenly across its four quarters
    (day-ahead has no sub-hourly granularity). Extra charge/discharge is
    bounded by the battery's power rating and SoC limits, net of the
    baseline's own quarter-share of power — but not net of any aFRR/FCR
    reservation, so this overlay can in principle draw on capacity already
    committed to those products.
    """
    battery = config["battery"]
    power_mw = battery["power_mw"]
    energy_mwh = battery["energy_mwh"]
    efficiency = battery["round_trip_efficiency"]
    cycle_cost = battery["cycle_cost_eur_mwh"]
    margin = config["market"].get("imbalance_margin_eur_mwh", 5.0)

    # Floor via UTC, not local time: local time repeats an hour every DST
    # fall-back (e.g. 2021-10-31 02:00 occurs twice), which makes flooring
    # directly in local time genuinely ambiguous. UTC has no DST, so
    # converting first sidesteps the ambiguity entirely.
    hours = imbalance.index.tz_convert("UTC").floor("h").tz_convert("Europe/Amsterdam")
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
            if long_price_prev[i] > day_ahead_price[i] + margin + cycle_cost:
                extra_discharge = max(0.0, min(remaining_power, soc))
            elif short_price_prev[i] < day_ahead_price[i] - margin - cycle_cost:
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
