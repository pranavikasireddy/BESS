import pandas as pd


def identify_curtailed_hours(day_ahead: pd.Series, curtailment_frac: float) -> pd.Series:
    """Flags the lowest-price curtailment_frac of hours within each
    calendar year as TDTR-restricted, standing in for TenneT's actual
    transport-restriction schedule.

    No public hourly historical record of when TenneT restricted transport
    exists — its capacity map is a current-status snapshot, not a time
    series — so low and negative day-ahead prices are used as a proxy for
    the oversupply conditions that drive restrictions: both stem from the
    same cause, more generation wanting to inject than the local grid or
    the wholesale market can absorb.

    Computed per calendar year rather than over the whole window, since
    TDTR's restriction share is an annual guarantee and price levels
    shifted substantially across the backtest period (e.g. the 2022
    energy crisis).
    """
    threshold = day_ahead.groupby(day_ahead.index.year).transform(
        lambda s: s.quantile(curtailment_frac)
    )
    return day_ahead <= threshold


def hourly_profitability(dispatch: pd.DataFrame) -> pd.Series:
    """Total per-hour revenue from the core three markets, net of cycling
    cost. Includes aFRR/FCR capacity reservation revenue, which
    curtailment doesn't touch — see _solve_lp in optimisation.py, where a
    curtailed hour only fixes the discharge variable to 0, leaving
    afrr_up_mw/afrr_down_mw/fcr_mw free. Used to separate two different
    reasons a curtailed hour might cost nothing: the battery earning
    nothing there across any market, versus the battery specifically not
    discharging there while still earning capacity revenue.
    """
    return (
        dispatch["day_ahead_revenue"]
        + dispatch["afrr_revenue"]
        + dispatch["fcr_revenue"]
        - dispatch["cycling_cost"]
    )


def curtailment_profitability_overlap(
    dispatch: pd.DataFrame, curtailed: pd.Series, profitability_frac: float = 0.15
) -> dict:
    """Checks how much identify_curtailed_hours' price-proxy curtailed
    hours overlap with the battery's own most- and least-profitable
    hours, ranked per calendar year for the same reason the curtailment
    proxy itself is (price levels shifted substantially across the
    backtest period).

    Also reports what fraction of curtailed hours already had zero
    discharge in the baseline dispatch — a near-tautological check
    (a profit-maximising LP structurally avoids discharging into the
    cheapest hours by definition), included as a contrast against the
    profitability overlap above, which is not guaranteed by the model's
    structure and reflects whether day-ahead and aFRR/FCR price levels
    actually move together in this data.
    """
    profitability = hourly_profitability(dispatch)
    high_threshold = profitability.groupby(profitability.index.year).transform(
        lambda s: s.quantile(1 - profitability_frac)
    )
    low_threshold = profitability.groupby(profitability.index.year).transform(
        lambda s: s.quantile(profitability_frac)
    )
    most_profitable = profitability >= high_threshold
    least_profitable = profitability <= low_threshold

    curtailed = curtailed.reindex(profitability.index, fill_value=False)
    n_curtailed = int(curtailed.sum())
    return {
        "n_curtailed_hours": n_curtailed,
        "overlap_with_most_profitable_frac": (curtailed & most_profitable).sum() / n_curtailed,
        "overlap_with_least_profitable_frac": (curtailed & least_profitable).sum() / n_curtailed,
        "curtailed_hours_zero_discharge_frac": (
            dispatch.loc[curtailed, "discharge_mwh"] == 0
        ).mean(),
    }


if __name__ == "__main__":
    import logging
    from pathlib import Path

    from src.data_pipeline import load_config
    from src.optimisation import run_rolling_horizon_lp

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    data_dir = Path("data/raw")
    day_ahead = pd.read_parquet(data_dir / "day_ahead.parquet")["day_ahead"]
    afrr = pd.read_parquet(data_dir / "afrr.parquet")
    fcr = pd.read_parquet(data_dir / "fcr.parquet")["fcr"]
    config = load_config()

    dispatch = run_rolling_horizon_lp(day_ahead, afrr["Up"], afrr["Down"], fcr, config)
    curtailed = identify_curtailed_hours(day_ahead, config["grid"]["tdtr_curtailment_frac"])
    overlap = curtailment_profitability_overlap(dispatch, curtailed)

    most_frac = overlap["overlap_with_most_profitable_frac"]
    least_frac = overlap["overlap_with_least_profitable_frac"]
    zero_discharge_frac = overlap["curtailed_hours_zero_discharge_frac"]
    print(f"\nCurtailed hours: {overlap['n_curtailed_hours']:,}")
    print(f"  overlap with most-profitable 15% (by year): {most_frac:.1%}")
    print(f"  overlap with least-profitable 15% (by year): {least_frac:.1%}")
    print(f"  curtailed hours with zero discharge in baseline: {zero_discharge_frac:.1%}")
