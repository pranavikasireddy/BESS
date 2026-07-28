import pandas as pd
import pytest

from src.heuristic import run_heuristic_dispatch


def _hourly_index() -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=24, freq="h", tz="Europe/Amsterdam")


def _config(power_mw=40.0, energy_mwh=1000.0, efficiency=0.9, cycle_cost=0.0, afrr_reserve=0.0):
    return {
        "battery": {
            "power_mw": power_mw,
            "energy_mwh": energy_mwh,
            "round_trip_efficiency": efficiency,
            "cycle_cost_eur_mwh": cycle_cost,
        },
        "market": {"aFRR_reserve_mw": afrr_reserve},
    }


def test_soc_efficiency_and_discharge_capping():
    """Charge 4 cheap hours, discharge 4 pricey hours, hand-verified numbers.

    Round-trip efficiency (0.9) means discharging the same MWh that was
    charged is physically impossible — the 4th discharge hour must be
    capped by whatever energy is actually left in the battery, not the
    full 40 MW power rating. This is the exact "charging then discharging
    loses exactly the efficiency amount" check.
    """
    index = _hourly_index()
    prices = [10] * 4 + [100] * 4 + [50] * 16  # hours 0-3 cheapest, 4-7 priciest, rest untouched
    day_ahead = pd.Series(prices, index=index, dtype=float)
    afrr_up = pd.Series(0.0, index=index)

    result = run_heuristic_dispatch(day_ahead, afrr_up, _config())

    assert result["charge_mwh"].iloc[0:4].tolist() == [40.0, 40.0, 40.0, 40.0]
    # SoC after each charge hour: 40*0.9=36 added each time
    expected_soc_after_charge = [36.0, 72.0, 108.0, 144.0]
    assert result["soc_mwh"].iloc[0:4].tolist() == pytest.approx(expected_soc_after_charge)

    # Discharges draw down 144 MWh at up to 40 MW/hour: 40, 40, 40, then
    # only 24 left for the 4th hour — capped by available energy, not power.
    assert result["discharge_mwh"].iloc[4:8].tolist() == pytest.approx([40.0, 40.0, 40.0, 24.0])
    assert result["soc_mwh"].iloc[7] == pytest.approx(0.0)

    # Untouched hours never move the battery.
    assert (result["charge_mwh"].iloc[8:] == 0.0).all()
    assert (result["discharge_mwh"].iloc[8:] == 0.0).all()


def test_revenue_accounting_identity():
    """net_revenue must equal its stated components exactly, and aFRR
    revenue must equal the fixed reservation times price every hour — a
    silent mismatch here would corrupt every downstream comparison.
    """
    index = _hourly_index()
    day_ahead = pd.Series([20, 80] * 12, index=index, dtype=float)
    afrr_up = pd.Series([5.0] * 24, index=index)
    config = _config(power_mw=50.0, cycle_cost=3.0, afrr_reserve=10.0)

    result = run_heuristic_dispatch(day_ahead, afrr_up, config)

    expected_net = result["day_ahead_revenue"] + result["afrr_revenue"] - result["cycling_cost"]
    assert result["net_revenue"].tolist() == pytest.approx(expected_net.tolist())
    assert result["afrr_revenue"].tolist() == pytest.approx([10.0 * 5.0] * 24)


def test_never_charges_and_discharges_in_the_same_hour():
    """The 4-cheapest/4-priciest hour sets must never overlap — if they
    did, the dispatch would be physically nonsensical.
    """
    index = _hourly_index()
    day_ahead = pd.Series(range(24), index=index, dtype=float)
    afrr_up = pd.Series(0.0, index=index)

    result = run_heuristic_dispatch(day_ahead, afrr_up, _config())

    both_active = (result["charge_mwh"] > 0) & (result["discharge_mwh"] > 0)
    assert not both_active.any()
