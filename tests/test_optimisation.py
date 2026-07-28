import pandas as pd
import pytest

from src.optimisation import run_lp_dispatch, run_rolling_horizon_lp


def _hourly_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="h", tz="Europe/Amsterdam")


def _config(power_mw=50.0, energy_mwh=200.0, efficiency=0.88, cycle_cost=3.0, boundary_frac=0.5):
    return {
        "battery": {
            "power_mw": power_mw,
            "energy_mwh": energy_mwh,
            "round_trip_efficiency": efficiency,
            "cycle_cost_eur_mwh": cycle_cost,
        },
        "market": {"rolling_horizon_boundary_soc_frac": boundary_frac},
    }


def test_ler_blocks_reservation_from_empty_battery():
    """With SoC=0 at the very first hour, the LP cannot reserve any Up
    capacity (aFRR or FCR) since there's no stored energy to back it — the
    LER constraint must force this to zero even at an absurdly high price
    that would otherwise make it very attractive to reserve.
    """
    index = _hourly_index(4)
    day_ahead = pd.Series([50.0] * 4, index=index)
    afrr_up = pd.Series([1000.0] * 4, index=index)
    afrr_down = pd.Series([0.0] * 4, index=index)
    fcr = pd.Series([0.0] * 4, index=index)

    result = run_lp_dispatch(day_ahead, afrr_up, afrr_down, fcr, _config())

    assert result["afrr_up_mw"].iloc[0] == pytest.approx(0.0)


def test_block_granularity_enforced():
    """aFRR reservation must stay constant within a detected price block —
    a hard structural constraint, true regardless of what the economically
    optimal reservation level happens to be.
    """
    index = _hourly_index(4)
    day_ahead = pd.Series([50.0] * 4, index=index)
    # hours 0-1 are one block (price 10), hours 2-3 are another (price 100)
    afrr_up = pd.Series([10.0, 10.0, 100.0, 100.0], index=index)
    afrr_down = pd.Series([0.0] * 4, index=index)
    fcr = pd.Series([0.0] * 4, index=index)

    result = run_lp_dispatch(day_ahead, afrr_up, afrr_down, fcr, _config())

    assert result["afrr_up_mw"].iloc[0] == pytest.approx(result["afrr_up_mw"].iloc[1])
    assert result["afrr_up_mw"].iloc[2] == pytest.approx(result["afrr_up_mw"].iloc[3])


def test_rolling_horizon_hits_terminal_soc_boundary():
    """Every day in the rolling-horizon solve must both start and end at
    the configured boundary SoC (day one starts at 0 instead, like every
    other method in this project). Without this constraint, isolated daily
    solves would always want to drain to 0 by day's end.
    """
    index = _hourly_index(48)  # 2 full days
    day_ahead = pd.Series([50.0] * 48, index=index)
    afrr_up = pd.Series([0.0] * 48, index=index)
    afrr_down = pd.Series([0.0] * 48, index=index)
    fcr = pd.Series([0.0] * 48, index=index)
    config = _config()

    result = run_rolling_horizon_lp(day_ahead, afrr_up, afrr_down, fcr, config)

    boundary_soc = (
        config["battery"]["energy_mwh"] * config["market"]["rolling_horizon_boundary_soc_frac"]
    )
    assert result["soc_mwh"].iloc[23] == pytest.approx(boundary_soc)  # end of day 1
    assert result["soc_mwh"].iloc[47] == pytest.approx(boundary_soc)  # end of day 2


def test_no_simultaneous_charge_and_discharge():
    """Confirms the finding this project relies on to justify NOT adding a
    MILP binary exclusivity constraint: round-trip efficiency loss alone
    should keep the plain LP from ever charging and discharging in the
    same hour.
    """
    index = _hourly_index(24)
    day_ahead = pd.Series([10] * 4 + [100] * 4 + [50] * 16, index=index, dtype=float)
    afrr_up = pd.Series([5.0] * 24, index=index)
    afrr_down = pd.Series([5.0] * 24, index=index)
    fcr = pd.Series([2.0] * 24, index=index)

    result = run_lp_dispatch(day_ahead, afrr_up, afrr_down, fcr, _config())

    both_active = (result["charge_mwh"] > 1e-6) & (result["discharge_mwh"] > 1e-6)
    assert not both_active.any()
