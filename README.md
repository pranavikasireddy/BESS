# BESS Revenue-Stacking Optimisation (Dutch Market)

**Status: in progress.** Core method comparison (heuristic vs. LP) is built and tested; forward-looking solar-cannibalization scenarios and the congestion comparison are not yet built. Numbers below are from a one-week test slice (2024-01-01 to 2024-01-08), not the full 2021–2025 backtest — treat them as sanity checks on the method, not final results.

## Research question

For a Dutch grid-scale battery (sized to EP NL's Enecogen asset — 50 MW / 200 MWh, 4-hour duration), how much of the achievable revenue-stacking value does a simple rule-based dispatch strategy capture compared to a linear-programming optimiser that co-optimises day-ahead arbitrage against aFRR and FCR capacity reservation? And separately: how does that value change as Dutch solar penetration increases toward the government's 2030 target?

## Method, in one paragraph

Two dispatch methods — a fixed-rule **heuristic** and a **MILP** (PuLP + CBC, full-horizon and rolling-horizon variants) — are run against identical real Dutch market data (day-ahead, aFRR, FCR, all from ENTSO-E), so any revenue difference is attributable purely to the scheduling method. A reactive, foresight-free **imbalance overlay** can be layered on top of either method's schedule. Full formulation, every constraint, and every data quirk found along the way (LER rule, block-granularity, resampling artifacts, single/dual imbalance pricing) are documented in **[methodology.md](methodology.md)** — this file stays intentionally short; that one has the detail.

## Preliminary findings (one-week test slice)

- Rolling-horizon LP captures **91.5%** of the full-horizon perfect-foresight ceiling (€128,604 vs. €140,533) — a believable gap given day-ahead and capacity products are both settled daily anyway.
- The LP substantially outperforms the heuristic even before imbalance is added, because the heuristic can't dynamically trade off arbitrage against aFRR/FCR reservation.
- Grid transport cost is large enough that the heuristic's gross arbitrage-only revenue for the test week doesn't cover its share of the annual transport charge — it's the combined revenue stack (aFRR + FCR + imbalance) that clears the bar, not arbitrage alone.

Full reasoning behind each of these, plus every simplification and open item, is in **[methodology.md](methodology.md)**, not duplicated here.

## How to run it

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Add your ENTSO-E API key to a `.env` file: `ENTSOE_API_KEY=your_token_here`.

```powershell
python -m src.heuristic        # heuristic dispatch on a test slice
python -m src.optimisation     # full-horizon + rolling-horizon LP
python -m src.imbalance        # imbalance overlay on top of the heuristic
pytest                          # unit tests (SoC/efficiency/accounting logic)
```

## Repository structure

```
src/
  data_pipeline.py    # ENTSO-E fetch functions: day-ahead, aFRR, FCR, imbalance
  heuristic.py         # fixed-rule dispatch (benchmark)
  optimisation.py       # PuLP/CBC LP — full-horizon and rolling-horizon
  imbalance.py           # reactive imbalance overlay
  costs.py                # fixed grid transport cost deduction
tests/
  test_heuristic.py        # SoC/efficiency/revenue-accounting unit tests
  test_optimisation.py      # LER, block-granularity, terminal SoC unit tests
config.yaml                 # battery specs, market params, backtest window
methodology.md                # detailed methodology (in progress)
```
