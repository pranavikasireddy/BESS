# BESS Revenue-Stacking Optimisation (Dutch Market)

**Status: in progress.** Core method comparison (heuristic vs. LP) is built and tested; forward-looking solar-cannibalization scenarios and the congestion comparison are not yet built. Numbers below are from a one-week test slice (2024-01-01 to 2024-01-08), not the full 2021–2025 backtest — treat them as sanity checks on the method, not final results.

## Research question

For a Dutch grid-scale battery (sized to EP NL's Enecogen asset — 50 MW / 200 MWh, 4-hour duration), how much of the achievable revenue-stacking value does a simple rule-based dispatch strategy capture compared to a linear-programming optimiser that co-optimises day-ahead arbitrage against aFRR and FCR capacity reservation? And separately: how does that value change as Dutch solar penetration increases toward the government's 2030 target?

## Method

Two dispatch methods are compared on identical price data, so any revenue difference is attributable purely to the scheduling method, not the market data:

- **Heuristic** (`src/heuristic.py`): a fixed rule — charge the 4 cheapest day-ahead hours each day, discharge the 4 priciest, reserve a fixed 10 MW for aFRR (Up-direction only) every hour. Fast, transparent, but can't handle the aFRR-vs-arbitrage trade-off.
- **LP** (`src/optimisation.py`, PuLP + CBC): co-optimises day-ahead arbitrage, asymmetric aFRR capacity (Up/Down), and symmetric FCR capacity reservation. Run two ways:
  - **Full-horizon** — sees the entire price series at once; a perfect-foresight ceiling, not a realistic strategy.
  - **Rolling-horizon** — re-solves one calendar day at a time, using only prices genuinely known by the time of dispatch (NL day-ahead and capacity auctions both clear on a d-1 basis, so this isn't forecasting). The realistic, defensible number.

A reactive **imbalance overlay** (`src/imbalance.py`) can be layered on top of either method's schedule — it only uses the *previous* quarter-hour's imbalance price (not the current one, which isn't finalized until after settlement) to decide whether to deviate from the baseline, avoiding look-ahead bias.

## Markets modelled

| Market | Role |
|---|---|
| Day-ahead energy | Core LP objective — arbitrage |
| aFRR capacity (asymmetric Up/Down) | Core LP objective, co-optimised with arbitrage; block-granularity enforced (24h blocks pre-2025-07, 4h after, detected empirically from the price data) |
| FCR capacity (symmetric) | Core LP objective, own block-granularity |
| Imbalance | Reactive overlay only, identical across both methods, foresight-free |
| Grid transport/connection cost | Fixed annual capacity charge (`src/costs.py`), deducted post-hoc — not dispatch-dependent |

**Explicitly not modelled**, with reasons: intraday (continuous order-book microstructure, not hourly/quarterly clearing — a different problem, and granular data isn't freely available); aFRR/mFRR activation energy (uncertain at scheduling time, same foresight issue as imbalance); Dutch capacity market payments (doesn't exist yet in NL). Congestion management is planned as a separate standalone comparison (congested vs. uncongested siting), not blended into the core objective — not yet built.

## Preliminary findings (one-week test slice)

- Rolling-horizon LP captures **91.5%** of the full-horizon perfect-foresight ceiling (€128,604 vs. €140,533) — a believable gap given day-ahead and capacity products are both settled daily anyway.
- The LP substantially outperforms the heuristic even before imbalance is added, because the heuristic can't dynamically trade off arbitrage against aFRR/FCR reservation.
- Zero hours of simultaneous charge-and-discharge in either LP variant — confirmed empirically rather than assumed, which is why no MILP binary exclusivity constraint was added.
- Grid transport cost is large enough that the heuristic's gross arbitrage-only revenue for the test week doesn't cover its share of the annual transport charge — it's the combined revenue stack (aFRR + FCR + imbalance) that clears the bar, not arbitrage alone.

## Known simplifications / open items

- aFRR/FCR bid granularity (real-world 1 MW minimum increments) isn't enforced — variables are continuous. Likely immaterial given typical optimal values are well above 1 MW, but not verified.
- Grid transport cost uses an approximate, unverified rate (see `config.yaml`); a second real cost component (monthly max-demand charge) isn't included yet.
- The imbalance-overlay margin (`config.yaml: imbalance_margin_eur_mwh`) was tuned on a single one-week sample — provisional pending a proper train/holdout recalibration once the full backtest is running.
- TenneT's LER (Limited Energy Resource) rule is enforced for the LP but not the heuristic, which only respects power-rating headroom, not energy-backing — a deliberate simplification, documented in `heuristic.py`.

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
config.yaml                 # battery specs, market params, backtest window
methodology.md                # detailed methodology (in progress)
```
