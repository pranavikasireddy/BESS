# Methodology

## Reference asset

Sized to EP NL and Eneco's real battery project at the Enecogen power plant, Europoort, Rotterdam port: 50 MW / 200 MWh (4-hour duration), online 2027, EP NL and Eneco each holding a 50% stake. Specs verified directly against EP NL's own announcement and Energy Storage News, not assumed from memory:
- [EP NL: "Eneco and EP NL invest in 200 MWh battery storage project at Enecogen"](https://epnl.nl/en/news-market-information/ep-nl-en-eneco-invest-in-200-mwh-battery-storage-project-at-enecogen-in-rotterdam-port/)
- [Energy Storage News: "Eneco and EPH partner on 50MW/200MWh Netherlands BESS"](https://www.energy-storage.news/eneco-and-eph-partner-on-50mw-200mwh-netherlands-bess/)

Battery isn't operational until 2027, so all dispatch is against real historical Dutch market data applied to a hypothetical operating history — not real Enecogen operational data.

## Data sources

All market data pulled from the **ENTSO-E Transparency Platform** via the `entsoe-py` client (`src/data_pipeline.py`), using a personal free API key. Four series, all NL bidding zone:

| Series | ENTSO-E query | Native resolution | Notes |
|---|---|---|---|
| Day-ahead price | `query_day_ahead_prices` | Hourly (until 2025-09-30) | Switches to 15-min from 2025-10-01 (SDAC 15-minute MTU) — see Backtest Window below |
| aFRR capacity (Up/Down) | `query_contracted_reserve_prices`, processType A51 | 15-min, resampled to hourly | Asymmetric — separate Up/Down prices |
| FCR capacity (Symmetric) | `query_contracted_reserve_prices`, processType A52 | 15-min, resampled to hourly | Single price — responds to frequency deviations in either direction, can't be bid asymmetrically |
| Imbalance price (Long/Short) | `query_imbalance_prices` | Native 15-min (kept, not resampled) | Long/Short usually equal (single pricing); diverge during TenneT's dual-pricing "regulation state 2" |

## Markets modelled

**Core LP objective (co-optimised):** day-ahead energy arbitrage, asymmetric aFRR capacity (Up/Down), symmetric FCR capacity. All three genuinely known before dispatch — NL's day-ahead auction clears ~noon the day before delivery, and both capacity auctions run on the same d-1 basis, so scheduling against them isn't forecasting.

**Reactive overlay (not in the core objective):** imbalance settlement, layered identically on top of whichever schedule (heuristic or LP) already exists, using only the *previous* quarter-hour's price — the true settled price for a quarter isn't known until that quarter closes (confirmed against TenneT's own documentation on real-time signal delay), so using the current quarter's price to decide that same quarter's action would be look-ahead bias.

**Not modelled**, with reasons:
- **Intraday** — continuous order-book microstructure, not hourly/quarterly clearing; a fundamentally different (and much harder) modelling problem, and granular order-book data isn't freely available the way day-ahead is.
- **aFRR/mFRR activation energy** — uncertain at scheduling time, same foresight problem as imbalance. Only capacity (reservation) revenue is modelled, not activation.
- **Capacity market payments** — the Netherlands doesn't have a capacity market.
- **Congestion management** — planned as a separate, standalone comparison (identical model run in a congested vs. uncongested zone), not blended into the core objective, since it's a siting question rather than a per-hour dispatch decision. Not yet built.

## Dispatch methods

**Heuristic** (`src/heuristic.py`): fixed rule, no optimisation. Charges the 4 cheapest day-ahead hours per day, discharges the 4 priciest, reserves a fixed 10 MW for aFRR (Up-direction only — a fixed rule has no basis to choose an asymmetric split) every hour. Efficiency applied on the charge leg only; charge/discharge amounts clip at SoC and power-rating bounds rather than being skipped outright. Does not enforce TenneT's LER rule (see below) — only respects power-rating headroom, not energy-backing. This is a deliberate simplification for the benchmark, not an oversight.

**LP / MILP** (`src/optimisation.py`, PuLP + CBC): co-optimises the three core markets. Technically a MILP: aFRR/FCR capacity variables are integer (see "1 MW bid increments" below); charge/discharge remain continuous, and no binary charge/discharge-exclusivity variable was added, because round-trip efficiency loss alone was confirmed (empirically, not assumed) to keep the plain LP from ever charging and discharging in the same hour, across every test performed.

Two variants:
- **Full-horizon** — sees the entire price series at once. A perfect-foresight upper bound, not a realistic strategy; reported as a ceiling for context, not as the headline number.
- **Rolling-horizon** — solves one calendar day at a time, using only that day's own prices. The realistic, defensible number. On the one-week test slice, rolling-horizon captures 91.5% of the full-horizon ceiling (€128,604 vs. €140,533).

**Terminal SoC boundary (rolling-horizon only):** solving each day in isolation makes an unconstrained LP want to end every day at 0 SoC — nothing to gain by holding energy it can't see tomorrow's value for. Fixed by requiring every day to both start and end at a fixed 50% SoC (`config.yaml: rolling_horizon_boundary_soc_frac`), except day one, which starts at 0 like every other method here, for a consistent initial condition across the whole comparison.

## Key technical findings (verified against real data, not assumed)

- **TenneT's LER (Limited Energy Resource) rule**: reserved aFRR/FCR capacity must be backed by enough real stored energy (Up-direction) or headroom (Down-direction) to sustain it for a full 15-minute ISP, not just fit under the power rating. Without this constraint, the LP could reserve Up capacity from an empty battery for free — caught by inspecting SoC=0 alongside nonzero reservation in early test output.
- **aFRR/FCR block-granularity**: reservation must stay constant within each real bidding block — 24-hour blocks before ~2025-07, 4-hour blocks after (both detected empirically from runs of identical price in the data, not a hardcoded cutover date, so the model self-adapts to any future TenneT change). Confirmed via direct query that pre-2025 aFRR pricing really is one flat price per calendar day (100% of sampled days showed exactly 1 unique price across 96 quarter-hour rows).
- **aFRR/FCR resampling artifact**: ENTSO-E's feed reports the very first quarter-hour at each block boundary as the *average* of the outgoing and incoming block's price (a data-stitching artifact, not a real price — confirmed by checking it against the exact midpoint of the adjacent block prices). Fixed by resampling to hourly via median rather than first-value, which is robust to this single corrupted quarter without losing any real signal.
- **1 MW aFRR/FCR bid increments**: the real market requires integer MW bids. Tested both ways before deciding: enforcing this costs only 0.17% of revenue (€239 of €140,533 on the test week) with no meaningful solve-time impact, so it's kept on by default as both accurate and cheap.
- **NL imbalance pricing is ~90% single-price, ~10% dual-price**: confirmed directly from TenneT's "Imbalance Pricing System" documentation. Dual pricing ("regulation state 2") applies specifically when both upward and downward regulation were activated within the same ISP, and exists deliberately to penalise both long and short positions — closing the profit opportunity in exactly the periods most vulnerable to gaming (large, sustained one-directional deviations are easy to detect statistically; short-lived mixed-direction periods are much harder to attribute to a single actor). The imbalance overlay skips dual-priced periods entirely rather than reacting to them.
- **Grid transport/connection cost**: fixed annual capacity charge (~€60,650/MW/year, TenneT 2024 rate — provisional, unverified against the primary tariff schedule, and excludes a second real monthly max-demand component). Applied as a lump-sum deduction after dispatch (`src/costs.py`), since it's capacity-based and doesn't affect hourly dispatch incentives. On the test week, this cost alone exceeds the heuristic's gross arbitrage-only revenue — it's the combined stack (aFRR + FCR + imbalance) that clears the bar, not day-ahead arbitrage on its own.

## Backtest window

2021-01-01 to 2025-09-30. The end date is deliberate: on 2025-10-01, the entire European day-ahead market (NL included) switched from hourly to 15-minute price intervals (SDAC 15-minute MTU go-live). Ending the backtest just before this keeps the whole window at a single, consistent hourly resolution matching the LP's design. Extending past this date would require resampling the 15-minute prices down to hourly, which would flatten exactly the intra-hour volatility a battery is meant to exploit — a worse trade-off than losing three months of the most recent data.

## Open items before final results

- Grid transport rate needs verification against TenneT's primary tariff schedule (tennet.eu/tariffs), and the second (monthly max-demand) cost component needs adding.
- The imbalance-overlay margin (`config.yaml: imbalance_margin_eur_mwh`) was tuned on a single one-week sample — needs re-tuning on a training slice (e.g. 2021-2023) with headline results reported on a holdout slice (e.g. 2024-2025) never used for tuning, to avoid overfitting the reported number.
- Full 2021-2025 backtest not yet run for either method — all figures above are from a one-week test slice.
