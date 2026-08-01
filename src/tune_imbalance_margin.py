import logging

import pandas as pd

from src.imbalance import apply_imbalance_overlay

logger = logging.getLogger(__name__)

DEFAULT_MARGIN_CANDIDATES = [5, 15, 30, 50, 75, 125, 175, 250, 400, 600]


def build_walk_forward_folds(
    start, end, train_months: int = 24, test_months: int = 6, step_months: int = 6
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Rolling train/test fold boundaries for walk-forward validation.

    Each fold trains on `train_months` of data and tests on the
    `test_months` immediately following it, then the window slides
    forward by `step_months`. Rolling rather than expanding: the backtest
    window spans a known structural break (the 2022 energy crisis), and
    an expanding window would keep that atypical period permanently
    blended into every later fold's training data instead of letting it
    age out. The final fold's test period is truncated to `end` if a
    full `test_months` would run past it.
    """
    folds = []
    train_start = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        if train_end >= end_ts:
            break
        test_end = min(train_end + pd.DateOffset(months=test_months), end_ts)
        folds.append((train_start, train_end, test_end))
        train_start = train_start + pd.DateOffset(months=step_months)
    return folds


def score_margin(
    dispatch: pd.DataFrame,
    day_ahead: pd.Series,
    imbalance: pd.DataFrame,
    margin: float,
    config: dict,
) -> float:
    """Net imbalance-overlay revenue for one candidate margin, over
    whichever slice of dispatch/day_ahead/imbalance is passed in.
    """
    cfg = {**config, "market": {**config["market"], "imbalance_margin_eur_mwh": margin}}
    overlay = apply_imbalance_overlay(dispatch, day_ahead, imbalance, cfg)
    return overlay["imbalance_net_revenue"].sum()


def run_walk_forward_validation(
    dispatch: pd.DataFrame,
    day_ahead: pd.Series,
    imbalance: pd.DataFrame,
    config: dict,
    margin_candidates: list[float] = DEFAULT_MARGIN_CANDIDATES,
    train_months: int = 24,
    test_months: int = 6,
    step_months: int = 6,
) -> pd.DataFrame:
    """Walk-forward validation for `market.imbalance_margin_eur_mwh`: for
    each rolling train/test fold, picks whichever candidate margin
    maximises net revenue on the train slice, then scores that choice on
    the held-out test slice.

    `dispatch` must already cover the whole window being validated — the
    core LP/heuristic schedule doesn't depend on the margin, so it only
    needs solving once regardless of how many folds or candidates get
    evaluated here (the margin only affects the overlay applied on top).

    Returns one row per fold with the margin chosen on that fold's
    training data, plus every candidate's out-of-sample score on that
    fold's test data — not just the chosen one — so a walk-forward result
    can be compared against simply fixing any single margin for the whole
    period. That comparison matters: on this project's data, a margin
    picked by revenue-maximizing search on one training slice (2021-2023)
    did not generalize, and re-running the same search on a rolling basis
    didn't fix it either — see methodology.md for the full result.
    """
    folds = build_walk_forward_folds(
        dispatch.index.min(), dispatch.index.max(), train_months, test_months, step_months
    )
    rows = []
    for train_start, train_end, test_end in folds:
        train_scores = {
            m: score_margin(
                dispatch.loc[train_start:train_end],
                day_ahead.loc[train_start:train_end],
                imbalance.loc[train_start:train_end],
                m,
                config,
            )
            for m in margin_candidates
        }
        best_margin = max(train_scores, key=train_scores.get)

        test_scores = {
            m: score_margin(
                dispatch.loc[train_end:test_end],
                day_ahead.loc[train_end:test_end],
                imbalance.loc[train_end:test_end],
                m,
                config,
            )
            for m in margin_candidates
        }
        row = {
            "train_start": train_start,
            "train_end": train_end,
            "test_end": test_end,
            "best_train_margin": best_margin,
            "walk_forward_test_revenue": test_scores[best_margin],
        }
        row.update({f"test_revenue_margin_{m}": v for m, v in test_scores.items()})
        rows.append(row)
        logger.info(
            "Fold [test %s-%s]: best_train_margin=%s, walk_forward_test_revenue=EUR %.0f",
            test_end.date(),
            train_end.date(),
            best_margin,
            test_scores[best_margin],
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from pathlib import Path

    from src.data_pipeline import load_config
    from src.optimisation import run_rolling_horizon_lp

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    data_dir = Path("data/raw")
    day_ahead = pd.read_parquet(data_dir / "day_ahead.parquet")["day_ahead"]
    afrr = pd.read_parquet(data_dir / "afrr.parquet")
    fcr = pd.read_parquet(data_dir / "fcr.parquet")["fcr"]
    imbalance = pd.read_parquet(data_dir / "imbalance.parquet")
    config = load_config()

    dispatch = run_rolling_horizon_lp(day_ahead, afrr["Up"], afrr["Down"], fcr, config)
    results = run_walk_forward_validation(dispatch, day_ahead, imbalance, config)

    print(
        results[
            [
                "train_start",
                "train_end",
                "test_end",
                "best_train_margin",
                "walk_forward_test_revenue",
            ]
        ]
    )

    candidate_cols = [c for c in results.columns if c.startswith("test_revenue_margin_")]
    totals = results[candidate_cols].sum().sort_values(ascending=False)
    print("\nOut-of-sample net revenue by fixed margin, summed across all folds:")
    print(totals.to_string())
    walk_forward_total = results["walk_forward_test_revenue"].sum()
    print(f"\nWalk-forward (re-tuned each fold): EUR {walk_forward_total:,.0f}")
