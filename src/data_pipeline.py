import logging
import os

import pandas as pd
import yaml
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

COUNTRY_CODE = "NL"
AFRR_PROCESS_TYPE = "A51"  # A51 = aFRR, A47 = mFRR (ENTSO-E processType code)
AFRR_MARKET_AGREEMENT = "A01"  # A01 = Daily contract (covers both 24h and 4h block eras)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _get_client() -> EntsoePandasClient:
    return EntsoePandasClient(api_key=os.environ["ENTSOE_API_KEY"])


def fetch_day_ahead_prices(start: str, end: str) -> pd.Series:
    """Fetch NL day-ahead hourly prices (EUR/MWh) from ENTSO-E for [start, end)."""
    client = _get_client()
    start_ts = pd.Timestamp(start, tz="Europe/Amsterdam")
    end_ts = pd.Timestamp(end, tz="Europe/Amsterdam")

    logger.info("Fetching day-ahead prices for %s from %s to %s", COUNTRY_CODE, start, end)
    prices = client.query_day_ahead_prices(COUNTRY_CODE, start=start_ts, end=end_ts)
    logger.info("Fetched %d rows", len(prices))
    return prices


def fetch_afrr_capacity_prices(start: str, end: str) -> pd.DataFrame:
    """Fetch NL aFRR capacity prices (EUR/MW, Up/Down columns) from ENTSO-E, resampled to hourly.

    Native resolution is 15 minutes, but the price only changes at block
    boundaries (24h blocks before ~2025-07, 4h blocks after), so taking the
    first value per hour is lossless and keeps this aligned with the hourly
    day-ahead series.
    """
    client = _get_client()
    start_ts = pd.Timestamp(start, tz="Europe/Amsterdam")
    end_ts = pd.Timestamp(end, tz="Europe/Amsterdam")

    logger.info("Fetching aFRR capacity prices for %s from %s to %s", COUNTRY_CODE, start, end)
    prices = client.query_contracted_reserve_prices(
        COUNTRY_CODE,
        process_type=AFRR_PROCESS_TYPE,
        type_marketagreement_type=AFRR_MARKET_AGREEMENT,
        start=start_ts,
        end=end_ts,
    )
    prices = prices.resample("1h").first()
    logger.info("Fetched %d hourly rows", len(prices))
    return prices


if __name__ == "__main__":
    config = load_config()
    backtest_start = config["backtest"]["start"]
    backtest_end = config["backtest"]["end"]

    day_ahead = fetch_day_ahead_prices(backtest_start, backtest_end)
    print(day_ahead.head())
    print(day_ahead.tail())

    afrr = fetch_afrr_capacity_prices(backtest_start, backtest_end)
    print(afrr.head())
    print(afrr.tail())
