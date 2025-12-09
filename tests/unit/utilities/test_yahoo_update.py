"""Unit tests for Yahoo Finance data updater utility."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import yaml

from qtrader.utilities.yahoo_update import (
    BATCH_SIZE,
    MIN_REQUEST_INTERVAL,
    discover_symbols,
    fetch_yahoo_data,
    get_existing_date_range,
    get_safe_end_date,
    get_yahoo_data_dir,
    is_market_closed,
    load_data_sources_config,
    load_universe,
    merge_data,
    update_dividends_calendar,
    update_symbol,
)

# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory for testing."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_csv_data() -> str:
    """Sample CSV data for testing."""
    return """Date,Open,High,Low,Close,Adj Close,Volume
2020-01-02,74.06,75.15,73.80,75.09,72.47,135480400
2020-01-03,74.29,75.14,74.13,74.36,71.76,146322800
2020-01-06,73.45,74.99,73.19,74.95,72.33,118387200"""


@pytest.fixture
def sample_price_dataframe() -> pd.DataFrame:
    """Sample price DataFrame for testing."""
    data = {
        "Open": [74.06, 74.29, 73.45],
        "High": [75.15, 75.14, 74.99],
        "Low": [73.80, 74.13, 73.19],
        "Close": [75.09, 74.36, 74.95],
        "Adj Close": [72.47, 71.76, 72.33],
        "Volume": [135480400, 146322800, 118387200],
    }
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    return df


@pytest.fixture
def sample_dividends() -> pd.Series:
    """Sample dividends Series for testing."""
    dates = pd.to_datetime(["2020-02-07", "2020-05-08"])
    return pd.Series([0.77, 0.82], index=dates, name="Dividends")


@pytest.fixture
def sample_data_sources_config() -> Dict[str, Any]:
    """Sample data_sources.yaml configuration."""
    return {
        "data_sources": {
            "yahoo-us-equity-1d-csv": {
                "provider": "yahoo",
                "asset_class": "equity",
                "adapter": "yahoo_csv",
                "root_path": "data/us-equity-yahoo-csv",
            },
            "custom-dataset": {
                "provider": "custom",
                "asset_class": "equity",
                "adapter": "custom_csv",
                "root_path": "/absolute/path/to/data",
            },
        }
    }


# ==============================================================================
# MARKET TIMING TESTS
# ==============================================================================


class TestMarketTiming:
    """Tests for market timing functions."""

    def test_is_market_closed_after_4pm(self):
        """Test market is closed after 4 PM ET."""
        # Arrange: Mock time to 5 PM ET
        mock_time = datetime(2024, 11, 19, 17, 0, 0, tzinfo=ZoneInfo("America/New_York"))

        # Act & Assert
        with patch("qtrader.utilities.yahoo_update.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time
            assert is_market_closed() is True

    def test_is_market_closed_at_4pm_exactly(self):
        """Test market is closed exactly at 4:00 PM ET."""
        # Arrange: Mock time to exactly 4 PM ET
        mock_time = datetime(2024, 11, 19, 16, 0, 0, tzinfo=ZoneInfo("America/New_York"))

        # Act & Assert
        with patch("qtrader.utilities.yahoo_update.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time
            assert is_market_closed() is True

    def test_is_market_closed_before_4pm(self):
        """Test market is open before 4 PM ET."""
        # Arrange: Mock time to 3 PM ET
        mock_time = datetime(2024, 11, 19, 15, 0, 0, tzinfo=ZoneInfo("America/New_York"))

        # Act & Assert
        with patch("qtrader.utilities.yahoo_update.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time
            assert is_market_closed() is False

    def test_get_safe_end_date_market_closed(self):
        """Test safe end date includes today when market is closed."""
        # Arrange: Mock time to 5 PM ET (market closed)
        mock_time = datetime(2024, 11, 19, 17, 0, 0, tzinfo=ZoneInfo("America/New_York"))

        # Act
        with patch("qtrader.utilities.yahoo_update.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time
            result = get_safe_end_date()

        # Assert: Should return today's date
        assert result == "2024-11-19"

    def test_get_safe_end_date_market_open(self):
        """Test safe end date excludes today when market is still open."""
        # Arrange: Mock time to 2 PM ET (market open)
        mock_time = datetime(2024, 11, 19, 14, 0, 0, tzinfo=ZoneInfo("America/New_York"))

        # Act
        with patch("qtrader.utilities.yahoo_update.datetime") as mock_dt:
            mock_dt.now.return_value = mock_time
            result = get_safe_end_date()

        # Assert: Should return yesterday's date
        assert result == "2024-11-18"


# ==============================================================================
# FILE I/O TESTS
# ==============================================================================


class TestFileOperations:
    """Tests for file reading and writing operations."""

    def test_get_existing_date_range_valid_csv(self, temp_data_dir: Path, sample_csv_data: str):
        """Test reading date range from valid CSV file."""
        # Arrange
        csv_path = temp_data_dir / "AAPL.csv"
        csv_path.write_text(sample_csv_data)

        # Act
        result = get_existing_date_range(csv_path)

        # Assert
        assert result is not None
        assert result[0] == "2020-01-02"
        assert result[1] == "2020-01-06"

    def test_get_existing_date_range_missing_file(self, temp_data_dir: Path):
        """Test reading date range from non-existent file returns None."""
        # Arrange
        csv_path = temp_data_dir / "NONEXISTENT.csv"

        # Act
        result = get_existing_date_range(csv_path)

        # Assert
        assert result is None

    def test_get_existing_date_range_empty_csv(self, temp_data_dir: Path):
        """Test reading date range from empty CSV returns None."""
        # Arrange
        csv_path = temp_data_dir / "EMPTY.csv"
        csv_path.write_text("Date,Open,High,Low,Close,Adj Close,Volume\n")

        # Act
        result = get_existing_date_range(csv_path)

        # Assert
        assert result is None

    def test_discover_symbols_finds_csv_files(self, temp_data_dir: Path):
        """Test discovering symbols from CSV files in directory."""
        # Arrange
        (temp_data_dir / "AAPL.csv").write_text("test")
        (temp_data_dir / "MSFT.csv").write_text("test")
        (temp_data_dir / "GOOGL.csv").write_text("test")
        (temp_data_dir / "readme.txt").write_text("not a csv")

        # Act
        symbols = discover_symbols(temp_data_dir)

        # Assert
        assert sorted(symbols) == ["AAPL", "GOOGL", "MSFT"]

    def test_discover_symbols_empty_directory(self, temp_data_dir: Path):
        """Test discovering symbols in empty directory returns empty list."""
        # Arrange: Empty directory

        # Act
        symbols = discover_symbols(temp_data_dir)

        # Assert
        assert symbols == []

    def test_load_universe_valid_json(self, temp_data_dir: Path):
        """Test loading universe from valid JSON file."""
        # Arrange
        universe_path = temp_data_dir / "universe.json"
        universe_data = {"tickers": ["AAPL", "MSFT", "GOOGL"]}
        universe_path.write_text(json.dumps(universe_data))

        # Act
        tickers = load_universe(universe_path)

        # Assert
        assert tickers == ["AAPL", "MSFT", "GOOGL"]

    def test_load_universe_missing_file(self, temp_data_dir: Path):
        """Test loading universe from non-existent file returns None."""
        # Arrange
        universe_path = temp_data_dir / "missing_universe.json"

        # Act
        tickers = load_universe(universe_path)

        # Assert
        assert tickers is None

    def test_load_universe_empty_tickers(self, temp_data_dir: Path):
        """Test loading universe with empty tickers list."""
        # Arrange
        universe_path = temp_data_dir / "universe.json"
        universe_data: Dict[str, list] = {"tickers": []}
        universe_path.write_text(json.dumps(universe_data))

        # Act
        tickers = load_universe(universe_path)

        # Assert
        assert tickers == []

    def test_load_universe_missing_tickers_key(self, temp_data_dir: Path):
        """Test loading universe without tickers key returns empty list."""
        # Arrange
        universe_path = temp_data_dir / "universe.json"
        universe_data = {"other_key": "value"}
        universe_path.write_text(json.dumps(universe_data))

        # Act
        tickers = load_universe(universe_path)

        # Assert
        assert tickers == []


# ==============================================================================
# DATA FETCHING TESTS
# ==============================================================================


class TestDataFetching:
    """Tests for Yahoo Finance data fetching."""

    @patch("qtrader.utilities.yahoo_update.yf.Ticker")
    def test_fetch_yahoo_data_success(self, mock_ticker_class: Mock, sample_price_dataframe: pd.DataFrame):
        """Test successful data fetch from Yahoo Finance."""
        # Arrange
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker

        # Create mock history with dividends
        hist_df = sample_price_dataframe.copy()
        hist_df["Dividends"] = [0.0, 0.0, 0.0]
        mock_ticker.history.return_value = hist_df

        # Act
        price_data, dividends = fetch_yahoo_data("AAPL", "2020-01-01", "2020-01-10")

        # Assert
        assert price_data is not None
        assert len(price_data) == 3
        assert "Open" in price_data.columns
        assert "Adj Close" in price_data.columns
        mock_ticker.history.assert_called_once()

    @patch("qtrader.utilities.yahoo_update.yf.Ticker")
    def test_fetch_yahoo_data_empty_result(self, mock_ticker_class: Mock):
        """Test fetch returns None for empty results."""
        # Arrange
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.history.return_value = pd.DataFrame()

        # Act
        price_data, dividends = fetch_yahoo_data("INVALID", "2020-01-01", "2020-01-10")

        # Assert
        assert price_data is None
        assert dividends is None

    @patch("qtrader.utilities.yahoo_update.yf.Ticker")
    def test_fetch_yahoo_data_with_dividends(
        self, mock_ticker_class: Mock, sample_price_dataframe: pd.DataFrame, sample_dividends: pd.Series
    ):
        """Test fetching data with dividends."""
        # Arrange
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker

        hist_df = sample_price_dataframe.copy()
        hist_df["Dividends"] = [0.0, 0.0, 0.0]
        # Add some dividends
        hist_df.loc[hist_df.index[1], "Dividends"] = 0.82
        mock_ticker.history.return_value = hist_df

        # Act
        price_data, dividends = fetch_yahoo_data("AAPL")

        # Assert
        assert price_data is not None
        assert dividends is not None
        assert len(dividends) == 1  # Only non-zero dividends
        assert dividends.iloc[0] == 0.82

    @patch("qtrader.utilities.yahoo_update.yf.Ticker")
    @patch("qtrader.utilities.yahoo_update.datetime")
    def test_fetch_yahoo_data_filters_intraday_when_market_open(self, mock_datetime: Mock, mock_ticker_class: Mock):
        """Test that intraday data is filtered out when market is still open."""
        # Arrange: Mock time to 2 PM ET (market open)
        mock_now = datetime(2024, 12, 9, 14, 0, 0, tzinfo=ZoneInfo("America/New_York"))
        mock_datetime.now.return_value = mock_now

        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker

        # yfinance returns data including today (Dec 9) despite end_date parameter
        hist_df = pd.DataFrame(
            {
                "Open": [280.0, 278.0, 279.0],
                "High": [281.0, 279.0, 280.0],
                "Low": [279.0, 277.0, 278.0],
                "Close": [280.5, 278.5, 279.5],
                "Adj Close": [280.0, 278.0, 279.0],
                "Volume": [40000000, 42000000, 31779],  # Last row has incomplete volume
                "Dividends": [0.0, 0.0, 0.0],
            },
            index=pd.to_datetime(["2024-12-05", "2024-12-06", "2024-12-09"]),  # Includes today
        )
        mock_ticker.history.return_value = hist_df

        # Act
        price_data, dividends = fetch_yahoo_data("AAPL")

        # Assert: Should filter out Dec 9 (today) since market is open
        assert price_data is not None
        assert len(price_data) == 2  # Should only have Dec 5 and Dec 6
        dates = price_data.index.strftime("%Y-%m-%d").tolist()
        assert "2024-12-09" not in dates
        assert "2024-12-05" in dates
        assert "2024-12-06" in dates

    @patch("qtrader.utilities.yahoo_update.yf.Ticker")
    @patch("qtrader.utilities.yahoo_update.time.sleep")
    def test_fetch_yahoo_data_retry_on_rate_limit(self, mock_sleep: Mock, mock_ticker_class: Mock):
        """Test retry logic on rate limit error."""
        # Arrange
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker

        # First call raises rate limit error, second succeeds
        mock_ticker.history.side_effect = [
            Exception("429 Too many requests"),
            pd.DataFrame(
                {
                    "Open": [100],
                    "High": [105],
                    "Low": [99],
                    "Close": [103],
                    "Adj Close": [102],
                    "Volume": [1000],
                    "Dividends": [0],
                },
                index=pd.to_datetime(["2020-01-02"]),
            ),
        ]

        # Act
        price_data, dividends = fetch_yahoo_data("AAPL", retry_count=0)

        # Assert
        assert price_data is not None
        assert mock_ticker.history.call_count == 2
        mock_sleep.assert_called_once()  # Should sleep before retry

    @patch("qtrader.utilities.yahoo_update.yf.Ticker")
    def test_fetch_yahoo_data_max_retries_exceeded(self, mock_ticker_class: Mock):
        """Test max retries exceeded returns None."""
        # Arrange
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.history.side_effect = Exception("rate limit exceeded")

        # Act
        with patch("qtrader.utilities.yahoo_update.time.sleep"):
            price_data, dividends = fetch_yahoo_data("AAPL", retry_count=3)  # Already at max

        # Assert
        assert price_data is None
        assert dividends is None


# ==============================================================================
# DATA MERGING TESTS
# ==============================================================================


class TestDataMerging:
    """Tests for merging new data with existing CSV files."""

    def test_merge_data_create_new_file(self, temp_data_dir: Path, sample_price_dataframe: pd.DataFrame):
        """Test creating new CSV file when none exists."""
        # Arrange
        csv_path = temp_data_dir / "NEW.csv"
        assert not csv_path.exists()

        # Act
        result = merge_data(csv_path, sample_price_dataframe, full_refresh=False)

        # Assert
        assert result is True
        assert csv_path.exists()

        # Verify content
        df = pd.read_csv(csv_path)
        assert len(df) == 3
        assert list(df.columns) == ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]

    def test_merge_data_full_refresh(
        self, temp_data_dir: Path, sample_csv_data: str, sample_price_dataframe: pd.DataFrame
    ):
        """Test full refresh replaces entire file."""
        # Arrange
        csv_path = temp_data_dir / "AAPL.csv"
        csv_path.write_text(sample_csv_data)

        # New data with different dates - keep Date as index with proper name
        new_data = sample_price_dataframe.copy().iloc[[0]]
        new_data.index = pd.DatetimeIndex(["2020-02-01"], name="Date")

        # Act
        result = merge_data(csv_path, new_data, full_refresh=True)

        # Assert
        assert result is True
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert df.iloc[0]["Date"] == "2020-02-01"

    def test_merge_data_incremental_append(self, temp_data_dir: Path, sample_csv_data: str):
        """Test incremental append of new data."""
        # Arrange
        csv_path = temp_data_dir / "AAPL.csv"
        csv_path.write_text(sample_csv_data)

        # New data after existing dates - Date as index like fetch_yahoo_data returns
        new_data = pd.DataFrame(
            {
                "Open": [75.00],
                "High": [76.00],
                "Low": [74.50],
                "Close": [75.50],
                "Adj Close": [73.00],
                "Volume": [100000],
            },
            index=pd.DatetimeIndex(["2020-01-07"], name="Date"),
        )

        # Act
        result = merge_data(csv_path, new_data, full_refresh=False)

        # Assert
        assert result is True
        df = pd.read_csv(csv_path)
        assert len(df) == 4  # Original 3 + 1 new
        assert df["Date"].iloc[-1] == "2020-01-07"

    def test_merge_data_no_new_data(self, temp_data_dir: Path, sample_csv_data: str):
        """Test no update when all data already exists."""
        # Arrange
        csv_path = temp_data_dir / "AAPL.csv"
        csv_path.write_text(sample_csv_data)

        # Same data as already exists
        old_data = pd.DataFrame(
            {
                "Open": [74.06],
                "High": [75.15],
                "Low": [73.80],
                "Close": [75.09],
                "Adj Close": [72.47],
                "Volume": [135480400],
            },
            index=pd.to_datetime(["2020-01-02"]),
        )
        old_data.index.name = "Date"

        # Act
        result = merge_data(csv_path, old_data, full_refresh=False)

        # Assert
        assert result is False  # No new data added
        df = pd.read_csv(csv_path)
        assert len(df) == 3  # Original count unchanged

    @patch("qtrader.utilities.yahoo_update.datetime")
    def test_merge_data_filters_future_dates_when_market_open(self, mock_datetime: Mock, temp_data_dir: Path):
        """Test that merge_data filters out dates beyond safe_end_date."""
        # Arrange: Mock time to 2 PM ET on Dec 9 (market open)
        mock_now = datetime(2024, 12, 9, 14, 0, 0, tzinfo=ZoneInfo("America/New_York"))
        mock_datetime.now.return_value = mock_now

        csv_path = temp_data_dir / "AAPL.csv"

        # New data includes today (Dec 9) which should be filtered
        new_data = pd.DataFrame(
            {
                "Open": [280.0, 278.0, 279.0],
                "High": [281.0, 279.0, 280.0],
                "Low": [279.0, 277.0, 278.0],
                "Close": [280.5, 278.5, 279.5],
                "Adj Close": [280.0, 278.0, 279.0],
                "Volume": [40000000, 42000000, 31779],  # Last has incomplete volume
            },
            index=pd.to_datetime(["2024-12-05", "2024-12-06", "2024-12-09"]),
        )
        new_data.index.name = "Date"

        # Act
        result = merge_data(csv_path, new_data, full_refresh=False)

        # Assert
        assert result is True
        df = pd.read_csv(csv_path)
        assert len(df) == 2  # Should only have Dec 5 and Dec 6, not Dec 9
        assert "2024-12-09" not in df["Date"].values
        assert "2024-12-05" in df["Date"].values
        assert "2024-12-06" in df["Date"].values

    def test_merge_data_none_dataframe(self, temp_data_dir: Path):
        """Test merge with None DataFrame returns False."""
        # Arrange
        csv_path = temp_data_dir / "AAPL.csv"

        # Act
        result = merge_data(csv_path, None, full_refresh=False)

        # Assert
        assert result is False

    def test_merge_data_empty_dataframe(self, temp_data_dir: Path):
        """Test merge with empty DataFrame returns False."""
        # Arrange
        csv_path = temp_data_dir / "AAPL.csv"
        empty_df = pd.DataFrame()

        # Act
        result = merge_data(csv_path, empty_df, full_refresh=False)

        # Assert
        assert result is False


# ==============================================================================
# DIVIDENDS CALENDAR TESTS
# ==============================================================================


class TestDividendsCalendar:
    """Tests for dividends calendar management."""

    def test_update_dividends_calendar_new_file(self, temp_data_dir: Path, sample_dividends: pd.Series):
        """Test creating new dividends calendar file."""
        # Arrange
        dividends_path = temp_data_dir / "dividends_calendar.json"
        assert not dividends_path.exists()

        # Act
        update_dividends_calendar(dividends_path, "AAPL", sample_dividends)

        # Assert
        assert dividends_path.exists()
        with dividends_path.open("r") as f:
            calendar = json.load(f)
        assert "AAPL" in calendar
        assert len(calendar["AAPL"]) == 2
        assert calendar["AAPL"][0]["amount"] == 0.77

    def test_update_dividends_calendar_add_to_existing(self, temp_data_dir: Path, sample_dividends: pd.Series):
        """Test adding dividends to existing calendar."""
        # Arrange
        dividends_path = temp_data_dir / "dividends_calendar.json"
        existing_calendar = {"MSFT": [{"date": "2020-01-15", "amount": 0.50}]}
        dividends_path.write_text(json.dumps(existing_calendar))

        # Act
        update_dividends_calendar(dividends_path, "AAPL", sample_dividends)

        # Assert
        with dividends_path.open("r") as f:
            calendar = json.load(f)
        assert "AAPL" in calendar
        assert "MSFT" in calendar
        assert len(calendar["AAPL"]) == 2
        assert len(calendar["MSFT"]) == 1

    def test_update_dividends_calendar_skip_duplicates(self, temp_data_dir: Path):
        """Test that duplicate dividends are not added."""
        # Arrange
        dividends_path = temp_data_dir / "dividends_calendar.json"
        existing_calendar = {"AAPL": [{"date": "2020-02-07", "amount": 0.77}]}
        dividends_path.write_text(json.dumps(existing_calendar))

        # Same dividend
        dividends = pd.Series([0.77], index=pd.to_datetime(["2020-02-07"]))

        # Act
        update_dividends_calendar(dividends_path, "AAPL", dividends)

        # Assert
        with dividends_path.open("r") as f:
            calendar = json.load(f)
        assert len(calendar["AAPL"]) == 1  # No duplicate

    def test_update_dividends_calendar_none_dividends(self, temp_data_dir: Path):
        """Test update with None dividends does nothing."""
        # Arrange
        dividends_path = temp_data_dir / "dividends_calendar.json"

        # Act
        update_dividends_calendar(dividends_path, "AAPL", None)

        # Assert
        assert not dividends_path.exists()

    def test_update_dividends_calendar_empty_dividends(self, temp_data_dir: Path):
        """Test update with empty dividends Series does nothing."""
        # Arrange
        dividends_path = temp_data_dir / "dividends_calendar.json"
        empty_dividends = pd.Series([], dtype=float)

        # Act
        update_dividends_calendar(dividends_path, "AAPL", empty_dividends)

        # Assert
        assert not dividends_path.exists()

    def test_update_dividends_calendar_sorts_by_date(self, temp_data_dir: Path):
        """Test dividends are sorted by date."""
        # Arrange
        dividends_path = temp_data_dir / "dividends_calendar.json"

        # Add dividends in reverse order
        dividends1 = pd.Series([0.82], index=pd.to_datetime(["2020-05-08"]))
        dividends2 = pd.Series([0.77], index=pd.to_datetime(["2020-02-07"]))

        # Act
        update_dividends_calendar(dividends_path, "AAPL", dividends1)
        update_dividends_calendar(dividends_path, "AAPL", dividends2)

        # Assert
        with dividends_path.open("r") as f:
            calendar = json.load(f)
        dates = [d["date"] for d in calendar["AAPL"]]
        assert dates == ["2020-02-07", "2020-05-08"]  # Sorted


# ==============================================================================
# CONFIG LOADING TESTS
# ==============================================================================


class TestConfigLoading:
    """Tests for data_sources.yaml configuration loading."""

    def test_load_data_sources_config_project_path(self, temp_data_dir: Path):
        """Test loading config from project directory."""
        # Arrange - create a temporary config file in project-like structure
        project_dir = temp_data_dir / "project"
        config_dir = project_dir / "config"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "data_sources.yaml"

        test_config = {"data_sources": {"test": {"root_path": "test/path"}}}
        with config_file.open("w") as f:
            yaml.dump(test_config, f)

        # Act - mock the __file__ location to point to our temp project
        module_file = project_dir / "src" / "qtrader" / "utilities" / "yahoo_update.py"
        module_file.parent.mkdir(parents=True)
        module_file.touch()

        with patch("qtrader.utilities.yahoo_update.__file__", str(module_file)):
            result = load_data_sources_config()

        # Assert
        assert result == test_config

    def test_load_data_sources_config_user_home(self, temp_data_dir: Path):
        """Test loading config from user home directory."""
        # Arrange - create config in temp home directory
        home_dir = temp_data_dir / "home"
        qtrader_dir = home_dir / ".qtrader"
        qtrader_dir.mkdir(parents=True)
        config_file = qtrader_dir / "data_sources.yaml"

        test_config = {"data_sources": {"test": {"root_path": "home/path"}}}
        with config_file.open("w") as f:
            yaml.dump(test_config, f)

        # Act - mock home directory and make project config not exist
        with patch("qtrader.utilities.yahoo_update.__file__", "/nonexistent/project/yahoo_update.py"):
            with patch("pathlib.Path.home", return_value=home_dir):
                result = load_data_sources_config()

        # Assert
        assert result == test_config

    def test_load_data_sources_config_no_file(self):
        """Test loading config when no file exists returns None."""
        # Act
        with patch("pathlib.Path.exists", return_value=False):
            result = load_data_sources_config()

        # Assert
        assert result is None

    def test_get_yahoo_data_dir_valid_source(self, sample_data_sources_config: Dict[str, Any]):
        """Test extracting data directory from valid config."""
        # Arrange
        with patch("qtrader.utilities.yahoo_update.load_data_sources_config", return_value=sample_data_sources_config):
            # Act
            result = get_yahoo_data_dir("yahoo-us-equity-1d-csv")

        # Assert
        assert result is not None
        assert "us-equity-yahoo-csv" in str(result)

    def test_get_yahoo_data_dir_absolute_path(self, sample_data_sources_config: Dict[str, Any]):
        """Test handling absolute path in config."""
        # Arrange
        with patch("qtrader.utilities.yahoo_update.load_data_sources_config", return_value=sample_data_sources_config):
            # Act
            result = get_yahoo_data_dir("custom-dataset")

        # Assert
        assert result is not None
        assert result == Path("/absolute/path/to/data")

    def test_get_yahoo_data_dir_missing_source(self, sample_data_sources_config: Dict[str, Any]):
        """Test getting data directory for non-existent source returns None."""
        # Arrange
        with patch("qtrader.utilities.yahoo_update.load_data_sources_config", return_value=sample_data_sources_config):
            # Act
            result = get_yahoo_data_dir("nonexistent-source")

        # Assert
        assert result is None

    def test_get_yahoo_data_dir_no_config(self):
        """Test getting data directory when config loading fails returns None."""
        # Arrange
        with patch("qtrader.utilities.yahoo_update.load_data_sources_config", return_value=None):
            # Act
            result = get_yahoo_data_dir("yahoo-us-equity-1d-csv")

        # Assert
        assert result is None


# ==============================================================================
# SYMBOL UPDATE INTEGRATION TESTS
# ==============================================================================


class TestSymbolUpdate:
    """Tests for complete symbol update workflow."""

    @patch("qtrader.utilities.yahoo_update.fetch_yahoo_data")
    @patch("qtrader.utilities.yahoo_update.time.sleep")
    def test_update_symbol_success(
        self, mock_sleep: Mock, mock_fetch: Mock, temp_data_dir: Path, sample_price_dataframe: pd.DataFrame
    ):
        """Test successful symbol update."""
        # Arrange
        dividends_path = temp_data_dir / "dividends_calendar.json"
        mock_fetch.return_value = (sample_price_dataframe, None)

        # Act
        result = update_symbol("AAPL", temp_data_dir, dividends_path, full_refresh=True, rate_limit=False)

        # Assert
        assert result["symbol"] == "AAPL"
        assert result["success"] is True
        assert result["action"] == "full_refresh"
        assert result["bars_added"] == 3

    @patch("qtrader.utilities.yahoo_update.fetch_yahoo_data")
    def test_update_symbol_incremental(
        self, mock_fetch: Mock, temp_data_dir: Path, sample_csv_data: str, sample_price_dataframe: pd.DataFrame
    ):
        """Test incremental symbol update."""
        # Arrange
        csv_path = temp_data_dir / "AAPL.csv"
        csv_path.write_text(sample_csv_data)
        dividends_path = temp_data_dir / "dividends_calendar.json"

        # New data after existing - use just one row with proper index
        new_data = sample_price_dataframe.copy().iloc[[0]]
        new_data.index = pd.DatetimeIndex(["2020-01-07"], name="Date")
        mock_fetch.return_value = (new_data, None)

        # Act
        result = update_symbol("AAPL", temp_data_dir, dividends_path, rate_limit=False)

        # Assert
        assert result["success"] is True
        assert result["action"] == "incremental"
        assert result["bars_added"] == 1

    @patch("qtrader.utilities.yahoo_update.fetch_yahoo_data")
    def test_update_symbol_already_up_to_date(self, mock_fetch: Mock, temp_data_dir: Path, sample_csv_data: str):
        """Test update when data is already up to date."""
        # Arrange
        csv_path = temp_data_dir / "AAPL.csv"
        csv_path.write_text(sample_csv_data)
        dividends_path = temp_data_dir / "dividends_calendar.json"

        # Mock safe end date to be before last CSV date
        with patch("qtrader.utilities.yahoo_update.get_safe_end_date", return_value="2020-01-05"):
            # Act
            result = update_symbol("AAPL", temp_data_dir, dividends_path, rate_limit=False)

        # Assert
        assert result["success"] is True
        assert result["action"] == "none"
        assert result["bars_added"] == 0
        mock_fetch.assert_not_called()

    @patch("qtrader.utilities.yahoo_update.fetch_yahoo_data")
    def test_update_symbol_fetch_failure(self, mock_fetch: Mock, temp_data_dir: Path):
        """Test symbol update when fetch fails."""
        # Arrange
        dividends_path = temp_data_dir / "dividends_calendar.json"
        mock_fetch.return_value = (None, None)

        # Act
        result = update_symbol("INVALID", temp_data_dir, dividends_path, rate_limit=False)

        # Assert
        assert result["success"] is False
        assert result["bars_added"] == 0

    @patch("qtrader.utilities.yahoo_update.fetch_yahoo_data")
    @patch("qtrader.utilities.yahoo_update.time.sleep")
    @patch("qtrader.utilities.yahoo_update.time.time")
    def test_update_symbol_with_rate_limiting(
        self,
        mock_time: Mock,
        mock_sleep: Mock,
        mock_fetch: Mock,
        temp_data_dir: Path,
        sample_price_dataframe: pd.DataFrame,
    ):
        """Test symbol update respects rate limiting."""
        # Arrange
        dividends_path = temp_data_dir / "dividends_calendar.json"
        mock_fetch.return_value = (sample_price_dataframe, None)

        # Mock timing: simulate that last request was recent
        # First call to time.time() returns 5.0, second returns 5.0 (same time, triggers sleep)
        mock_time.side_effect = [5.0, 5.0]

        # Import and patch the global _last_request_time to simulate recent request
        import qtrader.utilities.yahoo_update as yupdate

        original_last_time = yupdate._last_request_time
        yupdate._last_request_time = 3.0  # Set to 2 seconds ago

        try:
            # Act
            result = update_symbol("AAPL", temp_data_dir, dividends_path, rate_limit=True)

            # Assert
            assert result["success"] is True
            # Should sleep since only 2 seconds elapsed (less than MIN_REQUEST_INTERVAL of 3.0)
            assert mock_sleep.call_count >= 1, (
                f"Expected sleep to be called at least once, but call_count was {mock_sleep.call_count}"
            )
        finally:
            # Restore original value
            yupdate._last_request_time = original_last_time


# ==============================================================================
# RATE LIMITING TESTS
# ==============================================================================


class TestRateLimiting:
    """Tests for rate limiting configuration and behavior."""

    def test_rate_limiting_constants_defined(self):
        """Test that rate limiting constants are properly defined."""
        # Assert
        assert MIN_REQUEST_INTERVAL > 0
        assert BATCH_SIZE > 0
        assert MIN_REQUEST_INTERVAL == 60.0 / 20  # 20 requests per minute

    def test_batch_size_appropriate(self):
        """Test batch size is reasonable for API limits."""
        # Assert
        assert BATCH_SIZE <= 100  # Should not be too large
        assert BATCH_SIZE >= 10  # Should not be too small
