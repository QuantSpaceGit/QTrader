"""Unit tests for reporting writers module.

Tests cover:
- JSON report writing with Decimal handling
- JSON time-series writers (equity, returns, trades, drawdowns)
- CSV timeline writer with tall format
"""

import json
from decimal import Decimal
from pathlib import Path

import pytest

from qtrader.events.event_store import InMemoryEventStore
from qtrader.events.events import (
    FillEvent,
    IndicatorEvent,
    OrderEvent,
    PerformanceMetricsEvent,
    PriceBarEvent,
    SignalEvent,
)
from qtrader.libraries.performance.models import DrawdownPeriod, EquityCurvePoint, FullMetrics, ReturnPoint, TradeRecord
from qtrader.services.reporting.writers import (
    DecimalEncoder,
    write_drawdowns_json,
    write_equity_curve_json,
    write_json_report,
    write_returns_json,
    write_strategy_chart_data,
    write_trades_json,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Create temporary output directory for test files."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def sample_metrics() -> FullMetrics:
    """Sample metrics for testing - using model_construct to bypass validation."""
    return FullMetrics.model_construct(
        backtest_id="test_001",
        start_date="2023-01-01",
        end_date="2023-12-31",
        duration_days=365,
        initial_equity=Decimal("100000"),
        final_equity=Decimal("125000"),
        total_return_pct=Decimal("0.25"),
        cagr=Decimal("0.15"),
        best_day_return_pct=Decimal("0.05"),
        worst_day_return_pct=Decimal("-0.03"),
        volatility_annual_pct=Decimal("0.18"),
        max_drawdown_pct=Decimal("0.10"),
        sharpe_ratio=Decimal("1.5"),
        sortino_ratio=Decimal("1.8"),
        calmar_ratio=Decimal("1.2"),
        total_trades=100,
        winning_trades=60,
        losing_trades=40,
        win_rate=Decimal("0.60"),
        profit_factor=Decimal("2.0"),
        avg_win=Decimal("500"),
        avg_loss=Decimal("250"),
        largest_win=Decimal("2000"),
        largest_loss=Decimal("800"),
        expectancy=Decimal("150"),
        consecutive_wins=5,
        consecutive_losses=3,
        total_commissions=Decimal("1200"),
        commission_pct_of_pnl=Decimal("0.048"),
        monthly_returns=[],
        quarterly_returns=[],
    )


@pytest.fixture
def sample_equity_curve() -> list[EquityCurvePoint]:
    """Sample equity curve for testing."""
    from datetime import datetime, timezone

    return [
        EquityCurvePoint.model_construct(
            timestamp=datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            equity=Decimal("100000.0"),
            cash=Decimal("100000.0"),
            positions_value=Decimal("0.0"),
            num_positions=0,
            gross_exposure=Decimal("0.0"),
            net_exposure=Decimal("0.0"),
            leverage=Decimal("0.0"),
            total_return_pct=Decimal("0.0"),
            max_drawdown_pct=Decimal("0.0"),
            drawdown_pct=Decimal("0.0"),
            underwater=False,
        ),
        EquityCurvePoint.model_construct(
            timestamp=datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
            equity=Decimal("101500.0"),
            cash=Decimal("86500.0"),
            positions_value=Decimal("15000.0"),
            num_positions=1,
            gross_exposure=Decimal("15000.0"),
            net_exposure=Decimal("15000.0"),
            leverage=Decimal("0.148"),
            total_return_pct=Decimal("0.015"),
            max_drawdown_pct=Decimal("0.0"),
            drawdown_pct=Decimal("0.0"),
            underwater=False,
        ),
    ]


@pytest.fixture
def sample_returns() -> list[ReturnPoint]:
    """Sample returns for testing."""
    from datetime import datetime, timezone

    return [
        ReturnPoint.model_construct(
            timestamp=datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            period_return=Decimal("0.0"),
            cumulative_return=Decimal("0.0"),
            log_return=Decimal("0.0"),
        ),
        ReturnPoint.model_construct(
            timestamp=datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
            period_return=Decimal("0.015"),
            cumulative_return=Decimal("0.015"),
            log_return=Decimal("0.0149"),
        ),
    ]


@pytest.fixture
def sample_trades() -> list[TradeRecord]:
    """Create sample trade records."""
    from datetime import datetime, timezone

    return [
        TradeRecord.model_construct(
            trade_id="trade_001",
            strategy_id="test_strategy",
            symbol="AAPL",
            entry_timestamp=datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            exit_timestamp=datetime(2023, 1, 5, 15, 0, 0, tzinfo=timezone.utc),
            entry_price=Decimal("150.00"),
            exit_price=Decimal("155.00"),
            quantity=100,
            side="long",
            pnl=Decimal("500.0"),
            pnl_pct=Decimal("0.0333"),
            commission=Decimal("2.0"),
            duration_seconds=345600,
        ),
        TradeRecord.model_construct(
            trade_id="trade_002",
            strategy_id="test_strategy",
            symbol="MSFT",
            entry_timestamp=datetime(2023, 1, 3, 11, 0, 0, tzinfo=timezone.utc),
            exit_timestamp=datetime(2023, 1, 4, 14, 0, 0, tzinfo=timezone.utc),
            entry_price=Decimal("250.00"),
            exit_price=Decimal("245.00"),
            quantity=50,
            side="long",
            pnl=Decimal("-250.0"),
            pnl_pct=Decimal("-0.02"),
            commission=Decimal("1.5"),
            duration_seconds=97200,
        ),
    ]


@pytest.fixture
def sample_drawdowns() -> list[DrawdownPeriod]:
    """Create sample drawdown periods."""
    from datetime import datetime, timezone

    return [
        DrawdownPeriod.model_construct(
            drawdown_id=1,
            start_timestamp=datetime(2023, 1, 10, 0, 0, 0, tzinfo=timezone.utc),
            trough_timestamp=datetime(2023, 1, 15, 0, 0, 0, tzinfo=timezone.utc),
            end_timestamp=datetime(2023, 1, 20, 0, 0, 0, tzinfo=timezone.utc),
            peak_equity=Decimal("105000.0"),
            trough_equity=Decimal("95000.0"),
            depth_pct=Decimal("-0.0952"),
            duration_days=5,
            recovery_days=5,
            recovered=True,
        ),
        DrawdownPeriod.model_construct(
            drawdown_id=2,
            start_timestamp=datetime(2023, 2, 1, 0, 0, 0, tzinfo=timezone.utc),
            trough_timestamp=datetime(2023, 2, 10, 0, 0, 0, tzinfo=timezone.utc),
            end_timestamp=None,
            peak_equity=Decimal("110000.0"),
            trough_equity=Decimal("100000.0"),
            depth_pct=Decimal("-0.0909"),
            duration_days=9,
            recovery_days=None,
            recovered=False,
        ),
    ]


@pytest.fixture
def sample_event_store() -> InMemoryEventStore:
    """Create event store with sample events for CSV timeline testing."""
    store = InMemoryEventStore()

    # Add bar events
    bar1 = PriceBarEvent.model_construct(
        symbol="AAPL",
        timestamp="2023-01-01T10:00:00+00:00",
        open=Decimal("150.00"),
        high=Decimal("152.00"),
        low=Decimal("149.00"),
        close=Decimal("151.00"),
        volume=1000000,
        source="test",
    )
    bar2 = PriceBarEvent.model_construct(
        symbol="AAPL",
        timestamp="2023-01-02T10:00:00+00:00",
        open=Decimal("151.00"),
        high=Decimal("153.00"),
        low=Decimal("150.50"),
        close=Decimal("152.50"),
        volume=1100000,
        source="test",
    )
    store.append(bar1)
    store.append(bar2)

    # Add indicator events
    indicator1 = IndicatorEvent(
        strategy_id="test_strategy",
        symbol="AAPL",
        timestamp="2023-01-01T10:00:00+00:00",
        indicators={"sma_10": 150.5, "sma_20": 149.8, "crossover": True},
    )
    indicator2 = IndicatorEvent(
        strategy_id="test_strategy",
        symbol="AAPL",
        timestamp="2023-01-02T10:00:00+00:00",
        indicators={"sma_10": 151.2, "sma_20": 150.1},
    )
    store.append(indicator1)
    store.append(indicator2)

    # Add signal event
    signal = SignalEvent.model_construct(
        strategy_id="test_strategy",
        symbol="AAPL",
        timestamp="2023-01-01T10:00:00+00:00",
        intention="OPEN_LONG",
        price=Decimal("151.00"),
        confidence=Decimal("1.0"),
        signal_id="signal_001",
    )
    store.append(signal)

    # Add order event
    order = OrderEvent.model_construct(
        symbol="AAPL",
        timestamp="2023-01-01T10:00:00+00:00",
        quantity=100,
        side="BUY",
        order_type="MARKET",
        intent_id="intent_001",
        idempotency_key="idem_001",
    )
    store.append(order)

    # Add fill event
    fill = FillEvent.model_construct(
        symbol="AAPL",
        timestamp="2023-01-01T10:00:00+00:00",
        filled_quantity=100,
        fill_price=Decimal("151.10"),
        commission=Decimal("1.50"),
        fill_id="fill_001",
        source_order_id="order_001",
        side="BUY",
    )
    store.append(fill)

    # Add performance metrics
    perf1 = PerformanceMetricsEvent.model_construct(
        timestamp="2023-01-01T10:00:00+00:00",
        equity=Decimal("100000.0"),
        cash=Decimal("85000.0"),
        positions_value=Decimal("15000.0"),
        total_return_pct=Decimal("0.0"),
        max_drawdown_pct=Decimal("0.0"),
        current_drawdown_pct=Decimal("0.0"),
        num_positions=1,
        gross_exposure=Decimal("15000.0"),
        net_exposure=Decimal("15000.0"),
        leverage=Decimal("0.15"),
    )
    perf2 = PerformanceMetricsEvent.model_construct(
        timestamp="2023-01-02T10:00:00+00:00",
        equity=Decimal("101500.0"),
        cash=Decimal("86500.0"),
        positions_value=Decimal("15000.0"),
        total_return_pct=Decimal("0.015"),
        max_drawdown_pct=Decimal("0.0"),
        current_drawdown_pct=Decimal("0.0"),
        num_positions=1,
        gross_exposure=Decimal("15000.0"),
        net_exposure=Decimal("15000.0"),
        leverage=Decimal("0.148"),
    )
    store.append(perf1)
    store.append(perf2)

    return store


# ============================================================================
# DecimalEncoder Tests
# ============================================================================


class TestDecimalEncoder:
    """Test custom JSON encoder for Decimal types."""

    def test_encode_decimal_as_float(self):
        """Decimal values should be encoded as floats in JSON."""
        # Arrange
        data = {"value": Decimal("123.45")}

        # Act
        result = json.dumps(data, cls=DecimalEncoder)

        # Assert
        assert result == '{"value": 123.45}'

    def test_encode_nested_decimals(self):
        """Nested Decimal values should be encoded correctly."""
        # Arrange
        data = {
            "metrics": {
                "sharpe": Decimal("1.5"),
                "sortino": Decimal("2.0"),
            }
        }

        # Act
        result = json.dumps(data, cls=DecimalEncoder)

        # Assert
        parsed = json.loads(result)
        assert parsed["metrics"]["sharpe"] == 1.5
        assert parsed["metrics"]["sortino"] == 2.0

    def test_encode_standard_types_unchanged(self):
        """Standard types should pass through unchanged."""
        # Arrange
        data = {"string": "test", "int": 42, "float": 3.14, "bool": True, "null": None}

        # Act
        result = json.dumps(data, cls=DecimalEncoder)

        # Assert
        parsed = json.loads(result)
        assert parsed == data


# ============================================================================
# JSON Report Writer Tests
# ============================================================================


class TestWriteJsonReport:
    """Test JSON report writer."""

    def test_write_json_report_creates_file(self, sample_metrics: FullMetrics, tmp_output_dir: Path):
        """write_json_report should create JSON file with metrics."""
        # Arrange
        output_path = tmp_output_dir / "performance.json"

        # Act
        write_json_report(sample_metrics, output_path)

        # Assert
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_write_json_report_content_valid(self, sample_metrics: FullMetrics, tmp_output_dir: Path):
        """JSON content should be valid and contain expected metrics."""
        # Arrange
        output_path = tmp_output_dir / "performance.json"

        # Act
        write_json_report(sample_metrics, output_path)

        # Assert
        with output_path.open("r") as f:
            data = json.load(f)

        # Values may be strings or floats depending on Decimal encoder
        assert float(data["total_return_pct"]) == pytest.approx(0.25)
        assert float(data["cagr"]) == pytest.approx(0.15)
        assert float(data["sharpe_ratio"]) == pytest.approx(1.5)
        assert data["total_trades"] == 100

    def test_write_json_report_creates_parent_dirs(self, sample_metrics: FullMetrics, tmp_output_dir: Path):
        """write_json_report should create parent directories if missing."""
        # Arrange
        output_path = tmp_output_dir / "nested" / "dir" / "performance.json"

        # Act
        write_json_report(sample_metrics, output_path)

        # Assert
        assert output_path.exists()

    def test_write_json_report_handles_all_decimals(self, tmp_output_dir: Path):
        """JSON writer should handle all Decimal fields correctly."""
        # Arrange
        metrics = FullMetrics.model_construct(
            backtest_id="test_decimals",
            start_date="2023-01-01",
            end_date="2023-12-31",
            duration_days=365,
            initial_equity=Decimal("100000"),
            final_equity=Decimal("123456.789"),
            total_return_pct=Decimal("0.123456789"),
            cagr=Decimal("0.987654321"),
            best_day_return_pct=Decimal("0.05"),
            worst_day_return_pct=Decimal("-0.03"),
            volatility_annual_pct=Decimal("0.18"),
            max_drawdown_pct=Decimal("-0.15432"),
            max_drawdown_duration_days=30,
            avg_drawdown_pct=Decimal("-0.05"),
            current_drawdown_pct=Decimal("0.0"),
            sharpe_ratio=Decimal("1.23456"),
            sortino_ratio=Decimal("2.34567"),
            calmar_ratio=Decimal("1.5"),
            risk_free_rate=Decimal("0.02"),
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=Decimal("0.6789"),
            profit_factor=Decimal("1.8765"),
            avg_win=Decimal("0.0"),
            avg_win_pct=Decimal("0.0"),
            avg_loss=Decimal("0.0"),
            avg_loss_pct=Decimal("0.0"),
            largest_win=Decimal("0.0"),
            largest_win_pct=Decimal("0.0"),
            largest_loss=Decimal("0.0"),
            largest_loss_pct=Decimal("0.0"),
            expectancy=Decimal("0.0"),
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            total_commissions=Decimal("0.0"),
            commission_pct_of_pnl=Decimal("0.0"),
            monthly_returns=[],
            quarterly_returns=[],
        )
        output_path = tmp_output_dir / "performance.json"

        # Act
        write_json_report(metrics, output_path)

        # Assert
        with output_path.open("r") as f:
            data = json.load(f)

        # Decimal encoder outputs strings for precision preservation
        assert isinstance(data["total_return_pct"], str)
        assert float(data["total_return_pct"]) == pytest.approx(0.123456789)
        assert float(data["cagr"]) == pytest.approx(0.987654321)


# ============================================================================
# Equity Curve Writer Tests
# ============================================================================


class TestWriteEquityCurveJson:
    """Test equity curve JSON writer."""

    def test_write_equity_curve_creates_file(self, sample_equity_curve: list[EquityCurvePoint], tmp_output_dir: Path):
        """write_equity_curve_json should create JSON file."""
        # Arrange
        output_path = tmp_output_dir / "equity_curve.json"

        # Act
        write_equity_curve_json(sample_equity_curve, output_path)

        # Assert
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_write_equity_curve_content_valid(self, sample_equity_curve: list[EquityCurvePoint], tmp_output_dir: Path):
        """JSON content should match input data."""
        # Arrange
        output_path = tmp_output_dir / "equity_curve.json"

        # Act
        write_equity_curve_json(sample_equity_curve, output_path)

        # Assert
        with open(output_path) as f:
            data = json.load(f)
        assert len(data) == 2
        assert "timestamp" in data[0]
        assert "equity" in data[0]
        assert data[0]["equity"] == 100000.0
        assert data[1]["equity"] == 101500.0  # Match fixture data

    def test_write_equity_curve_empty_list_logs_warning(self, tmp_output_dir: Path, caplog):
        """Empty equity curve should log warning and not create file."""
        # Arrange
        output_path = tmp_output_dir / "equity_curve.json"

        # Act
        write_equity_curve_json([], output_path)

        # Assert
        assert not output_path.exists()
        assert "No equity curve data to write" in caplog.text

    def test_write_equity_curve_creates_parent_dirs(
        self, sample_equity_curve: list[EquityCurvePoint], tmp_output_dir: Path
    ):
        """Writer should create parent directories if missing."""
        # Arrange
        output_path = tmp_output_dir / "nested" / "equity_curve.json"

        # Act
        write_equity_curve_json(sample_equity_curve, output_path)

        # Assert
        assert output_path.exists()


# ============================================================================
# Returns Writer Tests
# ============================================================================


class TestWriteReturnsJson:
    """Test returns JSON writer."""

    def test_write_returns_creates_file(self, sample_returns: list[ReturnPoint], tmp_output_dir: Path):
        """write_returns_json should create JSON file."""
        # Arrange
        output_path = tmp_output_dir / "returns.json"

        # Act
        write_returns_json(sample_returns, output_path)

        # Assert
        assert output_path.exists()

    def test_write_returns_content_valid(self, sample_returns: list[ReturnPoint], tmp_output_dir: Path):
        """JSON content should match input data."""
        # Arrange
        output_path = tmp_output_dir / "returns.json"

        # Act
        write_returns_json(sample_returns, output_path)

        # Assert
        with open(output_path) as f:
            data = json.load(f)
        assert len(data) == 2
        assert "timestamp" in data[0]
        assert "period_return" in data[0]
        assert data[0]["period_return"] == 0.0
        assert data[1]["period_return"] == pytest.approx(0.015)  # Match fixture data

    def test_write_returns_empty_list_logs_warning(self, tmp_output_dir: Path, caplog):
        """Empty returns should log warning and not create file."""
        # Arrange
        output_path = tmp_output_dir / "returns.json"

        # Act
        write_returns_json([], output_path)

        # Assert
        assert not output_path.exists()
        assert "No returns data to write" in caplog.text


# ============================================================================
# Trades Writer Tests
# ============================================================================


class TestWriteTradesJson:
    """Test trades JSON writer."""

    def test_write_trades_creates_file(self, sample_trades: list[TradeRecord], tmp_output_dir: Path):
        """write_trades_json should create JSON file."""
        # Arrange
        output_path = tmp_output_dir / "trades.json"

        # Act
        write_trades_json(sample_trades, output_path)

        # Assert
        assert output_path.exists()

    def test_write_trades_content_valid(self, sample_trades: list[TradeRecord], tmp_output_dir: Path):
        """JSON content should match input data."""
        # Arrange
        output_path = tmp_output_dir / "trades.json"

        # Act
        write_trades_json(sample_trades, output_path)

        # Assert
        with open(output_path) as f:
            data = json.load(f)
        assert len(data) == 2
        assert "trade_id" in data[0]
        assert "symbol" in data[0]
        assert "pnl" in data[0]
        assert data[0]["symbol"] == "AAPL"
        assert data[0]["is_winner"]
        assert not data[1]["is_winner"]

    def test_write_trades_empty_list_logs_warning(self, tmp_output_dir: Path, caplog):
        """Empty trades should log warning and not create file."""
        # Arrange
        output_path = tmp_output_dir / "trades.json"

        # Act
        write_trades_json([], output_path)

        # Assert
        assert not output_path.exists()
        assert "No trades to write" in caplog.text


# ============================================================================
# Drawdowns Writer Tests
# ============================================================================


class TestWriteDrawdownsJson:
    """Test drawdowns JSON writer."""

    def test_write_drawdowns_creates_file(self, sample_drawdowns: list[DrawdownPeriod], tmp_output_dir: Path):
        """write_drawdowns_json should create JSON file."""
        # Arrange
        output_path = tmp_output_dir / "drawdowns.json"

        # Act
        write_drawdowns_json(sample_drawdowns, output_path)

        # Assert
        assert output_path.exists()

    def test_write_drawdowns_content_valid(self, sample_drawdowns: list[DrawdownPeriod], tmp_output_dir: Path):
        """JSON content should match input data."""
        # Arrange
        output_path = tmp_output_dir / "drawdowns.json"

        # Act
        write_drawdowns_json(sample_drawdowns, output_path)

        # Assert
        with open(output_path) as f:
            data = json.load(f)
        assert len(data) == 2
        assert "drawdown_id" in data[0]
        assert "depth_pct" in data[0]
        assert data[0]["recovered"]
        assert not data[1]["recovered"]

    def test_write_drawdowns_empty_list_logs_warning(self, tmp_output_dir: Path, caplog):
        """Empty drawdowns should log warning and not create file."""
        # Arrange
        output_path = tmp_output_dir / "drawdowns.json"

        # Act
        write_drawdowns_json([], output_path)

        # Assert
        assert not output_path.exists()
        assert "No drawdowns to write" in caplog.text


# ============================================================================
# Chart Data Writer Tests
# ============================================================================


class TestWriteStrategyChartData:
    """Test chart data JSON writer."""

    def test_write_chart_data_creates_file(self, sample_event_store: InMemoryEventStore, tmp_output_dir: Path):
        """write_strategy_chart_data should create JSON file."""
        # Arrange
        output_path = tmp_output_dir / "chart_data.json"

        # Act
        write_strategy_chart_data(sample_event_store, "test_strategy", output_path)

        # Assert
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_write_chart_data_has_correct_schema(self, sample_event_store: InMemoryEventStore, tmp_output_dir: Path):
        """JSON should have uniform schema with all expected fields."""
        # Arrange
        output_path = tmp_output_dir / "chart_data.json"

        # Act
        write_strategy_chart_data(sample_event_store, "test_strategy", output_path)

        # Assert
        with open(output_path) as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) > 0

        # Check first row has expected fields
        first_row = data[0]
        expected_fields = [
            "timestamp",
            "strategy_id",
            "ticker",
            "underlying",
            "open",
            "high",
            "low",
            "close",
            "volume",
            # Signal fields (Tier 1 + Tier 3)
            "signal_intention",
            "signal_price",
            "signal_confidence",  # NEW: Tier 1 - Signal strength (0.0-1.0)
            "signal_reason",  # NEW: Tier 3 - Why signal was generated
            "signal_event_id",
            "signal_correlation_id",
            "signal_causation_id",
            "signal_source_service",
            # Order fields (Tier 1 + Tier 2 + Tier 3)
            "order_id",
            "order_side",  # NEW: Tier 1 - BUY/SELL
            "order_type",  # NEW: Tier 2 - MARKET/LIMIT/STOP
            "order_qty",
            "order_timestamp",  # NEW: Tier 3 - Order creation time
            "order_event_id",
            "order_correlation_id",
            "order_causation_id",
            "order_source_service",
            # Fill fields (Tier 1 + Tier 2 + Tier 3)
            "fill_id",
            "fill_side",  # NEW: Tier 1 - BUY/SELL
            "fill_qty",
            "fill_price",
            "fill_slippage_bps",  # NEW: Tier 2 - Execution quality metric
            "fill_timestamp",  # NEW: Tier 3 - Fill execution time
            "commission",
            "fill_event_id",
            "fill_correlation_id",
            "fill_causation_id",
            "fill_source_service",
            # Trade fields (Tier 2 + Tier 3)
            "trade_id",
            "trade_status",  # NEW: Tier 2 - OPEN/CLOSED
            "trade_side",  # NEW: Tier 2 - LONG/SHORT
            "trade_entry_price",  # NEW: Tier 3 - Average entry price
            "trade_exit_price",  # NEW: Tier 3 - Average exit price
            "trade_realized_pnl",  # NEW: Tier 2 - P&L when closed
        ]
        for field in expected_fields:
            assert field in first_row, f"Missing field: {field}"

    def test_write_chart_data_includes_bar_rows(self, sample_event_store: InMemoryEventStore, tmp_output_dir: Path):
        """JSON should include objects for price bars."""
        # Arrange
        output_path = tmp_output_dir / "chart_data.json"

        # Act
        write_strategy_chart_data(sample_event_store, "test_strategy", output_path)

        # Assert
        with open(output_path) as f:
            data = json.load(f)

        # Find bar rows (ticker == underlying, not synthetic)
        bar_rows = [r for r in data if r["ticker"] == "AAPL" and r["underlying"] == "AAPL" and r["open"] is not None]
        assert len(bar_rows) == 2
        assert bar_rows[0]["open"] == 150.0
        assert bar_rows[0]["close"] == 151.0
        assert bar_rows[0]["volume"] == 1000000

    def test_write_chart_data_includes_indicator_rows(
        self, sample_event_store: InMemoryEventStore, tmp_output_dir: Path
    ):
        """JSON should include synthetic ticker objects for indicators."""
        # Arrange
        output_path = tmp_output_dir / "chart_data.json"

        # Act
        write_strategy_chart_data(sample_event_store, "test_strategy", output_path)

        # Assert
        with open(output_path) as f:
            data = json.load(f)

        # Find indicator rows (ticker is indicator name, underlying is symbol)
        indicator_rows = [r for r in data if r["ticker"] == "sma_10"]
        assert len(indicator_rows) == 2
        assert indicator_rows[0]["underlying"] == "AAPL"
        assert float(indicator_rows[0]["close"]) == pytest.approx(150.5)
        assert indicator_rows[0]["open"] is None  # Should be None for indicators

    def test_write_chart_data_includes_portfolio_metrics(
        self, sample_event_store: InMemoryEventStore, tmp_output_dir: Path
    ):
        """JSON should include synthetic ticker objects for portfolio metrics."""
        # Arrange
        output_path = tmp_output_dir / "chart_data.json"

        # Act
        write_strategy_chart_data(sample_event_store, "test_strategy", output_path)

        # Assert
        with open(output_path) as f:
            data = json.load(f)

        # Find portfolio metric rows
        equity_rows = [r for r in data if r["ticker"] == "EQUITY"]
        assert len(equity_rows) == 2
        assert equity_rows[0]["underlying"] == "PORTFOLIO"
        assert float(equity_rows[0]["close"]) == 100000.0

        # Check cash metric rows (both timestamps should have cash data)
        cash_rows = [r for r in data if r["ticker"] == "CASH"]
        assert len(cash_rows) == 2  # Both timestamps have cash
        assert float(cash_rows[0]["close"]) == 85000.0

    def test_write_chart_data_includes_trading_events(
        self, sample_event_store: InMemoryEventStore, tmp_output_dir: Path
    ):
        """JSON should include signal, order, and fill data in bar objects."""
        # Arrange
        output_path = tmp_output_dir / "chart_data.json"

        # Act
        write_strategy_chart_data(sample_event_store, "test_strategy", output_path)

        # Assert
        with open(output_path) as f:
            data = json.load(f)

        # Find bar row with trading events
        bar_with_events = [r for r in data if r["ticker"] == "AAPL" and r["signal_intention"]]
        assert len(bar_with_events) == 1
        assert bar_with_events[0]["signal_intention"] == "OPEN_LONG"
        assert float(bar_with_events[0]["signal_price"]) == 151.0
        assert int(bar_with_events[0]["order_qty"]) == 100
        assert int(bar_with_events[0]["fill_qty"]) == 100
        assert float(bar_with_events[0]["fill_price"]) == pytest.approx(151.10)
        assert float(bar_with_events[0]["commission"]) == pytest.approx(1.50)

    def test_write_chart_data_filters_by_strategy(self, tmp_output_dir: Path):
        """CSV should only include events for the specified strategy."""
        # Arrange
        store = InMemoryEventStore()

        # Add indicator for test_strategy
        indicator1 = IndicatorEvent(
            strategy_id="test_strategy",
            symbol="AAPL",
            timestamp="2023-01-01T10:00:00+00:00",
            indicators={"sma_10": 150.5},
        )
        # Add indicator for other_strategy
        indicator2 = IndicatorEvent(
            strategy_id="other_strategy",
            symbol="AAPL",
            timestamp="2023-01-01T10:00:00+00:00",
            indicators={"sma_10": 160.5},
        )
        store.append(indicator1)
        store.append(indicator2)

        output_path = tmp_output_dir / "chart_data.json"

        # Act
        write_strategy_chart_data(store, "test_strategy", output_path)

        # Assert
        with open(output_path) as f:
            data = json.load(f)

        # Should only have indicator for test_strategy
        indicator_rows = [r for r in data if r["ticker"] == "sma_10"]
        assert len(indicator_rows) == 1
        assert float(indicator_rows[0]["close"]) == pytest.approx(150.5)

    def test_write_chart_data_skips_non_numeric_indicators(self, tmp_output_dir: Path):
        """JSON should skip string indicators but convert boolean to numeric."""
        # Arrange
        store = InMemoryEventStore()

        indicator = IndicatorEvent(
            strategy_id="test_strategy",
            symbol="AAPL",
            timestamp="2023-01-01T10:00:00+00:00",
            indicators={
                "sma_10": 150.5,  # numeric - should be included
                "crossover": True,  # boolean - converts to 1.0
                "signal_text": "bullish",  # string - should be skipped
            },
        )
        store.append(indicator)

        output_path = tmp_output_dir / "chart_data.json"

        # Act
        write_strategy_chart_data(store, "test_strategy", output_path)

        # Assert
        with open(output_path) as f:
            data = json.load(f)

        indicator_rows = [r for r in data if r["ticker"] in ("sma_10", "crossover", "signal_text")]
        # Should have sma_10 and crossover (boolean converts to float), but not signal_text (string)
        assert len(indicator_rows) == 2
        sma_row = [r for r in indicator_rows if r["ticker"] == "sma_10"][0]
        assert float(sma_row["close"]) == pytest.approx(150.5)
        cross_row = [r for r in indicator_rows if r["ticker"] == "crossover"][0]
        assert float(cross_row["close"]) == pytest.approx(1.0)

    def test_write_chart_data_empty_store_logs_warning(self, tmp_output_dir: Path, caplog):
        """Empty event store should log warning."""
        # Arrange
        store = InMemoryEventStore()
        output_path = tmp_output_dir / "chart_data.json"

        # Act
        write_strategy_chart_data(store, "test_strategy", output_path)

        # Assert
        assert "strategy_chart_data.no_data" in caplog.text

    def test_write_chart_data_aggregates_multiple_orders(self, tmp_output_dir: Path):
        """Multiple orders at same timestamp should be aggregated."""
        # Arrange
        store = InMemoryEventStore()

        bar = PriceBarEvent.model_construct(
            symbol="AAPL",
            timestamp="2023-01-01T10:00:00+00:00",
            open=Decimal("150.00"),
            high=Decimal("152.00"),
            low=Decimal("149.00"),
            close=Decimal("151.00"),
            volume=1000000,
            source="test",
        )
        order1 = OrderEvent.model_construct(
            symbol="AAPL",
            timestamp="2023-01-01T10:00:00+00:00",
            quantity=50,
            side="BUY",
            order_type="MARKET",
            intent_id="intent_001",
            idempotency_key="idem_001",
        )
        order2 = OrderEvent.model_construct(
            symbol="AAPL",
            timestamp="2023-01-01T10:00:00+00:00",
            quantity=30,
            side="BUY",
            order_type="MARKET",
            intent_id="intent_002",
            idempotency_key="idem_002",
        )
        store.append(bar)
        store.append(order1)
        store.append(order2)

        output_path = tmp_output_dir / "chart_data.json"

        # Act
        write_strategy_chart_data(store, "test_strategy", output_path)

        # Assert
        with open(output_path) as f:
            data = json.load(f)

        bar_rows = [r for r in data if r["ticker"] == "AAPL" and r["order_qty"]]
        assert len(bar_rows) == 1
        assert int(bar_rows[0]["order_qty"]) == 80  # 50 + 30

    def test_write_chart_data_aggregates_multiple_fills(self, tmp_output_dir: Path):
        """Multiple fills at same timestamp should be aggregated with weighted average price."""
        # Arrange
        store = InMemoryEventStore()

        bar = PriceBarEvent.model_construct(
            symbol="AAPL",
            timestamp="2023-01-01T10:00:00+00:00",
            open=Decimal("150.00"),
            high=Decimal("152.00"),
            low=Decimal("149.00"),
            close=Decimal("151.00"),
            volume=1000000,
            source="test",
        )
        fill1 = FillEvent.model_construct(
            symbol="AAPL",
            timestamp="2023-01-01T10:00:00+00:00",
            filled_quantity=50,
            fill_price=Decimal("151.00"),
            commission=Decimal("1.00"),
            fill_id="fill_001",
            source_order_id="order_001",
            side="BUY",
        )
        fill2 = FillEvent.model_construct(
            symbol="AAPL",
            timestamp="2023-01-01T10:00:00+00:00",
            filled_quantity=30,
            fill_price=Decimal("151.50"),
            commission=Decimal("0.60"),
            fill_id="fill_002",
            source_order_id="order_002",
            side="BUY",
        )
        store.append(bar)
        store.append(fill1)
        store.append(fill2)

        output_path = tmp_output_dir / "chart_data.json"

        # Act
        write_strategy_chart_data(store, "test_strategy", output_path)

        # Assert
        with open(output_path) as f:
            data = json.load(f)

        bar_rows = [r for r in data if r["ticker"] == "AAPL" and r["fill_qty"]]
        assert len(bar_rows) == 1
        assert int(bar_rows[0]["fill_qty"]) == 80  # 50 + 30
        # Weighted average: (50*151.00 + 30*151.50) / 80 = 151.1875
        assert float(bar_rows[0]["fill_price"]) == pytest.approx(151.1875)
        assert float(bar_rows[0]["commission"]) == pytest.approx(1.60)  # 1.00 + 0.60

    def test_write_chart_data_creates_parent_dirs(self, sample_event_store: InMemoryEventStore, tmp_output_dir: Path):
        """Writer should create parent directories if missing."""
        # Arrange
        output_path = tmp_output_dir / "nested" / "dir" / "chart_data.json"

        # Act
        write_strategy_chart_data(sample_event_store, "test_strategy", output_path)

        # Assert
        assert output_path.exists()

    def test_write_chart_data_timestamps_sorted(self, tmp_output_dir: Path):
        """JSON objects should be sorted by timestamp."""
        # Arrange
        store = InMemoryEventStore()

        # Add bars out of order
        bar2 = PriceBarEvent.model_construct(
            symbol="AAPL",
            timestamp="2023-01-03T10:00:00+00:00",
            open=Decimal("152.00"),
            high=Decimal("153.00"),
            low=Decimal("151.50"),
            close=Decimal("152.50"),
            volume=1100000,
            source="test",
        )
        bar1 = PriceBarEvent.model_construct(
            symbol="AAPL",
            timestamp="2023-01-01T10:00:00+00:00",
            open=Decimal("150.00"),
            high=Decimal("152.00"),
            low=Decimal("149.00"),
            close=Decimal("151.00"),
            volume=1000000,
            source="test",
        )
        store.append(bar2)
        store.append(bar1)

        output_path = tmp_output_dir / "chart_data.json"

        # Act
        write_strategy_chart_data(store, "test_strategy", output_path)

        # Assert
        with open(output_path) as f:
            data = json.load(f)

        bar_rows = [r for r in data if r["ticker"] == "AAPL"]
        assert bar_rows[0]["timestamp"] == "2023-01-01T10:00:00+00:00"
        assert bar_rows[1]["timestamp"] == "2023-01-03T10:00:00+00:00"

    def test_write_chart_data_includes_indicator_display_names(self, tmp_output_dir: Path):
        """CSV should include indicator display names with parameters."""
        # Arrange
        store = InMemoryEventStore()

        bar = PriceBarEvent.model_construct(
            symbol="AAPL",
            timestamp="2023-01-01T10:00:00+00:00",
            open=Decimal("150.00"),
            high=Decimal("152.00"),
            low=Decimal("149.00"),
            close=Decimal("151.00"),
            volume=1000000,
            source="test",
        )

        # Indicator event with display names showing parameters
        indicator = IndicatorEvent(
            strategy_id="test_strategy",
            symbol="AAPL",
            timestamp="2023-01-01T10:00:00+00:00",
            indicators={
                "fast_sma(10)": 150.5,
                "slow_sma(50)": 149.8,
                "rsi(14)": 65.4,
                "bb_upper(period=20,std=2.0)": 152.0,
            },
        )

        store.append(bar)
        store.append(indicator)

        output_path = tmp_output_dir / "chart_data.json"

        # Act
        write_strategy_chart_data(store, "test_strategy", output_path)

        # Assert
        with open(output_path) as f:
            data = json.load(f)

        # Find indicator rows with display names
        fast_sma_rows = [r for r in data if r["ticker"] == "fast_sma(10)"]
        slow_sma_rows = [r for r in data if r["ticker"] == "slow_sma(50)"]
        rsi_rows = [r for r in data if r["ticker"] == "rsi(14)"]
        bb_rows = [r for r in data if r["ticker"] == "bb_upper(period=20,std=2.0)"]

        assert len(fast_sma_rows) == 1
        assert fast_sma_rows[0]["underlying"] == "AAPL"
        assert float(fast_sma_rows[0]["close"]) == pytest.approx(150.5)

        assert len(slow_sma_rows) == 1
        assert float(slow_sma_rows[0]["close"]) == pytest.approx(149.8)

        assert len(rsi_rows) == 1
        assert float(rsi_rows[0]["close"]) == pytest.approx(65.4)

        assert len(bb_rows) == 1
        assert float(bb_rows[0]["close"]) == pytest.approx(152.0)
