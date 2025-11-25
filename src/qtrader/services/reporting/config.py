"""Configuration for ReportingService.

Configuration options control:
- Event emission frequency and display
- Risk-free rate for risk-adjusted metrics
- Equity curve sampling for memory efficiency
- Output format and location
"""

from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ReportingConfig(BaseModel):
    """
    Configuration for performance reporting and metrics calculation.

    Implements user decisions from architecture design:
    1. Benchmark: Not implemented yet, but extensible
    2. Risk-free rate: Configurable (default 0.0)
    3. Time periods: Always calculate monthly/quarterly/annual
    4. Strategy metrics: Always calculate both portfolio and per-strategy
    5. Event display: Configurable flag
    6. Output formats: JSON + Parquet
    7. Trade definition: Round-trip (open + close)
    8. Sampling: Configurable max_equity_points
    """

    # Event emission
    emit_metrics_events: bool = Field(
        default=False,
        description="Emit PerformanceMetricsEvent during backtest for console display",
    )
    event_frequency: int = Field(
        default=100,
        description="Emit event every N bars (only if emit_metrics_events=True)",
    )

    # Risk-free rate for Sharpe/Sortino calculations
    risk_free_rate: Decimal = Field(
        default=Decimal("0.0"),
        description="Annual risk-free rate as decimal (e.g., 0.02 for 2%)",
    )

    # Equity curve sampling
    max_equity_points: int = Field(
        default=10_000,
        description="Maximum equity curve points to store (sampling applied if exceeded)",
    )

    # Output configuration
    # Note: output_dir comes from SystemConfig.output.experiments_root
    # and is passed to get_output_path() method
    write_json: bool = Field(
        default=True,
        description="Write performance.json summary report",
    )
    write_parquet: bool = Field(
        default=True,
        description="Write Parquet time-series files (equity_curve, returns, trades, drawdowns)",
    )
    write_csv_timeline: bool = Field(
        default=True,
        description="Write per-strategy CSV timeline files (human-friendly, one file per strategy)",
    )

    # Time-series output
    include_equity_curve: bool = Field(
        default=True,
        description="Include equity_curve.parquet in output",
    )
    include_returns: bool = Field(
        default=True,
        description="Include returns.parquet in output",
    )
    include_trades: bool = Field(
        default=True,
        description="Include trades.parquet in output",
    )
    include_drawdowns: bool = Field(
        default=True,
        description="Include drawdowns.parquet in output",
    )

    # Console output
    display_final_report: bool = Field(
        default=True,
        description="Display rich-formatted final report in console at teardown",
    )
    report_detail_level: Literal["summary", "standard", "full"] = Field(
        default="full",
        description="Report detail level: 'summary', 'standard', 'full'",
    )

    # HTML report output
    write_html_report: bool = Field(
        default=True,
        description="Generate standalone HTML report with interactive charts",
    )

    # Period breakdowns (always calculated, per user decision 3)
    calculate_monthly: bool = Field(
        default=True,
        description="Calculate monthly period metrics (always True per design decision)",
    )
    calculate_quarterly: bool = Field(
        default=True,
        description="Calculate quarterly period metrics (always True per design decision)",
    )
    calculate_annual: bool = Field(
        default=True,
        description="Calculate annual period metrics (always True per design decision)",
    )

    # Strategy metrics (always calculated, per user decision 4)
    calculate_strategy_metrics: bool = Field(
        default=True,
        description="Calculate per-strategy attribution (always True per design decision)",
    )

    # Benchmark support (extensible, not implemented yet per user decision 1)
    benchmark_symbol: str | None = Field(
        default=None,
        description="Benchmark symbol for comparison (not implemented yet)",
    )

    def get_output_path(self, output_dir: Path) -> Path:
        """
        Get output directory for backtest results.

        The output_dir is the complete timestamped directory created by the engine:
        {base_dir}/{backtest_id}/{timestamp}/

        Args:
            output_dir: Full timestamped output directory from engine

        Returns:
            Path to output directory (same as input, for consistency)
        """
        return output_dir

    def get_timeseries_path(self, output_dir: Path) -> Path:
        """
        Get time-series data directory for backtest.

        Args:
            output_dir: Full timestamped output directory from engine

        Returns:
            Path to time-series subdirectory
        """
        return output_dir / "timeseries"
