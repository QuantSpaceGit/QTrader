"""Reporting service for performance metrics and analysis."""

from qtrader.services.reporting.config import ReportingConfig
from qtrader.services.reporting.formatters import display_performance_report
from qtrader.services.reporting.service import ReportingService
from qtrader.services.reporting.writers import (
    write_drawdowns_json,
    write_equity_curve_json,
    write_json_report,
    write_returns_json,
    write_strategy_chart_data,
    write_trades_json,
)

__all__ = [
    "ReportingService",
    "ReportingConfig",
    "display_performance_report",
    "write_json_report",
    "write_equity_curve_json",
    "write_returns_json",
    "write_trades_json",
    "write_drawdowns_json",
    "write_strategy_chart_data",
]
