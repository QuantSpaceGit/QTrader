"""Integration tests for ReportingService with BacktestEngine.

Tests the full end-to-end flow of reporting service integration using fixtures.
All tests are self-contained and don't depend on user-modifiable config files.
"""

import json
from pathlib import Path
from unittest.mock import patch

from qtrader.engine.engine import BacktestEngine


class TestReportingIntegration:
    """Integration tests for reporting service."""

    def test_backtest_generates_performance_json(self, buy_hold_backtest_config, mock_system_config):
        """Test that backtest generates performance.json output.

        Uses programmatically-created config, not external files.
        Writes to temporary directory, not user-facing output folder.
        """
        config = buy_hold_backtest_config

        # Create engine with mocked system config
        with patch("qtrader.engine.engine.get_system_config", return_value=mock_system_config):
            engine = BacktestEngine.from_config(config)

            # Run backtest
            result = engine.run()

        # Check result
        assert result.bars_processed > 0

        # Find the output directory - use temp dir from mock config
        output_base = Path(mock_system_config.output.experiments_root) / config.sanitized_backtest_id
        assert output_base.exists()

        # Should have runs/ directory with timestamped subdirectories
        runs_dir = output_base / "runs"
        assert runs_dir.exists()
        timestamped_dirs = list(runs_dir.iterdir())
        assert len(timestamped_dirs) > 0

        latest_dir = max(timestamped_dirs, key=lambda p: p.name)

        # Check for performance.json
        perf_json = latest_dir / "performance.json"
        assert perf_json.exists(), f"performance.json not found in {latest_dir}"

        # Load and validate JSON
        with open(perf_json) as f:
            perf_data = json.load(f)

        # Validate key fields
        assert "backtest_id" in perf_data
        assert "total_return_pct" in perf_data
        assert "cagr" in perf_data
        assert "sharpe_ratio" in perf_data
        assert "total_trades" in perf_data

        # Validate metrics are reasonable
        assert float(perf_data["total_return_pct"]) != 0.0  # Should have returns
        assert int(perf_data["total_trades"]) >= 0  # Should have trade count

        # Cleanup
        engine.shutdown()

    def test_backtest_generates_timeseries_json(self, buy_hold_backtest_config, mock_system_config):
        """Test that backtest generates time-series JSON files.

        Uses programmatically-created config, not external files.
        Writes to temporary directory, not user-facing output folder.
        """
        config = buy_hold_backtest_config

        with patch("qtrader.engine.engine.get_system_config", return_value=mock_system_config):
            engine = BacktestEngine.from_config(config)
            result = engine.run()

        assert result.bars_processed > 0

        # Find output directory - use temp dir from mock config
        output_base = Path(mock_system_config.output.experiments_root) / config.sanitized_backtest_id
        runs_dir = output_base / "runs"
        timestamped_dirs = list(runs_dir.iterdir())
        latest_dir = max(timestamped_dirs, key=lambda p: p.name)

        # Check for timeseries directory
        ts_dir = latest_dir / "timeseries"
        assert ts_dir.exists(), f"timeseries directory not found in {latest_dir}"

        # Check for core expected JSON files
        # Note: trades.json is only generated if there are closed trades
        expected_files = [
            "equity_curve.json",
            "returns.json",
            "drawdowns.json",
        ]

        for filename in expected_files:
            filepath = ts_dir / filename
            assert filepath.exists(), f"{filename} not found in {ts_dir}"
            assert filepath.stat().st_size > 0, f"{filename} is empty"

        # trades.json is optional (only generated if there are closed trades)
        trades_file = ts_dir / "trades.json"
        if trades_file.exists():
            assert trades_file.stat().st_size > 0, "trades.json is empty"  # Cleanup
        engine.shutdown()

    def test_backtest_event_store_and_reporting_same_directory(self, buy_hold_backtest_config, mock_system_config):
        """Test that event store and reporting outputs are in same timestamped directory.

        Uses programmatically-created config, not external files.
        Writes to temporary directory, not user-facing output folder.
        """
        config = buy_hold_backtest_config

        with patch("qtrader.engine.engine.get_system_config", return_value=mock_system_config):
            engine = BacktestEngine.from_config(config)
            result = engine.run()

        assert result.bars_processed > 0

        # Find output directory - use temp dir from mock config
        output_base = Path(mock_system_config.output.experiments_root) / config.sanitized_backtest_id
        runs_dir = output_base / "runs"
        timestamped_dirs = list(runs_dir.iterdir())
        latest_dir = max(timestamped_dirs, key=lambda p: p.name)

        # Check that both event store and reporting are in same directory
        # (assuming parquet event store is configured in system.yaml)

        # Reporting outputs
        perf_json = latest_dir / "performance.json"
        ts_dir = latest_dir / "timeseries"

        assert perf_json.exists()
        assert ts_dir.exists()

        # Event store output (if using parquet backend)
        events_parquet = latest_dir / "events.parquet"
        if events_parquet.exists():
            # If event store is enabled, verify it's in same directory
            assert events_parquet.parent == perf_json.parent
            assert events_parquet.stat().st_size > 0

        # Cleanup
        engine.shutdown()

    def test_performance_metrics_accuracy(self, buy_hold_backtest_config, mock_system_config):
        """Test that calculated metrics match expected values for known backtest.

        Uses programmatically-created config, not external files.
        Writes to temporary directory, not user-facing output folder.
        """
        config = buy_hold_backtest_config

        with patch("qtrader.engine.engine.get_system_config", return_value=mock_system_config):
            engine = BacktestEngine.from_config(config)
            _ = engine.run()

        # Find performance.json - use temp dir from mock config
        output_base = Path(mock_system_config.output.experiments_root) / config.sanitized_backtest_id
        runs_dir = output_base / "runs"
        timestamped_dirs = list(runs_dir.iterdir())
        latest_dir = max(timestamped_dirs, key=lambda p: p.name)
        perf_json = latest_dir / "performance.json"

        with open(perf_json) as f:
            metrics = json.load(f)

        # Validate metric relationships and sanity checks
        total_return = float(metrics["total_return_pct"])
        cagr = float(metrics["cagr"])

        # If positive return, CAGR should also be positive
        if total_return > 0:
            assert cagr > 0, "CAGR should be positive with positive returns"

        # Sharpe ratio should be reasonable
        sharpe = float(metrics["sharpe_ratio"])
        assert -10 < sharpe < 10, f"Sharpe ratio {sharpe} seems unreasonable"

        # Trade statistics should be consistent
        total_trades = int(metrics["total_trades"])
        winning_trades = int(metrics["winning_trades"])
        losing_trades = int(metrics["losing_trades"])

        assert total_trades == winning_trades + losing_trades, "Total trades should equal winning + losing trades"

        if total_trades > 0:
            win_rate = float(metrics["win_rate"])
            expected_win_rate = (winning_trades / total_trades) * 100
            assert abs(win_rate - expected_win_rate) < 0.01, (
                f"Win rate {win_rate} doesn't match expected {expected_win_rate}"
            )

        # Cleanup
        engine.shutdown()

    def test_reporting_handles_zero_trades(self, zero_trades_backtest_config, mock_system_config):
        """Test that reporting handles backtests with zero trades gracefully.

        Uses config with empty strategy universe to generate zero trades.
        Writes to temporary directory, not user-facing output folder.
        """
        config = zero_trades_backtest_config

        with patch("qtrader.engine.engine.get_system_config", return_value=mock_system_config):
            engine = BacktestEngine.from_config(config)
            result = engine.run()

        assert result.bars_processed > 0, "Should have processed bars"

        # Find output directory - use temp dir from mock config
        output_base = Path(mock_system_config.output.experiments_root) / config.sanitized_backtest_id
        runs_dir = output_base / "runs"
        timestamped_dirs = list(runs_dir.iterdir())
        latest_dir = max(timestamped_dirs, key=lambda p: p.name)

        # Check performance.json exists
        perf_json = latest_dir / "performance.json"
        assert perf_json.exists()

        with open(perf_json) as f:
            metrics = json.load(f)

        # Validate zero trades scenario
        assert int(metrics["total_trades"]) == 0, "Should have zero trades"
        assert int(metrics["winning_trades"]) == 0
        assert int(metrics["losing_trades"]) == 0
        assert float(metrics["win_rate"]) == 0.0

        # Note: Returns may be non-zero due to dividends/corporate actions
        # The key is that trade statistics are correctly handled at zero
        total_return = float(metrics["total_return_pct"])
        assert -100 < total_return < 100, "Return should be reasonable even with zero trades"

        # Cleanup
        engine.shutdown()

    def test_total_commissions_includes_open_positions(self, buy_hold_backtest_config, mock_system_config):
        """Test that total_commissions includes commissions from open positions.

        Buy-and-hold strategy opens a position but never closes it.
        The commission from the entry fill should still be reported in total_commissions.

        Regression test for bug where total_commissions only summed from closed trades,
        showing $0.00 for strategies with only open positions.
        """
        config = buy_hold_backtest_config

        with patch("qtrader.engine.engine.get_system_config", return_value=mock_system_config):
            engine = BacktestEngine.from_config(config)
            result = engine.run()

        assert result.bars_processed > 0

        # Find output directory
        output_base = Path(mock_system_config.output.experiments_root) / config.sanitized_backtest_id
        runs_dir = output_base / "runs"
        timestamped_dirs = list(runs_dir.iterdir())
        latest_dir = max(timestamped_dirs, key=lambda p: p.name)

        # Load performance.json
        perf_json = latest_dir / "performance.json"
        with open(perf_json) as f:
            metrics = json.load(f)

        # Buy-and-hold with default commission ($0.005/share, $1.00 min) should have non-zero commission
        # The strategy buys ~$100k of AAPL at ~$100-500/share = ~200-1000 shares
        # Commission = max(shares * $0.005, $1.00) > $1.00
        total_commissions = float(metrics["total_commissions"])
        assert total_commissions > 0, (
            f"total_commissions should be > 0 for buy-and-hold with open position, got {total_commissions}"
        )

        # Verify no closed trades (position is still open)
        total_trades = int(metrics["total_trades"])
        assert total_trades == 0, f"Buy-and-hold should have 0 closed trades, got {total_trades}"

        # The fact that total_commissions > 0 while total_trades == 0 proves
        # we're correctly including commissions from open positions
        engine.shutdown()

    def test_reporting_handles_large_backtest(self, buy_hold_backtest_config, mock_system_config):
        """Test that reporting correctly samples equity curve for large backtests.

        Verifies max_equity_points sampling works correctly.
        Writes to temporary directory, not user-facing output folder.
        """
        # Extend date range for more data points
        config = buy_hold_backtest_config
        config.reporting.max_equity_points = 50  # Force sampling

        with patch("qtrader.engine.engine.get_system_config", return_value=mock_system_config):
            engine = BacktestEngine.from_config(config)
            result = engine.run()

        assert result.bars_processed > 0

        # Find output - use temp dir from mock config
        output_base = Path(mock_system_config.output.experiments_root) / config.sanitized_backtest_id
        runs_dir = output_base / "runs"
        timestamped_dirs = list(runs_dir.iterdir())
        latest_dir = max(timestamped_dirs, key=lambda p: p.name)

        # Check equity curve was sampled
        import json

        equity_file = latest_dir / "timeseries" / "equity_curve.json"
        with open(equity_file) as f:
            data = json.load(f)
        row_count = len(data)

        # Should be <= max_equity_points (plus a few for peaks/troughs)
        assert row_count <= config.reporting.max_equity_points + 10, f"Equity curve should be sampled: {row_count} rows"

        # Cleanup
        engine.shutdown()

    def test_performance_json_contains_period_aggregations(self, buy_hold_backtest_config, mock_system_config):
        """Test that performance.json includes period aggregation metrics.

        Validates that monthly, quarterly, and annual returns are populated
        with correct structure and data types.
        """
        config = buy_hold_backtest_config

        with patch("qtrader.engine.engine.get_system_config", return_value=mock_system_config):
            engine = BacktestEngine.from_config(config)
            _ = engine.run()

        # Find performance.json
        output_base = Path(mock_system_config.output.experiments_root) / config.sanitized_backtest_id
        runs_dir = output_base / "runs"
        timestamped_dirs = list(runs_dir.iterdir())
        latest_dir = max(timestamped_dirs, key=lambda p: p.name)
        perf_json = latest_dir / "performance.json"

        with open(perf_json) as f:
            metrics = json.load(f)

        # Validate monthly_returns field exists and has correct structure
        assert "monthly_returns" in metrics, "monthly_returns field missing from performance.json"
        monthly_returns = metrics["monthly_returns"]
        assert isinstance(monthly_returns, list), "monthly_returns should be a list"

        if monthly_returns:
            # Validate structure of first monthly period
            month = monthly_returns[0]
            assert "period" in month, "Monthly period missing 'period' field"
            assert "period_type" in month, "Monthly period missing 'period_type' field"
            assert "start_date" in month, "Monthly period missing 'start_date' field"
            assert "end_date" in month, "Monthly period missing 'end_date' field"
            assert "return_pct" in month, "Monthly period missing 'return_pct' field"
            assert "num_trades" in month, "Monthly period missing 'num_trades' field"
            assert "winning_trades" in month, "Monthly period missing 'winning_trades' field"
            assert "losing_trades" in month, "Monthly period missing 'losing_trades' field"

            # Validate period format (YYYY-MM)
            assert len(month["period"]) == 7, "Monthly period format should be YYYY-MM"
            assert month["period"][4] == "-", "Monthly period should contain hyphen"
            assert month["period_type"] == "monthly", "Period type should be 'monthly'"

            # Validate data types
            assert isinstance(month["return_pct"], str), "return_pct should be string (Decimal)"
            assert isinstance(month["num_trades"], int), "num_trades should be integer"

        # Validate quarterly_returns field exists and has correct structure
        assert "quarterly_returns" in metrics, "quarterly_returns field missing from performance.json"
        quarterly_returns = metrics["quarterly_returns"]
        assert isinstance(quarterly_returns, list), "quarterly_returns should be a list"

        if quarterly_returns:
            # Validate structure of first quarterly period
            quarter = quarterly_returns[0]
            assert "period" in quarter, "Quarterly period missing 'period' field"
            assert "period_type" in quarter, "Quarterly period missing 'period_type' field"

            # Validate period format (YYYY-Q#)
            assert "Q" in quarter["period"], "Quarterly period should contain 'Q'"
            assert quarter["period_type"] == "quarterly", "Period type should be 'quarterly'"

        # Validate annual_returns field exists and has correct structure
        assert "annual_returns" in metrics, "annual_returns field missing from performance.json"
        annual_returns = metrics["annual_returns"]
        assert isinstance(annual_returns, list), "annual_returns should be a list"

        if annual_returns:
            # Validate structure of first annual period
            year = annual_returns[0]
            assert "period" in year, "Annual period missing 'period' field"
            assert "period_type" in year, "Annual period missing 'period_type' field"

            # Validate period format (YYYY)
            assert len(year["period"]) == 4, "Annual period format should be YYYY"
            assert year["period_type"] == "annual", "Period type should be 'annual'"

        # Cleanup
        engine.shutdown()

    def test_performance_json_contains_strategy_performance(self, buy_hold_backtest_config, mock_system_config):
        """Test that performance.json includes per-strategy performance metrics.

        Validates that strategy_performance is populated with correct structure
        and contains metrics for each strategy in the backtest.

        Note: Currently strategy_performance may be empty because per-strategy
        equity tracking is not yet implemented. This test validates the structure
        exists and will validate content once equity tracking is added.
        """
        config = buy_hold_backtest_config

        with patch("qtrader.engine.engine.get_system_config", return_value=mock_system_config):
            engine = BacktestEngine.from_config(config)
            _ = engine.run()

        # Find performance.json
        output_base = Path(mock_system_config.output.experiments_root) / config.sanitized_backtest_id
        runs_dir = output_base / "runs"
        timestamped_dirs = list(runs_dir.iterdir())
        latest_dir = max(timestamped_dirs, key=lambda p: p.name)
        perf_json = latest_dir / "performance.json"

        with open(perf_json) as f:
            metrics = json.load(f)

        # Validate strategy_performance field exists
        assert "strategy_performance" in metrics, "strategy_performance field missing from performance.json"
        strategy_performance = metrics["strategy_performance"]
        assert isinstance(strategy_performance, list), "strategy_performance should be a list"

        # If strategies are tracked, validate structure
        if len(strategy_performance) > 0:
            # Validate structure of first strategy
            strategy = strategy_performance[0]
            assert "strategy_id" in strategy, "Strategy missing 'strategy_id' field"
            assert "equity_allocated" in strategy, "Strategy missing 'equity_allocated' field"
            assert "final_equity" in strategy, "Strategy missing 'final_equity' field"
            assert "return_pct" in strategy, "Strategy missing 'return_pct' field"
            assert "num_positions" in strategy, "Strategy missing 'num_positions' field"
            assert "total_trades" in strategy, "Strategy missing 'total_trades' field"
            assert "winning_trades" in strategy, "Strategy missing 'winning_trades' field"
            assert "losing_trades" in strategy, "Strategy missing 'losing_trades' field"
            assert "win_rate" in strategy, "Strategy missing 'win_rate' field"
            assert "avg_win_pct" in strategy, "Strategy missing 'avg_win_pct' field"
            assert "avg_loss_pct" in strategy, "Strategy missing 'avg_loss_pct' field"
            assert "profit_factor" in strategy, "Strategy missing 'profit_factor' field"
            assert "sharpe_ratio" in strategy, "Strategy missing 'sharpe_ratio' field"
            assert "max_drawdown_pct" in strategy, "Strategy missing 'max_drawdown_pct' field"

            # Validate data types
            assert isinstance(strategy["strategy_id"], str), "strategy_id should be string"
            assert isinstance(strategy["equity_allocated"], str), "equity_allocated should be string (Decimal)"
            assert isinstance(strategy["final_equity"], str), "final_equity should be string (Decimal)"
            assert isinstance(strategy["return_pct"], str), "return_pct should be string (Decimal)"
            assert isinstance(strategy["total_trades"], int), "total_trades should be integer"

            # Validate metrics are reasonable
            assert strategy["total_trades"] >= 0, "total_trades should be non-negative"
            assert strategy["total_trades"] == strategy["winning_trades"] + strategy["losing_trades"], (
                "total_trades should equal winning_trades + losing_trades"
            )

        # Cleanup
        engine.shutdown()

    def test_period_aggregations_calculate_correct_returns(self, buy_hold_backtest_config, mock_system_config):
        """Test that period aggregations calculate returns correctly.

        Validates that period returns are calculated accurately and trade
        statistics are correctly assigned to periods.
        """
        config = buy_hold_backtest_config

        with patch("qtrader.engine.engine.get_system_config", return_value=mock_system_config):
            engine = BacktestEngine.from_config(config)
            _ = engine.run()

        # Find performance.json
        output_base = Path(mock_system_config.output.experiments_root) / config.sanitized_backtest_id
        runs_dir = output_base / "runs"
        timestamped_dirs = list(runs_dir.iterdir())
        latest_dir = max(timestamped_dirs, key=lambda p: p.name)
        perf_json = latest_dir / "performance.json"

        with open(perf_json) as f:
            metrics = json.load(f)

        # Validate monthly returns
        monthly_returns = metrics["monthly_returns"]
        if monthly_returns:
            for period in monthly_returns:
                # Return percentage should be reasonable
                return_pct = float(period["return_pct"])
                assert -100 <= return_pct <= 1000, f"Monthly return {return_pct}% seems unreasonable"

                # Trade counts should be consistent
                num_trades = period["num_trades"]
                winning_trades = period["winning_trades"]
                losing_trades = period["losing_trades"]
                assert num_trades == winning_trades + losing_trades, f"Period {period['period']}: trades don't add up"

                # Dates should be valid ISO format
                start_date = period["start_date"]
                end_date = period["end_date"]
                assert len(start_date) == 10, "start_date should be YYYY-MM-DD"
                assert len(end_date) == 10, "end_date should be YYYY-MM-DD"
                assert start_date <= end_date, "start_date should be <= end_date"

        # Validate that periods are in chronological order
        if len(monthly_returns) > 1:
            for i in range(len(monthly_returns) - 1):
                current_period = monthly_returns[i]["period"]
                next_period = monthly_returns[i + 1]["period"]
                assert current_period < next_period, "Monthly periods should be in chronological order"

        # Cleanup
        engine.shutdown()

    def test_strategy_performance_metrics_are_reasonable(self, buy_hold_backtest_config, mock_system_config):
        """Test that strategy performance metrics have reasonable values.

        Validates that strategy metrics are within expected ranges and
        relationships between metrics are consistent.

        Note: Currently strategy_performance may be empty because per-strategy
        equity tracking is not yet implemented. This test will validate metrics
        once equity tracking is added.
        """
        config = buy_hold_backtest_config

        with patch("qtrader.engine.engine.get_system_config", return_value=mock_system_config):
            engine = BacktestEngine.from_config(config)
            _ = engine.run()

        # Find performance.json
        output_base = Path(mock_system_config.output.experiments_root) / config.sanitized_backtest_id
        runs_dir = output_base / "runs"
        timestamped_dirs = list(runs_dir.iterdir())
        latest_dir = max(timestamped_dirs, key=lambda p: p.name)
        perf_json = latest_dir / "performance.json"

        with open(perf_json) as f:
            metrics = json.load(f)

        strategy_performance = metrics["strategy_performance"]

        # If strategies are tracked, validate metrics
        if len(strategy_performance) > 0:
            # Each strategy should have reasonable metrics
            for strategy in strategy_performance:
                # Equity allocated should be non-negative
                equity_allocated = float(strategy["equity_allocated"])
                assert equity_allocated >= 0, f"Strategy {strategy['strategy_id']} has negative allocation"

                # Final equity should be non-negative
                final_equity = float(strategy["final_equity"])
                assert final_equity >= 0, f"Strategy {strategy['strategy_id']} has negative final equity"

                # Win rate should be 0-100
                win_rate = float(strategy["win_rate"])
                assert 0 <= win_rate <= 100, f"Strategy {strategy['strategy_id']} has invalid win rate: {win_rate}"

                # Max drawdown should be non-negative
                max_dd = float(strategy["max_drawdown_pct"])
                assert max_dd >= 0, f"Strategy {strategy['strategy_id']} has negative drawdown: {max_dd}"

                # If there are trades, validate trade statistics
                if strategy["total_trades"] > 0:
                    # Win rate calculation should be consistent
                    expected_win_rate = (strategy["winning_trades"] / strategy["total_trades"]) * 100
                    assert abs(win_rate - expected_win_rate) < 0.01, (
                        f"Strategy {strategy['strategy_id']}: win rate mismatch"
                    )

        # Cleanup
        engine.shutdown()
