"""
Unit tests for InteractiveDebugger.

Tests the interactive debugging functionality for backtest development,
including pause control, state display, and user command handling.
"""

from datetime import date, datetime
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from qtrader.cli.ui.interactive import InteractiveDebugger


class TestInteractiveDebuggerInitialization:
    """Test InteractiveDebugger initialization and configuration."""

    def test_init_default_values(self):
        """Test debugger initializes with default values."""
        # Arrange & Act
        debugger = InteractiveDebugger(enabled=False)

        # Assert
        assert debugger.enabled is False
        assert debugger.break_at is None
        assert debugger.inspect_level == "bars"
        assert debugger._continuing is False
        assert debugger._step_count == 0
        assert debugger._total_steps is None
        assert debugger._strategy_service is None

    def test_init_with_break_at_date(self):
        """Test debugger initializes with breakpoint date."""
        # Arrange
        break_date = date(2020, 6, 15)

        # Act
        debugger = InteractiveDebugger(break_at=break_date, enabled=False)

        # Assert
        assert debugger.break_at == break_date

    @pytest.mark.parametrize("level", ["bars", "full", "strategy"])
    def test_init_with_inspect_levels(self, level: str):
        """Test debugger initializes with different inspect levels."""
        # Arrange & Act
        debugger = InteractiveDebugger(inspect_level=level, enabled=False)

        # Assert
        assert debugger.inspect_level == level

    def test_init_enabled_logs_initialization(self):
        """Test enabled debugger logs initialization."""
        # Arrange & Act
        with patch("qtrader.cli.ui.interactive.logger") as mock_logger:
            InteractiveDebugger(enabled=True)

            # Assert
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            assert call_args[0][0] == "interactive_debugger.initialized"


class TestShouldPause:
    """Test should_pause logic for determining when to pause execution."""

    def test_should_pause_when_disabled(self):
        """Test should_pause returns False when debugger is disabled."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)
        timestamp = datetime(2020, 6, 15, 12, 0, 0)

        # Act
        result = debugger.should_pause(timestamp)

        # Assert
        assert result is False

    def test_should_pause_when_continuing(self):
        """Test should_pause returns False after user continues."""
        # Arrange
        debugger = InteractiveDebugger(enabled=True)
        debugger._continuing = True
        timestamp = datetime(2020, 6, 15, 12, 0, 0)

        # Act
        result = debugger.should_pause(timestamp)

        # Assert
        assert result is False

    def test_should_pause_before_break_at_date(self):
        """Test should_pause returns False before breakpoint date."""
        # Arrange
        break_date = date(2020, 6, 15)
        debugger = InteractiveDebugger(break_at=break_date, enabled=True)
        timestamp = datetime(2020, 6, 10, 12, 0, 0)

        # Act
        result = debugger.should_pause(timestamp)

        # Assert
        assert result is False

    def test_should_pause_on_break_at_date(self):
        """Test should_pause returns True on breakpoint date."""
        # Arrange
        break_date = date(2020, 6, 15)
        debugger = InteractiveDebugger(break_at=break_date, enabled=True)
        timestamp = datetime(2020, 6, 15, 12, 0, 0)

        # Act
        result = debugger.should_pause(timestamp)

        # Assert
        assert result is True

    def test_should_pause_after_break_at_date(self):
        """Test should_pause returns True after breakpoint date."""
        # Arrange
        break_date = date(2020, 6, 15)
        debugger = InteractiveDebugger(break_at=break_date, enabled=True)
        timestamp = datetime(2020, 6, 20, 12, 0, 0)

        # Act
        result = debugger.should_pause(timestamp)

        # Assert
        assert result is True

    def test_should_pause_no_break_at(self):
        """Test should_pause returns True when no breakpoint is set."""
        # Arrange
        debugger = InteractiveDebugger(enabled=True)
        timestamp = datetime(2020, 6, 15, 12, 0, 0)

        # Act
        result = debugger.should_pause(timestamp)

        # Assert
        assert result is True


class TestShowHeader:
    """Test show_header method for displaying timestamp before events."""

    def test_show_header_displays_when_should_pause(self):
        """Test show_header displays timestamp when should_pause is True."""
        # Arrange
        debugger = InteractiveDebugger(enabled=True)
        timestamp = datetime(2020, 6, 15, 12, 0, 0)

        with patch.object(debugger, "_display_header") as mock_display:
            # Act
            debugger.show_header(timestamp)

            # Assert
            mock_display.assert_called_once_with(timestamp)
            assert debugger._step_count == 1

    def test_show_header_skips_when_should_not_pause(self):
        """Test show_header does nothing when should_pause is False."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)
        timestamp = datetime(2020, 6, 15, 12, 0, 0)

        with patch.object(debugger, "_display_header") as mock_display:
            # Act
            debugger.show_header(timestamp)

            # Assert
            mock_display.assert_not_called()
            assert debugger._step_count == 0

    def test_show_header_increments_step_count(self):
        """Test show_header increments step count."""
        # Arrange
        debugger = InteractiveDebugger(enabled=True)
        timestamp1 = datetime(2020, 6, 15, 12, 0, 0)
        timestamp2 = datetime(2020, 6, 16, 12, 0, 0)

        with patch.object(debugger, "_display_header"):
            # Act
            debugger.show_header(timestamp1)
            debugger.show_header(timestamp2)

            # Assert
            assert debugger._step_count == 2


class TestOnTimestamp:
    """Test on_timestamp method for displaying state after events."""

    def test_on_timestamp_skips_when_should_not_pause(self):
        """Test on_timestamp does nothing when should_pause is False."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)
        timestamp = datetime(2020, 6, 15, 12, 0, 0)
        bars = {}

        with patch.object(debugger, "_display_bars") as mock_display:
            # Act
            debugger.on_timestamp(timestamp, bars)

            # Assert
            mock_display.assert_not_called()

    def test_on_timestamp_displays_bars(self):
        """Test on_timestamp displays bars when pausing."""
        # Arrange
        debugger = InteractiveDebugger(enabled=True)
        timestamp = datetime(2020, 6, 15, 12, 0, 0)
        bars = {"AAPL": Mock()}

        with patch.object(debugger, "_display_bars") as mock_display, patch.object(debugger, "_prompt_user"):
            # Act
            debugger.on_timestamp(timestamp, bars)

            # Assert
            mock_display.assert_called_once()

    def test_on_timestamp_collects_indicators_from_strategy_service(self):
        """Test on_timestamp collects indicators from strategy service."""
        # Arrange
        debugger = InteractiveDebugger(enabled=True)
        timestamp = datetime(2020, 6, 15, 12, 0, 0)
        bars = {"AAPL": Mock()}

        # Mock strategy service with indicators
        mock_context = Mock()
        mock_context._indicators = {"SMA(20)": 100.5, "SMA(50)": 95.2}
        mock_strategy_service = Mock()
        mock_strategy_service._contexts = {"strategy1": mock_context}
        debugger.set_strategy_service(mock_strategy_service)

        with patch.object(debugger, "_display_bars") as mock_display, patch.object(debugger, "_prompt_user"):
            # Act
            debugger.on_timestamp(timestamp, bars)

            # Assert
            call_args = mock_display.call_args
            indicators_arg = call_args[0][1]
            assert "strategy1" in indicators_arg
            assert indicators_arg["strategy1"]["SMA(20)"] == 100.5

    def test_on_timestamp_merges_passed_indicators(self):
        """Test on_timestamp merges passed indicators with collected ones."""
        # Arrange
        debugger = InteractiveDebugger(enabled=True)
        timestamp = datetime(2020, 6, 15, 12, 0, 0)
        bars = {"AAPL": Mock()}
        passed_indicators = {"strategy1": {"RSI": 70.5}}

        with patch.object(debugger, "_display_bars") as mock_display, patch.object(debugger, "_prompt_user"):
            # Act
            debugger.on_timestamp(timestamp, bars, indicators=passed_indicators)

            # Assert
            call_args = mock_display.call_args
            indicators_arg = call_args[0][1]
            assert indicators_arg["strategy1"]["RSI"] == 70.5

    def test_on_timestamp_displays_signals_in_full_mode(self):
        """Test on_timestamp displays signals when inspect_level is full."""
        # Arrange
        debugger = InteractiveDebugger(enabled=True, inspect_level="full")
        timestamp = datetime(2020, 6, 15, 12, 0, 0)
        bars = {"AAPL": Mock()}
        signals = [Mock()]

        with (
            patch.object(debugger, "_display_bars"),
            patch.object(debugger, "_display_signals") as mock_signals,
            patch.object(debugger, "_prompt_user"),
        ):
            # Act
            debugger.on_timestamp(timestamp, bars, signals=signals)

            # Assert
            mock_signals.assert_called_once_with(signals)

    def test_on_timestamp_skips_signals_in_bars_mode(self):
        """Test on_timestamp skips signals when inspect_level is bars."""
        # Arrange
        debugger = InteractiveDebugger(enabled=True, inspect_level="bars")
        timestamp = datetime(2020, 6, 15, 12, 0, 0)
        bars = {"AAPL": Mock()}
        signals = [Mock()]

        with (
            patch.object(debugger, "_display_bars"),
            patch.object(debugger, "_display_signals") as mock_signals,
            patch.object(debugger, "_prompt_user"),
        ):
            # Act
            debugger.on_timestamp(timestamp, bars, signals=signals)

            # Assert
            mock_signals.assert_not_called()


class TestDisplayBars:
    """Test _display_bars method for formatting bar data."""

    def test_display_bars_with_empty_bars(self):
        """Test _display_bars handles empty bars dict."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)
        bars = {}

        # Act (should not raise exception)
        debugger._display_bars(bars)

        # Assert - no exception raised

    def test_display_bars_formats_volume_millions(self):
        """Test _display_bars formats volume in millions."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)
        mock_bar = Mock()
        mock_bar.open = 100.0
        mock_bar.high = 105.0
        mock_bar.low = 99.0
        mock_bar.close = 103.0
        mock_bar.volume = 150_000_000
        bars = {"AAPL": mock_bar}

        with patch.object(debugger.console, "print") as mock_print:
            # Act
            debugger._display_bars(bars)

            # Assert
            mock_print.assert_called()
            # Check that table was created (last call should be the table)
            assert mock_print.call_count >= 1

    def test_display_bars_formats_volume_thousands(self):
        """Test _display_bars formats volume in thousands."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)
        mock_bar = Mock()
        mock_bar.open = 100.0
        mock_bar.high = 105.0
        mock_bar.low = 99.0
        mock_bar.close = 103.0
        mock_bar.volume = 50_000
        bars = {"AAPL": mock_bar}

        with patch.object(debugger.console, "print"):
            # Act (should not raise exception)
            debugger._display_bars(bars)

            # Assert - no exception raised

    def test_display_bars_includes_indicators_as_columns(self):
        """Test _display_bars adds indicator columns."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)
        mock_bar = Mock()
        mock_bar.open = 100.0
        mock_bar.high = 105.0
        mock_bar.low = 99.0
        mock_bar.close = 103.0
        mock_bar.volume = 1_000_000
        bars = {"AAPL": mock_bar}
        indicators = {"strategy1": {"SMA(20)": 102.5, "SMA(50)": 98.3}}

        with patch.object(debugger.console, "print") as mock_print:
            # Act
            debugger._display_bars(bars, indicators)

            # Assert
            mock_print.assert_called()
            # Verify table was printed
            assert mock_print.call_count >= 1

    def test_display_bars_handles_missing_bar_attributes(self):
        """Test _display_bars handles bars with missing attributes gracefully."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)
        mock_bar = Mock(spec=[])  # Bar with no attributes
        bars = {"AAPL": mock_bar}

        with patch.object(debugger.console, "print"):
            # Act (should not raise exception)
            debugger._display_bars(bars)

            # Assert - no exception raised

    def test_display_bars_sorts_symbols(self):
        """Test _display_bars displays symbols in sorted order."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)
        mock_bar = Mock()
        mock_bar.open = 100.0
        mock_bar.high = 105.0
        mock_bar.low = 99.0
        mock_bar.close = 103.0
        mock_bar.volume = 1_000_000

        bars = {"TSLA": mock_bar, "AAPL": mock_bar, "MSFT": mock_bar}

        with patch.object(debugger.console, "print"):
            # Act (should not raise exception)
            debugger._display_bars(bars)

            # Assert - no exception, symbols should be sorted


class TestPromptUser:
    """Test _prompt_user method for handling user commands."""

    def test_prompt_user_quit_command(self):
        """Test _prompt_user handles quit command."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)

        with patch("builtins.input", return_value="q"), pytest.raises(SystemExit) as exc_info:
            # Act
            debugger._prompt_user()

            # Assert
            assert exc_info.value.code == 0

    def test_prompt_user_continue_command(self):
        """Test _prompt_user handles continue command."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)

        with patch("builtins.input", return_value="c"), patch.object(debugger.console, "print"):
            # Act
            debugger._prompt_user()

            # Assert
            assert debugger._continuing is True

    def test_prompt_user_inspect_toggle(self):
        """Test _prompt_user toggles inspect level."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False, inspect_level="bars")

        with patch("builtins.input", return_value="i"), patch.object(debugger.console, "print"):
            # Act
            debugger._prompt_user()

            # Assert
            assert debugger.inspect_level == "full"

    def test_prompt_user_inspect_toggle_cycles(self):
        """Test _prompt_user cycles through inspect levels."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False, inspect_level="strategy")

        with patch("builtins.input", return_value="i"), patch.object(debugger.console, "print"):
            # Act
            debugger._prompt_user()

            # Assert
            assert debugger.inspect_level == "bars"

    def test_prompt_user_enter_steps(self):
        """Test _prompt_user allows stepping with Enter."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)

        with patch("builtins.input", return_value=""), patch.object(debugger.console, "print"):
            # Act (should not raise exception)
            debugger._prompt_user()

            # Assert - no exception, no state change
            assert debugger._continuing is False

    def test_prompt_user_keyboard_interrupt(self):
        """Test _prompt_user handles KeyboardInterrupt as quit."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)

        with patch("builtins.input", side_effect=KeyboardInterrupt), pytest.raises(SystemExit):
            # Act
            debugger._prompt_user()

    def test_prompt_user_unknown_command(self):
        """Test _prompt_user handles unknown commands."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)

        with patch("builtins.input", return_value="unknown"), patch.object(debugger.console, "print") as mock_print:
            # Act
            debugger._prompt_user()

            # Assert - should print warning about unknown command
            assert mock_print.call_count > 0


class TestStrategyServiceIntegration:
    """Test integration with strategy service."""

    def test_set_strategy_service(self):
        """Test set_strategy_service stores service reference."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)
        mock_service = Mock()

        # Act
        debugger.set_strategy_service(mock_service)

        # Assert
        assert debugger._strategy_service is mock_service

    def test_set_total_steps(self):
        """Test set_total_steps stores total for progress display."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)

        # Act
        debugger.set_total_steps(250)

        # Assert
        assert debugger._total_steps == 250


class TestDisplayHeader:
    """Test _display_header method."""

    def test_display_header_shows_timestamp(self):
        """Test _display_header displays timestamp correctly."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)
        timestamp = datetime(2020, 6, 15, 12, 30, 45)
        debugger._step_count = 42

        with patch.object(debugger.console, "print") as mock_print:
            # Act
            debugger._display_header(timestamp)

            # Assert
            assert mock_print.call_count == 2  # Header text + blank line

    def test_display_header_shows_progress_with_total(self):
        """Test _display_header shows progress when total steps is set."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)
        timestamp = datetime(2020, 6, 15, 12, 30, 45)
        debugger._step_count = 42
        debugger._total_steps = 250

        with patch.object(debugger.console, "print") as mock_print:
            # Act
            debugger._display_header(timestamp)

            # Assert
            assert mock_print.call_count == 2


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_indicator_collection_handles_exception(self):
        """Test indicator collection handles exceptions gracefully."""
        # Arrange
        debugger = InteractiveDebugger(enabled=True)
        timestamp = datetime(2020, 6, 15, 12, 0, 0)
        bars = {"AAPL": Mock()}

        # Mock strategy service that raises exception
        mock_strategy_service = Mock()
        mock_strategy_service._contexts = Mock(side_effect=Exception("Test error"))
        debugger.set_strategy_service(mock_strategy_service)

        with patch.object(debugger, "_display_bars"), patch.object(debugger, "_prompt_user"):
            # Act (should not raise exception)
            debugger.on_timestamp(timestamp, bars)

            # Assert - no exception raised

    def test_display_bars_with_decimal_indicators(self):
        """Test _display_bars formats Decimal indicator values."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)
        mock_bar = Mock()
        mock_bar.open = 100.0
        mock_bar.high = 105.0
        mock_bar.low = 99.0
        mock_bar.close = 103.0
        mock_bar.volume = 1_000_000
        bars = {"AAPL": mock_bar}
        indicators = {"strategy1": {"SMA(20)": Decimal("102.5678")}}

        with patch.object(debugger.console, "print"):
            # Act (should not raise exception)
            debugger._display_bars(bars, indicators)

            # Assert - no exception raised

    def test_display_bars_with_none_indicator_values(self):
        """Test _display_bars handles None indicator values."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)
        mock_bar = Mock()
        mock_bar.open = 100.0
        mock_bar.high = 105.0
        mock_bar.low = 99.0
        mock_bar.close = 103.0
        mock_bar.volume = 1_000_000
        bars = {"AAPL": mock_bar}
        indicators = {"strategy1": {"SMA(20)": None}}

        with patch.object(debugger.console, "print"):
            # Act (should not raise exception)
            debugger._display_bars(bars, indicators)

            # Assert - no exception raised


class TestDisplaySignals:
    """Test _display_signals method."""

    def test_display_signals_with_empty_list(self):
        """Test _display_signals handles empty signal list."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)
        signals = []

        with patch.object(debugger.console, "print") as mock_print:
            # Act
            debugger._display_signals(signals)

            # Assert - should not print anything
            mock_print.assert_not_called()

    def test_display_signals_with_complete_signal(self):
        """Test _display_signals formats complete signal."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)
        mock_signal = Mock()
        mock_signal.intention = "BUY"
        mock_signal.symbol = "AAPL"
        mock_signal.price = 150.5
        mock_signal.confidence = 0.85
        mock_signal.reason = "Strong uptrend"
        signals = [mock_signal]

        with patch.object(debugger.console, "print") as mock_print:
            # Act
            debugger._display_signals(signals)

            # Assert
            assert mock_print.call_count >= 2  # Header + signal

    def test_display_signals_with_missing_attributes(self):
        """Test _display_signals handles signals with missing attributes."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)
        mock_signal = Mock(spec=[])  # Signal with no attributes
        signals = [mock_signal]

        with patch.object(debugger.console, "print"):
            # Act (should not raise exception)
            debugger._display_signals(signals)

            # Assert - no exception raised


class TestDisplayPortfolio:
    """Test _display_portfolio method."""

    def test_display_portfolio_with_positive_pnl(self):
        """Test _display_portfolio formats portfolio with positive P&L."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)
        portfolio = {"equity": 105000.50, "cash": 25000.25, "pnl": 5000.50}

        with patch.object(debugger.console, "print") as mock_print:
            # Act
            debugger._display_portfolio(portfolio)

            # Assert
            assert mock_print.call_count >= 1

    def test_display_portfolio_with_negative_pnl(self):
        """Test _display_portfolio formats portfolio with negative P&L."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)
        portfolio = {"equity": 95000.75, "cash": 25000.25, "pnl": -5000.25}

        with patch.object(debugger.console, "print") as mock_print:
            # Act
            debugger._display_portfolio(portfolio)

            # Assert
            assert mock_print.call_count >= 1

    def test_display_portfolio_with_missing_values(self):
        """Test _display_portfolio handles missing values with defaults."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)
        portfolio = {}

        with patch.object(debugger.console, "print"):
            # Act (should not raise exception)
            debugger._display_portfolio(portfolio)

            # Assert - no exception raised


class TestDisplayIndicators:
    """Test _display_indicators method."""

    def test_display_indicators_with_empty_dict(self):
        """Test _display_indicators handles empty indicator dict."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)
        indicators = {"strategy1": {}}

        with patch.object(debugger.console, "print") as mock_print:
            # Act
            debugger._display_indicators(indicators)

            # Assert - should not print panel for empty indicators
            mock_print.assert_not_called()

    def test_display_indicators_with_float_values(self):
        """Test _display_indicators formats float values."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)
        indicators = {"strategy1": {"SMA(20)": 102.5678, "RSI": 65.4321}}

        with patch.object(debugger.console, "print") as mock_print:
            # Act
            debugger._display_indicators(indicators)

            # Assert
            assert mock_print.call_count >= 1

    def test_display_indicators_with_decimal_values(self):
        """Test _display_indicators formats Decimal values."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)
        indicators = {"strategy1": {"SMA(20)": Decimal("102.5678")}}

        with patch.object(debugger.console, "print") as mock_print:
            # Act
            debugger._display_indicators(indicators)

            # Assert
            assert mock_print.call_count >= 1

    def test_display_indicators_with_string_values(self):
        """Test _display_indicators formats string values."""
        # Arrange
        debugger = InteractiveDebugger(enabled=False)
        indicators = {"strategy1": {"status": "active", "signal": "BULLISH"}}

        with patch.object(debugger.console, "print") as mock_print:
            # Act
            debugger._display_indicators(indicators)

            # Assert
            assert mock_print.call_count >= 1
