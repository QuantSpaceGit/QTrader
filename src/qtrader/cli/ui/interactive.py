"""
Interactive Debugger for Strategy Development.

Provides step-by-step execution control for backtests, allowing developers
to inspect state at each timestamp during strategy development and debugging.

Features:
- Pause at each timestamp (or specific dates)
- Display bars, indicators, signals, portfolio state
- Interactive commands: step, continue, quit, inspect
- Rich console formatting with color-coded output

Usage:
    debugger = InteractiveDebugger(
        break_at=date(2020, 6, 15),
        inspect_level="full"
    )

    # In DataService.stream_universe():
    debugger.on_timestamp(timestamp, bars, signals, portfolio)
"""

import sys
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Optional

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

logger = structlog.get_logger(__name__)


class InteractiveDebugger:
    """
    Interactive debugger for step-by-step backtest execution.

    Pauses at each timestamp (or from a breakpoint date) and displays:
    - Current timestamp and progress
    - Price bars for all symbols
    - Strategy indicators (if tracked)
    - Emitted signals
    - Portfolio state (if available)

    User can then:
    - Press Enter to step to next timestamp
    - Type 'c' + Enter to continue without pausing
    - Type 'q' + Enter to quit
    - Type 'i' + Enter to inspect detailed state
    """

    def __init__(
        self,
        break_at: Optional[date] = None,
        inspect_level: str = "bars",
        enabled: bool = True,
    ):
        """
        Initialize interactive debugger.

        Args:
            break_at: Optional date to start pausing (None = pause from start)
            inspect_level: Level of detail ('bars', 'full', 'strategy')
            enabled: Whether debugger is active (False = no-op)
        """
        self.break_at = break_at
        self.inspect_level = inspect_level
        self.enabled = enabled
        self._continuing = False  # User typed 'c' to continue
        self._step_count = 0
        self._total_steps: Optional[int] = None
        self._strategy_service: Optional[Any] = None  # Will be set by engine
        self.console = Console()

        if self.enabled:
            logger.info(
                "interactive_debugger.initialized",
                break_at=str(break_at) if break_at else "start",
                inspect_level=inspect_level,
            )

    def set_strategy_service(self, strategy_service: Any) -> None:
        """Set strategy service for collecting indicators."""
        self._strategy_service = strategy_service

    def should_pause(self, timestamp: datetime) -> bool:
        """
        Determine if we should pause at this timestamp.

        Args:
            timestamp: Current simulation timestamp

        Returns:
            True if should pause and wait for user input
        """
        if not self.enabled:
            return False

        if self._continuing:
            return False

        if self.break_at is not None:
            ts_date = timestamp.date()
            if ts_date < self.break_at:
                return False

        return True

    def show_header(self, timestamp: datetime) -> None:
        """
        Show timestamp header before events are published.

        This is called BEFORE events are published to ensure the timestamp
        appears first in the console output.

        Args:
            timestamp: Current simulation timestamp
        """
        if not self.should_pause(timestamp):
            return

        self._step_count += 1
        self._display_header(timestamp)

    def on_timestamp(
        self,
        timestamp: datetime,
        bars: dict[str, Any],
        indicators: Optional[dict[str, dict[str, Any]]] = None,
        signals: Optional[list[Any]] = None,
        portfolio: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Called at each timestamp during backtest (after events are published).

        Displays state and waits for user input if should_pause() is True.

        Args:
            timestamp: Current simulation timestamp
            bars: Dict of symbol -> bar data
            indicators: Optional dict of strategy_id -> indicator values
            signals: Optional list of signals emitted at this timestamp
            portfolio: Optional portfolio state snapshot
        """
        if not self.should_pause(timestamp):
            return

        # Collect indicators from strategy service if available
        collected_indicators: dict[str, dict[str, Any]] = {}
        if self._strategy_service is not None:
            try:
                if hasattr(self._strategy_service, "_contexts"):
                    for strategy_id, context in self._strategy_service._contexts.items():
                        if hasattr(context, "_indicators") and context._indicators:
                            collected_indicators[strategy_id] = dict(context._indicators)
            except Exception:
                pass  # Silently ignore if we can't collect indicators

        # Merge with any indicators passed in
        if indicators:
            for strategy_id, ind_dict in indicators.items():
                if strategy_id not in collected_indicators:
                    collected_indicators[strategy_id] = {}
                collected_indicators[strategy_id].update(ind_dict)

        # Use collected indicators if we got any
        final_indicators = collected_indicators if collected_indicators else indicators

        # Display bars with indicators in unified table
        self._display_bars(bars, final_indicators)

        # Keep separate indicator display only for 'full' inspection mode
        # if indicators and self.inspect_level == "full":
        #     self._display_indicators(indicators)

        # Display signals if available
        if signals and self.inspect_level in ("full", "strategy"):
            self._display_signals(signals)

        # Display portfolio if available
        if portfolio and self.inspect_level == "full":
            self._display_portfolio(portfolio)

        # Wait for user input
        self._prompt_user()

    def _display_header(self, timestamp: datetime) -> None:
        """Display timestamp header with progress."""
        progress_str = f"Step {self._step_count}"
        if self._total_steps:
            progress_str += f"/{self._total_steps}"

        header_text = Text()
        header_text.append("📅 Timestamp: ", style="bold cyan")
        header_text.append(timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"), style="bold yellow")
        header_text.append(f"  ({progress_str})", style="dim")

        self.console.print(header_text)
        self.console.print()

    def _display_bars(self, bars: dict[str, Any], indicators: Optional[dict[str, dict[str, Any]]] = None) -> None:
        """Display price bars and indicators for all symbols in unified table."""
        if not bars:
            return

        # Add blank line before table for visual separation
        self.console.print()

        # Collect all unique indicator names across all strategies
        indicator_names_set: set[str] = set()
        if indicators:
            for strategy_indicators in indicators.values():
                indicator_names_set.update(strategy_indicators.keys())
        indicator_names: list[str] = sorted(indicator_names_set)

        table = Table(title=f"📊 Bars ({len(bars)} symbols)", title_style="bold green")
        table.add_column("Symbol", style="cyan", no_wrap=True)
        table.add_column("Open", justify="right", style="white")
        table.add_column("High", justify="right", style="green")
        table.add_column("Low", justify="right", style="red")
        table.add_column("Close", justify="right", style="yellow bold")
        table.add_column("Volume", justify="right", style="blue")

        # Add columns for each indicator
        for ind_name in indicator_names:
            table.add_column(ind_name, justify="right", style="magenta")

        for symbol in sorted(bars.keys()):
            bar = bars[symbol]

            # Format volume with abbreviations (M, K)
            volume = bar.volume if hasattr(bar, "volume") else 0
            if volume >= 1_000_000:
                vol_str = f"{volume / 1_000_000:.1f}M"
            elif volume >= 1_000:
                vol_str = f"{volume / 1_000:.1f}K"
            else:
                vol_str = str(volume)

            # Build row with bar data
            row = [
                symbol,
                f"${bar.open:.2f}" if hasattr(bar, "open") else "-",
                f"${bar.high:.2f}" if hasattr(bar, "high") else "-",
                f"${bar.low:.2f}" if hasattr(bar, "low") else "-",
                f"${bar.close:.2f}" if hasattr(bar, "close") else "-",
                vol_str,
            ]

            # Add indicator values for this symbol
            if indicators:
                for ind_name in indicator_names:
                    # Look for this indicator across all strategies
                    value = None
                    for strategy_indicators in indicators.values():
                        if ind_name in strategy_indicators:
                            value = strategy_indicators[ind_name]
                            break

                    # Format the value
                    if value is None:
                        row.append("-")
                    elif isinstance(value, (Decimal, float)):
                        row.append(f"{float(value):.2f}")
                    else:
                        row.append(str(value))

            table.add_row(*row)

        self.console.print(table)

    def _display_indicators(self, indicators: dict[str, dict[str, Any]]) -> None:
        """Display strategy indicators."""
        for strategy_id, indicator_values in indicators.items():
            if not indicator_values:
                continue

            self.console.print()
            panel_content = Text()

            for name, value in sorted(indicator_values.items()):
                if isinstance(value, Decimal):
                    value_str = f"{value:.4f}"
                elif isinstance(value, float):
                    value_str = f"{value:.4f}"
                else:
                    value_str = str(value)

                panel_content.append(f"  {name}: ", style="dim")
                panel_content.append(value_str, style="bold cyan")
                panel_content.append("\n")

            self.console.print(
                Panel(
                    panel_content,
                    title=f"📈 Strategy: {strategy_id}",
                    title_align="left",
                    border_style="green",
                )
            )

    def _display_signals(self, signals: list[Any]) -> None:
        """Display emitted signals."""
        if not signals:
            return

        self.console.print()
        self.console.print(f"[bold magenta]💡 Signals ({len(signals)}):[/bold magenta]")

        for signal in signals:
            intention = signal.intention if hasattr(signal, "intention") else "UNKNOWN"
            symbol = signal.symbol if hasattr(signal, "symbol") else "?"
            price = signal.price if hasattr(signal, "price") else 0
            confidence = signal.confidence if hasattr(signal, "confidence") else 0
            reason = signal.reason if hasattr(signal, "reason") else ""

            signal_text = Text()
            signal_text.append("  → ", style="dim")
            signal_text.append(str(intention), style="bold yellow")
            signal_text.append(f" {symbol}", style="cyan")
            signal_text.append(f" @ ${price:.2f}", style="white")
            signal_text.append(f" (confidence: {confidence:.2f})", style="dim")

            if reason:
                signal_text.append(f"\n    Reason: {reason}", style="italic dim")

            self.console.print(signal_text)

    def _display_portfolio(self, portfolio: dict[str, Any]) -> None:
        """Display portfolio state."""
        self.console.print()

        equity = portfolio.get("equity", 0)
        cash = portfolio.get("cash", 0)
        pnl = portfolio.get("pnl", 0)

        portfolio_text = Text()
        portfolio_text.append("💼 Portfolio:  ", style="bold blue")
        portfolio_text.append(f"Equity: ${equity:,.2f}", style="white")
        portfolio_text.append("  |  ", style="dim")
        portfolio_text.append(f"Cash: ${cash:,.2f}", style="white")
        portfolio_text.append("  |  ", style="dim")

        pnl_style = "green" if pnl >= 0 else "red"
        pnl_sign = "+" if pnl >= 0 else ""
        portfolio_text.append(f"P&L: {pnl_sign}${pnl:,.2f}", style=pnl_style)

        self.console.print(portfolio_text)

    def _prompt_user(self) -> None:
        """Prompt user for next action and handle input."""
        self.console.print()
        self.console.print("[dim]Commands: [Enter]=next [c]=continue [q]=quit [i]=inspect[/dim]")
        self.console.rule(style="cyan")

        try:
            user_input = input().strip().lower()
        except (KeyboardInterrupt, EOFError):
            user_input = "q"

        if user_input == "q":
            self.console.print("[yellow]⚠️  User quit - stopping backtest[/yellow]")
            logger.info("interactive_debugger.user_quit", step=self._step_count)
            sys.exit(0)
        elif user_input == "c":
            self.console.print("[green]▶ Continuing without pauses...[/green]")
            self._continuing = True
            logger.info("interactive_debugger.user_continue", step=self._step_count)
        elif user_input == "i":
            # Toggle inspect level
            levels = ["bars", "full", "strategy"]
            current_idx = levels.index(self.inspect_level)
            self.inspect_level = levels[(current_idx + 1) % len(levels)]
            self.console.print(f"[cyan]ℹ️  Inspect level: {self.inspect_level}[/cyan]")
        elif user_input == "" or user_input == "n":
            # Step to next (Enter or 'n')
            pass
        else:
            self.console.print(f"[yellow]Unknown command: {user_input}[/yellow]")

    def set_total_steps(self, total: int) -> None:
        """Set total number of timestamps for progress display."""
        self._total_steps = total
