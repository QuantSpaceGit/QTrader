"""
Interactive Debugger for Strategy Development.

Provides step-by-step execution control for backtests, allowing developers
to inspect state at each timestamp during strategy development and debugging.

Features:
- Pause at each timestamp (or specific dates)
- Pause on specific events (signals, with optional filters)
- Display bars, indicators, signals, portfolio state
- Interactive commands: step, continue, quit, inspect
- Rich console formatting with color-coded output

Usage:
    # Step-through from start
    debugger = InteractiveDebugger(enabled=True)

    # Step-through from specific date
    debugger = InteractiveDebugger(break_at=date(2020, 6, 15))

    # Pause only on signals
    debugger = InteractiveDebugger(break_on=["signal"])

    # Pause only on BUY signals from specific date
    debugger = InteractiveDebugger(
        break_at=date(2020, 6, 15),
        break_on=["signal:BUY"]
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

from qtrader.cli.ui.breakpoints import BreakpointContext, BreakpointRule, parse_breakpoints

logger = structlog.get_logger(__name__)


class InteractiveDebugger:
    """
    Interactive debugger for step-by-step backtest execution.

    Supports two modes:

    1. Step-through mode (default): Pauses at every timestamp, optionally
       from a specific date via --break-at.
    2. Event-triggered mode: Pauses only when specific events occur,
       configured via --break-on (e.g., "signal", "signal:BUY").

    At each pause, displays current timestamp, price bars, strategy indicators,
    emitted signals, and portfolio state.

    User commands: Enter (step), 'c' (continue), 'q' (quit), 'i' (toggle inspect).
    """

    def __init__(
        self,
        break_at: Optional[date] = None,
        break_on: Optional[list[str]] = None,
        inspect_level: str = "bars",
        enabled: bool = True,
    ):
        """
        Initialize interactive debugger.

        Args:
            break_at: Optional date to start debugging from (None = start immediately).
                     Acts as a "start condition" - before this date, no pausing occurs.
            break_on: Optional list of event breakpoints (e.g., ["signal", "signal:BUY"]).
                     If provided, only pauses when matching events occur.
                     If empty/None, pauses at every timestamp (step-through mode).
            inspect_level: Level of detail ('bars', 'full', 'strategy')
            enabled: Whether debugger is active (False = no-op for zero overhead)
        """
        self.break_at = break_at
        self.inspect_level = inspect_level
        self.enabled = enabled
        self._continuing = False  # User typed 'c' to continue
        self._step_count = 0
        self._total_steps: Optional[int] = None
        self._strategy_service: Optional[Any] = None  # Will be set by engine
        self._portfolio_service: Optional[Any] = None  # Will be set by engine
        self._event_bus: Optional[Any] = None  # Will be set by engine for signal subscription
        self._pending_signals: list[Any] = []  # Signals collected for current timestamp
        self.console = Console()

        # Parse breakpoint rules from strings
        self._breakpoints: list[BreakpointRule] = []
        if break_on:
            self._breakpoints = parse_breakpoints(break_on)

        # Determine mode
        self._event_triggered_mode = len(self._breakpoints) > 0

        if self.enabled:
            mode = "event-triggered" if self._event_triggered_mode else "step-through"
            breakpoint_descs = [bp.description() for bp in self._breakpoints]
            logger.info(
                "interactive_debugger.initialized",
                mode=mode,
                break_at=str(break_at) if break_at else "start",
                break_on=breakpoint_descs if breakpoint_descs else None,
                inspect_level=inspect_level,
            )

    @property
    def breakpoints(self) -> list[BreakpointRule]:
        """Get the list of configured breakpoint rules."""
        return self._breakpoints

    def set_strategy_service(self, strategy_service: Any) -> None:
        """Set strategy service for collecting indicators."""
        self._strategy_service = strategy_service

    def set_portfolio_service(self, portfolio_service: Any) -> None:
        """Set portfolio service for collecting portfolio state."""
        self._portfolio_service = portfolio_service

    def set_event_bus(self, event_bus: Any) -> None:
        """
        Set event bus for signal subscription.

        When set, the debugger subscribes to SignalEvent to collect signals
        for breakpoint evaluation. This is required for --break-on signal.

        Args:
            event_bus: EventBus instance to subscribe to
        """
        self._event_bus = event_bus
        if self._event_bus is not None and self._event_triggered_mode:
            # Subscribe to signals for event-triggered mode
            self._event_bus.subscribe("signal", self._on_signal_event)
            logger.debug("interactive_debugger.subscribed_to_signals")

    def _on_signal_event(self, event: Any) -> None:
        """
        Handle incoming signal events.

        Collects signals for the current timestamp so they can be evaluated
        against breakpoint rules in on_timestamp().

        Args:
            event: SignalEvent instance
        """
        self._pending_signals.append(event)

    def _check_start_condition(self, timestamp: datetime) -> bool:
        """
        Check if we've passed the start condition (break_at date).

        Args:
            timestamp: Current simulation timestamp

        Returns:
            True if we're past the start condition and can consider pausing
        """
        if self.break_at is not None:
            return timestamp.date() >= self.break_at
        return True  # No start condition, always eligible

    def should_pause_before_events(self, timestamp: datetime) -> bool:
        """
        Check if we should pause BEFORE events are published.

        This is only true in step-through mode (no event breakpoints).
        In event-triggered mode, we must wait to see if events match.

        Args:
            timestamp: Current simulation timestamp

        Returns:
            True if should show header and prepare to pause
        """
        if not self.enabled or self._continuing:
            return False

        if not self._check_start_condition(timestamp):
            return False

        # In event-triggered mode, wait until after events
        if self._event_triggered_mode:
            return False

        return True

    def should_pause(self, timestamp: datetime) -> bool:
        """
        Determine if we should pause at this timestamp (legacy compatibility).

        For step-through mode, this is equivalent to should_pause_before_events.
        For event-triggered mode, use should_pause_after_events with context.

        Args:
            timestamp: Current simulation timestamp

        Returns:
            True if should pause and wait for user input
        """
        return self.should_pause_before_events(timestamp)

    def should_pause_after_events(self, context: BreakpointContext) -> bool:
        """
        Check if we should pause AFTER events are published.

        Evaluates all breakpoint rules against the context to determine
        if any trigger a pause.

        Args:
            context: Context with timestamp, signals, bars, etc.

        Returns:
            True if any breakpoint triggers or in step-through mode
        """
        if not self.enabled or self._continuing:
            return False

        if not self._check_start_condition(context.timestamp):
            return False

        # In step-through mode, always pause (after start condition)
        if not self._event_triggered_mode:
            return True

        # In event-triggered mode, check if any breakpoint triggers
        for bp in self._breakpoints:
            if bp.should_trigger(context):
                logger.debug(
                    "interactive_debugger.breakpoint_triggered",
                    breakpoint=bp.description(),
                    timestamp=context.timestamp.isoformat(),
                )
                return True

        return False

    def show_header(self, timestamp: datetime) -> None:
        """
        Show timestamp header before events are published.

        Only called in step-through mode. In event-triggered mode,
        the header is shown in on_timestamp() after we know we should pause.

        Args:
            timestamp: Current simulation timestamp
        """
        if not self.should_pause_before_events(timestamp):
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

        In step-through mode, displays state and waits for user input.
        In event-triggered mode, checks breakpoints and only pauses if triggered.

        Args:
            timestamp: Current simulation timestamp
            bars: Dict of symbol -> bar data
            indicators: Optional dict of strategy_id -> indicator values
            signals: Optional list of signals emitted at this timestamp
            portfolio: Optional portfolio state snapshot
        """
        # Always clear pending signals to prevent accumulation, even when not pausing
        pending = self._pending_signals.copy()
        self._pending_signals.clear()

        if not self.enabled or self._continuing:
            return

        # Collect indicators from strategy service if available
        collected_indicators = self._collect_indicators()

        # Merge with any indicators passed in
        if indicators:
            for strategy_id, ind_dict in indicators.items():
                if strategy_id not in collected_indicators:
                    collected_indicators[strategy_id] = {}
                collected_indicators[strategy_id].update(ind_dict)

        final_indicators = collected_indicators if collected_indicators else indicators

        # Use signals passed in OR collected via EventBus subscription
        current_signals = signals if signals else pending

        # Collect portfolio state from service if not passed in
        final_portfolio = portfolio if portfolio else self._collect_portfolio(timestamp)

        # Build context for breakpoint evaluation
        context = BreakpointContext(
            timestamp=timestamp,
            signals=current_signals,
            bars=bars,
            indicators=final_indicators or {},
            portfolio=final_portfolio,
        )

        # Check if we should pause
        if not self.should_pause_after_events(context):
            return

        # In event-triggered mode, show header now (wasn't shown in show_header)
        if self._event_triggered_mode:
            self._step_count += 1
            self._display_header(timestamp)

        # Display state
        self._display_bars(bars, final_indicators)

        # Display signals if available (show in event-triggered mode regardless of inspect level)
        if current_signals and (self._event_triggered_mode or self.inspect_level in ("full", "strategy")):
            self._display_signals(current_signals)

        # Display portfolio if available (show in event-triggered mode or full inspect level)
        if final_portfolio and (self._event_triggered_mode or self.inspect_level == "full"):
            self._display_portfolio(final_portfolio)

        # Wait for user input
        self._prompt_user()

    def _collect_indicators(self) -> dict[str, dict[str, Any]]:
        """Collect indicators from strategy service contexts."""
        collected: dict[str, dict[str, Any]] = {}
        if self._strategy_service is not None:
            try:
                if hasattr(self._strategy_service, "_contexts"):
                    for strategy_id, context in self._strategy_service._contexts.items():
                        if hasattr(context, "_indicators") and context._indicators:
                            collected[strategy_id] = dict(context._indicators)
            except Exception:
                pass  # Silently ignore if we can't collect indicators
        return collected

    def _collect_portfolio(self, timestamp: datetime) -> Optional[dict[str, Any]]:
        """Collect portfolio state from portfolio service."""
        if self._portfolio_service is None:
            return None

        try:
            # Get basic portfolio metrics
            if hasattr(self._portfolio_service, "get_equity"):
                equity = self._portfolio_service.get_equity()
                cash = self._portfolio_service.get_cash()

                # Calculate P&L if we have initial equity
                # Try _initial_cash first (direct attribute), then config.initial_cash
                initial: float = 0.0
                if hasattr(self._portfolio_service, "_initial_cash"):
                    initial = float(self._portfolio_service._initial_cash)
                elif hasattr(self._portfolio_service, "config"):
                    config = self._portfolio_service.config
                    if hasattr(config, "initial_cash"):
                        initial = float(config.initial_cash)

                if initial > 0:
                    pnl = float(equity) - initial
                    pnl_pct = pnl / initial * 100
                else:
                    pnl = 0.0
                    pnl_pct = 0.0

                return {
                    "equity": float(equity),
                    "cash": float(cash),
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                }
        except Exception:
            pass  # Silently ignore if we can't collect portfolio
        return None

    def _display_header(self, timestamp: datetime) -> None:
        """Display timestamp header with progress."""
        progress_str = f"Step {self._step_count}"
        if self._total_steps:
            progress_str += f"/{self._total_steps}"

        header_text = Text()
        header_text.append("\n\n")
        header_text.append("📅 Timestamp: ", style="bold cyan")
        header_text.append(timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"), style="bold yellow")
        header_text.append(f"  ({progress_str})", style="dim")

        # Show triggered breakpoints in event-triggered mode
        if self._event_triggered_mode:
            bp_descs = ", ".join(bp.description() for bp in self._breakpoints)
            header_text.append(f"  [break-on: {bp_descs}]", style="dim magenta")

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
