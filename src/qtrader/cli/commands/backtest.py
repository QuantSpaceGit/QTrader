"""Backtest execution command."""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from qtrader.cli.ui.interactive import InteractiveDebugger
from qtrader.engine.config import load_backtest_config
from qtrader.engine.engine import BacktestEngine
from qtrader.engine.experiment import ExperimentMetadata, ExperimentResolver, RunMetadata
from qtrader.system.config import get_system_config, reload_system_config

console = Console()


@click.command("backtest")
@click.argument(
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--file",
    "-f",
    "config_file_deprecated",
    type=click.Path(exists=True, path_type=Path),
    help="[Deprecated] Use positional argument instead",
    hidden=True,
)
@click.option(
    "--silent",
    "-s",
    is_flag=True,
    help="Silent mode: no event display (fastest execution)",
)
@click.option(
    "--replay-speed",
    "-r",
    type=float,
    help="Override replay speed (-1=silent, 0=instant, >0=delay in seconds)",
)
@click.option(
    "--start-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Override start date (YYYY-MM-DD)",
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Override end date (YYYY-MM-DD)",
)
@click.option(
    "--log-level",
    "-l",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Set logging level (DEBUG shows all initialization details)",
)
@click.option(
    "--html-report/--no-html-report",
    default=True,
    help="Generate interactive HTML report (default: enabled)",
)
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    help="Interactive mode: pause at each timestamp for debugging",
)
@click.option(
    "--break-at",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Start interactive mode from specific date (YYYY-MM-DD)",
)
@click.option(
    "--break-on",
    multiple=True,
    help="Pause only on specific events (e.g., 'signal', 'signal:BUY'). Can be repeated.",
)
@click.option(
    "--inspect",
    type=click.Choice(["bars", "full", "strategy"], case_sensitive=False),
    default="bars",
    help="Inspection level in interactive mode (bars=basic, full=everything, strategy=indicators only)",
)
def backtest_command(
    config_path: Path,
    config_file_deprecated: Optional[Path],
    silent: bool,
    replay_speed: Optional[float],
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    log_level: Optional[str],
    html_report: bool,
    interactive: bool,
    break_at: Optional[datetime],
    break_on: tuple[str, ...],
    inspect: str,
):
    """
    Run a backtest from experiment directory or configuration file.

    Supports two invocation patterns:
    1. Experiment directory: qtrader backtest experiments/my_strategy/
       - Automatically finds my_strategy.yaml inside
       - Creates timestamped run directory with full metadata
    2. Direct config file: qtrader backtest path/to/config.yaml
       - Traditional file-based invocation
       - Uses experiment structure if inside experiments/

    CLI options override config file values without modifying files.

    \b
    Examples:
        # Run experiment (preferred)
        qtrader backtest experiments/momentum_strategy

        # Run with config file
        qtrader backtest experiments/momentum_strategy/momentum_strategy.yaml

        # Silent mode (fastest execution)
        qtrader backtest experiments/momentum_strategy --silent

        # Override replay speed
        qtrader backtest experiments/momentum_strategy -r 0.5

        # Quick date range test
        qtrader backtest experiments/momentum_strategy \\
            --start-date 2020-01-01 --end-date 2020-03-31

        # Debug mode (show all initialization logs)
        qtrader backtest experiments/momentum_strategy -l debug

        # Interactive debugging (pause at each timestamp)
        qtrader backtest experiments/momentum_strategy --interactive

        # Interactive from specific date with full inspection
        qtrader backtest experiments/momentum_strategy --interactive \\
            --break-at 2020-06-15 --inspect full

        # Pause only on trading signals (event-triggered mode)
        qtrader backtest experiments/momentum_strategy --interactive \\
            --break-on signal

        # Pause only on BUY signals from specific date
        qtrader backtest experiments/momentum_strategy --interactive \\
            --break-at 2020-06-15 --break-on signal:BUY

    \b
    Output Structure:
        experiments/{experiment_id}/
            {experiment_id}.yaml           # Configuration
            runs/
                {timestamp}/               # This run's artifacts
                    run_manifest.json      # Metadata
                    config_snapshot.yaml   # Config copy
                    events.{backend}       # Event store
                    logs/                  # If file logging enabled
                latest -> {timestamp}/     # Symlink to latest run
    """
    try:
        # Header
        console.rule("[bold blue]QTrader Backtest[/bold blue]")
        console.print()

        # Handle deprecated --file option
        if config_file_deprecated:
            console.print("[yellow]Note: --file/-f is deprecated. Use positional argument instead.[/yellow]")
            config_path = config_file_deprecated

        # Resolve config file (supports both directory and file paths)
        console.print("[cyan]Resolving experiment configuration...[/cyan]")
        resolved_config_path = ExperimentResolver.resolve_config_path(config_path)
        experiment_dir = ExperimentResolver.get_experiment_dir(resolved_config_path)

        console.print(f"  Config: [dim]{resolved_config_path}[/dim]")
        console.print(f"  Experiment: [cyan]{experiment_dir.name}[/cyan]")
        console.print()

        # Validate experiment structure
        try:
            ExperimentResolver.validate_experiment_structure(experiment_dir, resolved_config_path)
        except ValueError as e:
            console.print(f"[yellow]Warning: {e}[/yellow]")

        # Load configuration
        console.print("[cyan]Loading configuration...[/cyan]")
        reload_system_config()
        config = load_backtest_config(resolved_config_path)
        system_config = get_system_config()

        # Create run directory and metadata
        run_id = ExperimentResolver.generate_run_id(system_config.output.run_id_format)
        run_dir = ExperimentResolver.create_run_dir(experiment_dir, run_id)

        console.print(f"  Run ID: [yellow]{run_id}[/yellow]")
        console.print(f"  Run directory: [dim]{run_dir}[/dim]")
        console.print()

        # Initialize run metadata
        run_metadata = RunMetadata(
            experiment_id=experiment_dir.name,
            run_id=run_id,
            started_at=datetime.now().isoformat(),
            config_sha256=ExperimentMetadata.compute_config_hash(resolved_config_path),
        )

        # Capture git info if enabled
        if system_config.output.capture_git_info:
            run_metadata.git = ExperimentMetadata.capture_git_info()
            if run_metadata.git and run_metadata.git.dirty:
                console.print("[yellow]⚠ Git repository has uncommitted changes[/yellow]")

        # Capture environment if enabled
        if system_config.output.capture_environment:
            run_metadata.environment = ExperimentMetadata.capture_environment()

        # Save config snapshot
        ExperimentMetadata.save_config_snapshot(resolved_config_path, run_dir)

        # Apply log level override if specified
        if log_level:
            from typing import Literal, cast

            from qtrader.system import LoggerFactory

            system_config = get_system_config()
            # Type cast since click already validated the choice
            level = cast(Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], log_level.upper())
            system_config.logging.level = level
            LoggerFactory.configure(system_config.logging.to_logger_config())

        # Apply CLI overrides
        if silent:
            config.replay_speed = -1.0
            config.display_events = None
        elif replay_speed is not None:
            config.replay_speed = replay_speed

        if start_date:
            config.start_date = start_date

        if end_date:
            config.end_date = end_date

        # Apply HTML report setting
        if config.reporting:
            config.reporting.write_html_report = html_report

        # Initialize interactive debugger if requested
        debugger = None
        if interactive or break_on:
            # Enable interactive mode if --break-on is used (even without --interactive)
            debugger = InteractiveDebugger(
                break_at=break_at.date() if break_at else None,
                break_on=list(break_on) if break_on else None,
                inspect_level=inspect.lower(),
                enabled=True,
            )
            mode = "event-triggered" if break_on else "step-through"
            console.print(f"[cyan]Interactive Mode:[/cyan] [yellow]{mode}[/yellow]")
            if break_at:
                console.print(f"  Break at: [yellow]{break_at.date()}[/yellow]")
            if break_on:
                console.print(f"  Break on: [magenta]{', '.join(break_on)}[/magenta]")
            console.print(f"  Inspect level: [yellow]{inspect}[/yellow]")
            console.print()

        # Display config summary
        console.print(f"  Backtest ID: [yellow]{config.backtest_id}[/yellow]")
        console.print(
            f"  Date Range: [yellow]{config.start_date.date()}[/yellow] to [yellow]{config.end_date.date()}[/yellow]"
        )
        console.print(f"  Universe: [magenta]{list(config.all_symbols)}[/magenta]")

        if config.replay_speed == -1.0:
            console.print("  Display: [dim]Silent mode (no events)[/dim]")
        elif config.replay_speed == 0:
            console.print(f"  Display: [yellow]{config.display_events}[/yellow] (instant)")
        else:
            console.print(f"  Display: [yellow]{config.display_events}[/yellow] ({config.replay_speed}s per event)")
        console.print()

        # Initialize engine with run directory for artifact output
        with console.status("[cyan]Initializing backtest engine...[/cyan]"):
            engine = BacktestEngine.from_config(config, results_dir=run_dir, debugger=debugger)

        # Run backtest
        try:
            result = engine.run()

            # Update run metadata with success
            run_metadata.status = "success"
            run_metadata.finished_at = datetime.now().isoformat()
            run_metadata.metrics = {
                "bars_processed": result.bars_processed,
                "duration_seconds": result.duration.total_seconds(),
            }
        except Exception as e:
            # Update run metadata with failure
            run_metadata.status = "failed"
            run_metadata.finished_at = datetime.now().isoformat()
            run_metadata.error = str(e)
            raise
        finally:
            # Always write metadata
            ExperimentMetadata.write_run_metadata(run_dir, run_metadata)
            ExperimentMetadata.create_latest_symlink(experiment_dir, run_id)

        console.print()
        console.print("[bold green]✓ Backtest completed successfully![/bold green]")
        console.print()

        # Display results
        console.rule("[bold green]RESULTS[/bold green]")
        console.print()
        console.print(f"[cyan]Experiment:[/cyan]      {experiment_dir.name}")
        console.print(f"[cyan]Run ID:[/cyan]          {run_id}")
        console.print(f"[cyan]Date Range:[/cyan]      {result.start_date} to {result.end_date}")
        console.print(f"[cyan]Bars Processed:[/cyan]  {result.bars_processed:,}")
        console.print(f"[cyan]Duration:[/cyan]        {result.duration}")
        console.print()

        # Experiment artifacts
        console.print(f"[cyan]Run Directory:[/cyan]   {run_dir}")
        console.print(f"[cyan]Metadata:[/cyan]        run_manifest.json")

        # Event store info
        backend_type = system_config.output.event_store.backend

        if backend_type == "memory":
            console.print("[cyan]Event Store:[/cyan]     memory (no files created)")
        elif hasattr(engine, "_results_dir") and engine._results_dir:
            event_store_filename = system_config.output.event_store.filename
            if "{backend}" in event_store_filename:
                extension_map = {"sqlite": "sqlite", "parquet": "parquet"}
                event_store_filename = event_store_filename.replace("{backend}", extension_map[backend_type])

            event_file = engine._results_dir / event_store_filename
            if event_file.exists():
                size_mb = os.path.getsize(event_file) / (1024 * 1024)
                console.print(f"[cyan]Event Store:[/cyan]     {event_file.name} ({size_mb:.2f} MB)")

        # Check for latest symlink
        latest_link = experiment_dir / "runs" / "latest"
        if latest_link.exists():
            console.print(f"[cyan]Latest Run:[/cyan]      runs/latest → {run_id}")

        console.print()
        console.rule()
        console.print()

        # Cleanup
        engine.shutdown()

        sys.exit(0)

    except Exception as e:
        console.print()
        console.print(f"[bold red]✗ Backtest failed:[/bold red] {e}")
        import traceback

        console.print()
        console.print("[dim]" + traceback.format_exc() + "[/dim]")
        sys.exit(1)
