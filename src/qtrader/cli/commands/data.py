"""Data management commands - thin CLI orchestration layer."""

import sys
import time
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from qtrader.cli.ui import (
    create_bar_table,
    create_cache_info_table,
    create_update_progress,
    create_update_summary_table,
)
from qtrader.cli.ui.formatters import add_bar_data, add_cache_info_row, add_update_result_row
from qtrader.services.data.adapters.resolver import DataSourceResolver
from qtrader.services.data.models import Instrument
from qtrader.services.data.update_service import UpdateService
from qtrader.utilities.yahoo_update import (
    BATCH_PAUSE,
    BATCH_SIZE,
    MIN_REQUEST_INTERVAL,
    REQUESTS_PER_MINUTE,
    discover_symbols,
    get_safe_end_date,
    get_yahoo_data_dir,
    is_market_closed,
    load_universe,
    update_symbol,
)


@click.group("data")
def data_group():
    """Data management commands - browse, fetch, cache, and update market data"""
    pass


@data_group.command("list")
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed information about each dataset",
)
def list_datasets(verbose: bool):
    """
    List all available datasets configured in data_sources.yaml.

    Displays dataset names, providers, adapters, and asset classes.
    Use --verbose for additional configuration details.

    Example:
        qtrader data list
        qtrader data list --verbose
    """
    console = Console()

    try:
        # Load resolver to access configured datasets
        resolver = DataSourceResolver()

        # Get list of all datasets
        datasets = resolver.list_sources()

        if not datasets:
            console.print("[yellow]No datasets configured in data_sources.yaml[/yellow]")
            return

        # Display summary
        console.print(f"\n[cyan]Found {len(datasets)} configured dataset(s)[/cyan]")
        console.print(f"[dim]Configuration file: {resolver.config_path}[/dim]\n")

        # Create table
        table = Table(title="Available Datasets", show_header=True, header_style="bold cyan")
        table.add_column("Dataset Name", style="green", no_wrap=True)
        table.add_column("Provider", style="cyan")
        table.add_column("Adapter", style="yellow")
        table.add_column("Asset Class", style="magenta")

        if verbose:
            table.add_column("Frequency", style="blue")
            table.add_column("Cache", style="white")

        # Add rows for each dataset
        for dataset_name in sorted(datasets):
            config = resolver.get_source_config(dataset_name)

            provider = config.get("provider", "N/A")
            adapter = config.get("adapter", "N/A")
            asset_class = config.get("asset_class", "N/A")

            if verbose:
                frequency = config.get("frequency", "N/A")
                cache_status = "✓" if config.get("cache_root") else "✗"
                table.add_row(
                    dataset_name,
                    provider,
                    adapter,
                    asset_class,
                    frequency,
                    cache_status,
                )
            else:
                table.add_row(dataset_name, provider, adapter, asset_class)

        console.print(table)
        console.print()

        # Show helpful tips
        if verbose:
            console.print("[dim]Cache column: ✓ = caching enabled, ✗ = no cache[/dim]")
        else:
            console.print("[dim]Tip: Use --verbose for more details[/dim]")

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]No data_sources.yaml found. Create one in config/ directory.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        import traceback

        console.print(traceback.format_exc())
        sys.exit(1)


@data_group.command("raw")
@click.option("--symbol", required=True, help="Symbol to load (e.g., AAPL)")
@click.option("--start-date", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end-date", required=True, help="End date (YYYY-MM-DD)")
@click.option(
    "--dataset",
    type=str,
    required=True,
    help="Dataset identifier (e.g., yahoo-us-equity-1d-csv)",
)
def raw_data(symbol: str, start_date: str, end_date: str, dataset: str):
    """
    Browse raw unadjusted historical data bars interactively.

    Displays data exactly as provided by the source.
    Press ENTER to display next bar, CTRL+C to exit.

    Example:
        Examples:
        # Fetch raw bars for AAPL
        qtrader data raw --symbol AAPL --start-date 2020-01-01 --end-date 2020-01-31 --dataset yahoo-us-equity-1d-csv
    """
    console = Console()

    try:
        # Parse and validate dates
        try:
            datetime.strptime(start_date, "%Y-%m-%d").date()
            datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError as e:
            console.print("[red]Error: Invalid date format. Use YYYY-MM-DD[/red]")
            console.print(f"[red]{e}[/red]")
            sys.exit(1)

        # Load data directly from adapter
        console.print(f"[cyan]Loading data for {symbol} from {dataset}...[/cyan]")

        resolver = DataSourceResolver()
        instrument = Instrument(symbol=symbol)
        adapter = resolver.resolve_by_dataset(dataset, instrument)

        # Read bars and convert to PriceBarEvents
        raw_bars = adapter.read_bars(start_date, end_date)
        bars = [adapter.to_price_bar_event(bar) for bar in raw_bars]

        if not bars:
            console.print(f"[yellow]No data found for {symbol} between {start_date} and {end_date}[/yellow]")
            sys.exit(0)

        console.print(f"[green]Loaded {len(bars)} bars[/green]")
        console.print("[dim]Displaying data as come from the source[/dim]")
        console.print("[dim]Press ENTER to view next bar, CTRL+C to exit[/dim]\n")

        # Display bars one by one
        for idx, bar in enumerate(bars, 1):
            # Create table for this bar
            table = create_bar_table(symbol, idx, len(bars))

            # Prepare bar data from PriceBarEvent
            bar_data = {
                "date": bar.timestamp.split("T")[0],  # Extract date from ISO timestamp
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": bar.volume,
                "dividend": 0.0,  # PriceBarEvent doesn't have dividend field directly
            }

            # Add data to table
            add_bar_data(table, bar_data)

            # Display table
            console.print(table)

            # Wait for user input (except on last bar)
            if idx < len(bars):
                try:
                    input()
                except KeyboardInterrupt:
                    console.print("\n[yellow]Exiting...[/yellow]")
                    break
            else:
                console.print("\n[green]End of data[/green]")

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Make sure the data files exist and the path is configured correctly.[/yellow]")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        import traceback

        console.print(traceback.format_exc())
        sys.exit(1)


@data_group.command("update")
@click.option(
    "-d",
    "--dataset",
    required=True,
    help="Dataset identifier (e.g., yahoo-us-equity-1d-csv)",
)
@click.option(
    "--symbols",
    help="Comma-separated list of symbols to update (default: all cached symbols)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be updated without making changes",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Show detailed update progress",
)
@click.option(
    "--force-reprime",
    is_flag=True,
    help="Delete existing cache and re-prime from scratch (useful for adapters without incremental support)",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Skip confirmation prompts (auto-confirm force-reprime deletion)",
)
def update_dataset(dataset: str, symbols: str, dry_run: bool, verbose: bool, force_reprime: bool, yes: bool):
    """
    Update cached data to latest available.

    Incrementally updates cached data by fetching only new bars since
    last update. Works with any dataset that supports incremental updates.

    If adapter doesn't support incremental updates, use --force-reprime to
    automatically delete cache and re-prime from scratch.

    If no symbols specified, updates all symbols found in cache.

    Examples:

        Examples:
        # Update dataset
        qtrader data update --dataset yahoo-us-equity-1d-csv

        # Force re-prime for adapter without incremental support
        qtrader data update --dataset my-custom-dataset --force-reprime
    """
    console = Console()

    try:
        # Parse symbols if provided
        symbol_list = [s.strip() for s in symbols.split(",")] if symbols else None

        # Show mode
        mode_str = "[yellow]DRY RUN[/yellow]" if dry_run else "[green]UPDATING[/green]"
        reprime_str = " [yellow](FORCE RE-PRIME)[/yellow]" if force_reprime else ""
        console.print(f"\n{mode_str}{reprime_str} Dataset: [cyan]{dataset}[/cyan]\n")

        # Create update service
        service = UpdateService(dataset)

        # Force-reprime confirmation prompt
        if force_reprime and not dry_run and not yes:
            # Get cache directory path to show user what will be deleted
            cache_root = service.updater._get_cache_root()

            console.print("[yellow]⚠️  WARNING: Force re-prime will DELETE existing cache data[/yellow]")
            console.print(f"[dim]Cache directory: {cache_root}[/dim]\n")

            # Parse symbols to show what will be affected
            symbol_list_preview = [s.strip() for s in symbols.split(",")] if symbols else None
            if symbol_list_preview:
                console.print(f"[yellow]Symbols to re-prime: {', '.join(symbol_list_preview)}[/yellow]")
            else:
                console.print("[yellow]This will delete caches for ALL symbols in the dataset[/yellow]")

            console.print()

            if not click.confirm("Are you sure you want to continue?", default=False):
                console.print("[yellow]Aborted.[/yellow]")
                return

        if force_reprime and not dry_run:
            console.print("[yellow]⚠ Force re-prime: Will delete existing caches and re-download all data[/yellow]\n")

        # Get symbols to update (service handles priority logic)
        symbols_to_update, source_desc = service.get_symbols_to_update(symbol_list)

        if not symbols_to_update:
            console.print("[yellow]No symbols found to update[/yellow]")
            console.print("[dim]Tip: Create universe.csv in cache directory or use --symbols[/dim]")
            return

        console.print(f"[cyan]Updating {source_desc}...[/cyan]\n")

        # Update with progress bar
        results = []

        with create_update_progress(console) as progress:
            task = progress.add_task("[cyan]Updating symbols...", total=len(symbols_to_update))

            # Update each symbol and show progress
            for result in service.update_symbols(
                symbols_to_update, dry_run=dry_run, verbose=verbose, force_reprime=force_reprime
            ):
                status_emoji = "✓" if result.success else "✗"
                progress.update(task, description=f"[cyan]{status_emoji} {result.symbol}", advance=1)
                results.append(result)

        console.print()  # Add blank line after progress

        # Show results summary
        if not results:
            console.print("[yellow]No symbols found to update[/yellow]")
            return

        # Create summary table
        table = create_update_summary_table()

        successful = 0
        total_bars = 0
        errors = []

        for result in results:
            # Get full cached metadata
            start_date, end_date, row_count = service.get_cache_metadata(result.symbol)

            if result.success:
                successful += 1
                total_bars += result.bars_added
            else:
                errors.append((result.symbol, result.error))

            # Add row to table
            add_update_result_row(
                table,
                result.symbol,
                result.success,
                result.bars_added,
                start_date,
                end_date,
                row_count,
                result.error,
            )

        console.print(table)

        # Summary stats
        console.print(f"\n[green]Successful:[/green] {successful}/{len(results)}")
        console.print(f"[cyan]Total bars added:[/cyan] {total_bars:,}")

        if errors:
            console.print(f"\n[red]Errors ({len(errors)}):[/red]")
            for symbol, error in errors:
                console.print(f"  [red]•[/red] {symbol}: {error}")

        if dry_run:
            console.print("\n[yellow]This was a dry run. Use --no-dry-run to actually update data.[/yellow]")

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        import traceback

        console.print(traceback.format_exc())
        sys.exit(1)


@data_group.command("cache-info")
@click.option(
    "--dataset",
    required=True,
    help="Dataset identifier (e.g., yahoo-us-equity-1d-csv)",
)
def cache_info(dataset: str):
    """
    Show cache information for a dataset.

    Displays cached symbols, date ranges, and update status.

    Examples:
        qtrader data cache-info --dataset yahoo-us-equity-1d-csv
    """
    console = Console()

    try:
        # Create update service
        service = UpdateService(dataset)

        # Get cache info
        cache_root = service.get_cache_root()
        if not cache_root or not cache_root.exists():
            console.print(f"[yellow]No cache found for dataset: {dataset}[/yellow]")
            return

        symbols = service.scan_cached_symbols()
        if not symbols:
            console.print(f"[yellow]Cache directory empty: {cache_root}[/yellow]")
            return

        # Show summary
        console.print(f"\n[cyan]Dataset:[/cyan] {dataset}")
        console.print(f"[cyan]Cache location:[/cyan] {cache_root}")
        console.print(f"[cyan]Cached symbols:[/cyan] {len(symbols)}\n")

        # Create table
        table = create_cache_info_table()

        # Add rows for each symbol
        for symbol in symbols:
            metadata = service.read_symbol_metadata(symbol, cache_root)
            add_cache_info_row(
                table,
                metadata["symbol"],
                metadata["start_date"],
                metadata["end_date"],
                str(metadata["row_count"]),
                metadata["last_update"],
            )

        console.print(table)

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        import traceback

        console.print(traceback.format_exc())
        sys.exit(1)


@data_group.command("yahoo-update")
@click.argument("symbols", nargs=-1)
@click.option("--start", help="Start date (YYYY-MM-DD)")
@click.option("--end", help="End date (YYYY-MM-DD)")
@click.option("--full-refresh", is_flag=True, help="Re-download all data")
@click.option(
    "--data-source",
    default="yahoo-us-equity-1d-csv",
    help="Data source name from data_sources.yaml (default: yahoo-us-equity-1d-csv)",
)
@click.option(
    "--data-dir",
    type=click.Path(exists=False),
    help="Override data directory path (default: read from data_sources.yaml)",
)
def yahoo_update_command(symbols: tuple, start: str, end: str, full_refresh: bool, data_source: str, data_dir: str):
    """
    Update Yahoo Finance data for symbols.

    Uses yahoo_update.py functions to fetch and update OHLCV data and dividends.
    Performs incremental updates by default, fetching only new bars since last update.

    Features:
    - Incremental updates (only new data since last date)
    - Full refresh mode (re-download all data)
    - Dividend tracking (maintains dividends_calendar.json)
    - Rate limiting (20 req/min to avoid API blacklisting)
    - Batch processing (50 symbols per batch with 30s pauses)
    - Automatic retry with exponential backoff

    If no symbols provided, updates all symbols from universe.json or discovers
    symbols from existing CSV files in the data directory.

    Examples:

        \b
        # Update all symbols from universe.json
        qtrader data yahoo-update

        # Update specific symbols
        qtrader data yahoo-update AAPL MSFT GOOGL

        # Use different data source
        qtrader data yahoo-update --data-source yahoo-us-equity-1d-csv

        # Override data directory
        qtrader data yahoo-update --data-dir /path/to/data

        # Update with date range
        qtrader data yahoo-update --start 2020-01-01 --end 2024-12-31

        # Force full refresh
        qtrader data yahoo-update --full-refresh
    """
    console = Console()

    try:
        # Resolve data directory
        data_path: Path
        if data_dir:
            # Use explicit override
            data_path = Path(data_dir)
            if not data_path.is_absolute():
                # Make relative to current directory
                data_path = Path.cwd() / data_path
            console.print(f"[dim]Using explicit data directory: {data_path}[/dim]")
        else:
            # Load from data_sources.yaml
            loaded_path = get_yahoo_data_dir(data_source)
            if loaded_path is None:
                console.print(
                    f"[yellow]Could not find '{data_source}' in data_sources.yaml, using default path[/yellow]"
                )
                # Fallback to default
                data_path = Path.cwd() / "data" / "us-equity-yahoo-csv"
            else:
                data_path = loaded_path
                console.print(f"[dim]Loaded data directory from data_sources.yaml: {data_path}[/dim]")

        if not data_path.exists():
            console.print(f"[red]Error: Data directory not found: {data_path}[/red]")
            console.print("[yellow]Tip: Check data_sources.yaml or use --data-dir to specify path[/yellow]")
            sys.exit(1)

        universe_path = data_path / "universe.json"
        dividends_path = data_path / "dividends_calendar.json"

        # Determine symbols to update
        symbol_list: list[str]
        if symbols:
            symbol_list = list(symbols)
        else:
            # Try to load from universe.json first
            loaded_symbols = load_universe(universe_path)
            if loaded_symbols is None:
                # Fallback: discover from existing CSV files
                console.print("[yellow]universe.json not found, discovering symbols from CSV files...[/yellow]")
                symbol_list = discover_symbols(data_path)
                if not symbol_list:
                    console.print(f"[yellow]No CSV files found in {data_path}[/yellow]")
                    sys.exit(0)
            elif not loaded_symbols:
                console.print("[yellow]universe.json exists but contains no tickers[/yellow]")
                sys.exit(0)
            else:
                symbol_list = loaded_symbols

        console.print(f"\n[cyan]Yahoo Finance Data Updater[/cyan]")
        console.print(f"Data directory: {data_path}")
        console.print(f"Symbols: {', '.join(symbol_list)}")
        console.print(f"Mode: {'Full Refresh' if full_refresh else 'Incremental Update'}")
        if start:
            console.print(f"Start date: {start}")
        if end:
            console.print(f"End date: {end}")
        else:
            safe_date = get_safe_end_date()
            if not is_market_closed():
                console.print(f"[yellow]Market is still open - excluding today's data (end date: {safe_date})[/yellow]")
            else:
                console.print(f"[green]Market is closed - including today's data (end date: {safe_date})[/green]")
        console.print()

        # Update symbols with progress bar and batch processing
        results = []
        total_symbols = len(symbol_list)
        num_batches = (total_symbols + BATCH_SIZE - 1) // BATCH_SIZE

        console.print(f"[dim]Processing {total_symbols} symbols in {num_batches} batch(es) of up to {BATCH_SIZE}[/dim]")
        console.print(
            f"[dim]Rate limit: {REQUESTS_PER_MINUTE} requests/minute ({MIN_REQUEST_INTERVAL:.2f}s between requests)[/dim]"
        )
        if num_batches > 1:
            console.print(f"[dim]Batch pause: {BATCH_PAUSE:.0f}s between batches to avoid rate limiting[/dim]")
        console.print()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Updating symbols...", total=total_symbols)

            # Process symbols in batches
            for batch_num in range(num_batches):
                start_idx = batch_num * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, total_symbols)
                batch_symbols = symbol_list[start_idx:end_idx]

                # Update each symbol in batch
                for symbol in batch_symbols:
                    progress.update(
                        task, description=f"[cyan]Batch {batch_num + 1}/{num_batches}: Updating {symbol}..."
                    )
                    result = update_symbol(
                        symbol,
                        data_path,
                        dividends_path,
                        start_date=start,
                        end_date=end,
                        full_refresh=full_refresh,
                        rate_limit=True,
                    )
                    results.append(result)
                    progress.advance(task)

                # Pause between batches (except after last batch)
                if batch_num < num_batches - 1:
                    progress.update(
                        task,
                        description=f"[yellow]Pausing {BATCH_PAUSE:.0f}s before next batch to avoid rate limiting...[/yellow]",
                    )
                    time.sleep(BATCH_PAUSE)

        # Display results
        console.print("\n[bold]Update Summary:[/bold]\n")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Symbol", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Action")
        table.add_column("Bars Added", justify="right")
        table.add_column("Dividends", justify="right")

        success_count = 0
        total_bars = 0
        total_dividends = 0

        for result in results:
            status = "✓" if result["success"] else "✗"

            table.add_row(
                result["symbol"],
                status,
                result["action"],
                str(result["bars_added"]),
                str(result["dividends_added"]),
            )

            if result["success"]:
                success_count += 1
                total_bars += result["bars_added"]
                total_dividends += result["dividends_added"]

        console.print(table)

        console.print(f"\n[bold]Results:[/bold]")
        console.print(f"  Successful: {success_count}/{len(symbol_list)}")
        console.print(f"  Total bars added: {total_bars}")
        console.print(f"  Total dividends added: {total_dividends}")

        if dividends_path.exists():
            console.print(f"\n[dim]Dividends calendar: {dividends_path}[/dim]")

    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        import traceback

        console.print(traceback.format_exc())
        sys.exit(1)
