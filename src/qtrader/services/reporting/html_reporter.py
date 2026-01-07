"""HTML Report Generator for Backtest Results.

Generates a standalone, interactive HTML report with embedded charts
that users can open directly in their browser without any dependencies.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qtrader.system import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class HTMLReportGenerator:
    """Generates standalone HTML reports for backtest results."""

    def __init__(self, output_dir: Path):
        """
        Initialize HTML report generator.

        Args:
            output_dir: Path to the backtest output directory containing:
                - performance.json
                - manifest.json (optional)
                - timeseries/*.json files
                - events.parquet
        """
        self.output_dir = Path(output_dir)
        self.timeseries_dir = self.output_dir / "timeseries"

    def generate(self) -> Path:
        """
        Generate standalone HTML report.

        Returns:
            Path to generated report.html file

        Raises:
            FileNotFoundError: If required data files are missing
            ValueError: If data is invalid or corrupted
        """
        # Load data
        performance = self._load_performance()
        manifest = self._load_manifest()
        metadata = self._load_metadata()
        config_snapshot = self._load_config_snapshot()
        equity_curve = self._load_timeseries("equity_curve.json")
        returns = self._load_timeseries("returns.json")
        drawdowns = self._load_timeseries("drawdowns.json")
        trades = self._load_trades()

        # Generate HTML sections
        html = self._build_html(
            performance=performance,
            manifest=manifest,
            metadata=metadata,
            config_snapshot=config_snapshot,
            equity_curve=equity_curve,
            returns=returns,
            drawdowns=drawdowns,
            trades=trades,
        )

        # Write to file
        report_path = self.output_dir / "report.html"
        report_path.write_text(html, encoding="utf-8")

        return report_path

    def _load_performance(self) -> dict[str, Any]:
        """Load performance.json."""
        perf_path = self.output_dir / "performance.json"
        if not perf_path.exists():
            raise FileNotFoundError(f"performance.json not found: {perf_path}")
        data: dict[str, Any] = json.loads(perf_path.read_text())
        return data

    def _load_manifest(self) -> dict[str, Any] | None:
        """Load manifest.json (optional)."""
        manifest_path = self.output_dir / "manifest.json"
        if manifest_path.exists():
            data: dict[str, Any] = json.loads(manifest_path.read_text())
            return data
        return None

    def _load_timeseries(self, filename: str) -> pd.DataFrame | None:
        """Load a timeseries JSON file (returns None if not found)."""
        filepath = self.timeseries_dir / filename
        if filepath.exists():
            # Load JSON data
            with filepath.open("r") as f:
                data = json.load(f)

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Convert timestamp strings to datetime if present
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])

                # Handle duplicate timestamps by keeping the last value for each timestamp
                # This can occur when multiple portfolio state events are emitted at the same time
                if df["timestamp"].duplicated().any():
                    df = df.drop_duplicates(subset=["timestamp"], keep="last")

            return df
        return None

    def _load_trades(self) -> list[dict[str, Any]]:
        """Load trade events from events.parquet."""
        import json

        events_path = self.output_dir / "events.parquet"
        if not events_path.exists():
            return []

        try:
            events_df = pd.read_parquet(events_path)
            trade_events = events_df[events_df["event_type"] == "trade"]

            trades = []
            for _, row in trade_events.iterrows():
                payload = json.loads(row["payload"])
                trades.append(payload)

            return trades
        except Exception:
            return []

    def _load_metadata(self) -> dict[str, Any] | None:
        """Load metadata.json if available."""
        import json

        metadata_path = self.output_dir / "metadata.json"
        if not metadata_path.exists():
            return None

        try:
            data: dict[str, Any] = json.loads(metadata_path.read_text())
            return data
        except Exception:
            return None

    def _load_config_snapshot(self) -> dict[str, Any] | None:
        """Load config_snapshot.yaml if available."""
        import yaml

        config_path = self.output_dir / "config_snapshot.yaml"
        if not config_path.exists():
            return None

        try:
            with open(config_path) as f:
                data: dict[str, Any] | None = yaml.safe_load(f)
                return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _build_html(
        self,
        performance: dict[str, Any],
        manifest: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
        config_snapshot: dict[str, Any] | None,
        equity_curve: pd.DataFrame | None,
        returns: pd.DataFrame | None,
        drawdowns: pd.DataFrame | None,
        trades: list[dict[str, Any]] | None,
    ) -> str:
        """Build complete HTML document."""
        # Generate charts
        combined_chart = self._create_combined_chart(equity_curve, drawdowns)

        # Build HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report: {performance.get("backtest_id", "Unknown")}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #f5f5f7;
            color: #1d1d1f;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        header h1 {{
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}
        header p {{
            opacity: 0.9;
            font-size: 0.95rem;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .metric-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border-left: 4px solid #667eea;
        }}
        .metric-card.positive {{
            border-left-color: #10b981;
        }}
        .metric-card.negative {{
            border-left-color: #ef4444;
        }}
        .metric-label {{
            font-size: 0.85rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.5rem;
        }}
        .metric-value {{
            font-size: 1.75rem;
            font-weight: 700;
            color: #1d1d1f;
        }}
        .chart-container {{
            background: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 2rem;
        }}
        .chart-title {{
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #1d1d1f;
        }}
        table {{
            width: 100%;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        th {{
            background: #f9fafb;
            padding: 1rem;
            text-align: left;
            font-weight: 600;
            color: #374151;
            border-bottom: 2px solid #e5e7eb;
        }}
        td {{
            padding: 1rem;
            border-bottom: 1px solid #e5e7eb;
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        tr:hover {{
            background: #f9fafb;
        }}
        .heatmap-table td {{
            text-align: center;
            padding: 0.75rem;
            font-size: 0.875rem;
        }}
        .heatmap-table th {{
            text-align: center;
            padding: 0.75rem;
            font-size: 0.875rem;
        }}
        .heatmap-table td.positive {{
            background: #d1fae5;
            color: #065f46;
        }}
        .heatmap-table td.negative {{
            background: #fee2e2;
            color: #991b1b;
        }}
        .heatmap-table td.neutral {{
            background: #f9fafb;
            color: #6b7280;
        }}
        .heatmap-table tr:hover td {{
            opacity: 0.9;
        }}
        .footer {{
            text-align: center;
            padding: 2rem;
            color: #6b7280;
            font-size: 0.875rem;
        }}
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .badge.success {{
            background: #d1fae5;
            color: #065f46;
        }}
        .badge.warning {{
            background: #fef3c7;
            color: #92400e;
        }}
        .badge.danger {{
            background: #fee2e2;
            color: #991b1b;
        }}
        .info-section {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .info-section h3 {{
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #374151;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
        }}
        .info-item {{
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid #f3f4f6;
        }}
        .info-item:last-child {{
            border-bottom: none;
        }}
        .info-label {{
            color: #6b7280;
            font-size: 0.875rem;
        }}
        .info-value {{
            font-weight: 600;
            color: #1d1d1f;
            font-size: 0.875rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 {performance.get("backtest_id", "Backtest Report")}</h1>
            <p>{performance.get("start_date")} to {performance.get("end_date")} ({performance.get("duration_days")} days)</p>
            {self._build_manifest_badges(manifest)}
        </header>

        {self._build_key_metrics(performance)}

        {self._build_run_info(performance, manifest, metadata, config_snapshot)}

        {self._build_price_chart()}

        {self._build_indicator_charts()}

        <div class="chart-container">
            <div class="chart-title">📈 Portfolio Performance</div>
            {combined_chart}
        </div>

        {self._build_monthly_breakdown_table(performance)}

        {self._build_performance_table(performance)}

        {self._build_trades_table(trades, equity_curve, performance)}

        <div class="footer">
            <p>Generated by QTrader • {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p style="margin-top: 0.5rem;">Raw data available in: {self.output_dir.name}</p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def _build_manifest_badges(self, manifest: dict[str, Any] | None) -> str:
        """Build status badges from manifest."""
        if not manifest:
            return ""

        badges = []
        if git_info := manifest.get("git"):
            if commit := git_info.get("commit_hash"):
                badges.append(f'<span class="badge success">Git: {commit[:7]}</span>')
            if git_info.get("has_uncommitted_changes"):
                badges.append('<span class="badge warning">Uncommitted Changes</span>')

        if badges:
            return f'<p style="margin-top: 1rem;">{" ".join(badges)}</p>'
        return ""

    def _build_key_metrics(self, performance: dict[str, Any]) -> str:
        """Build key metrics cards."""
        # Determine metric card classes based on values
        total_return = float(performance.get("total_return_pct", 0))
        sharpe = float(performance.get("sharpe_ratio", 0))
        max_dd = float(performance.get("max_drawdown_pct", 0))

        return_class = "positive" if total_return > 0 else "negative" if total_return < 0 else ""
        sharpe_class = "positive" if sharpe > 1 else "negative" if sharpe < 0 else ""
        dd_class = "negative" if abs(max_dd) > 10 else ""

        return f"""
        <div class="metrics-grid">
            <div class="metric-card {return_class}">
                <div class="metric-label">Total Return</div>
                <div class="metric-value">{performance.get("total_return_pct", "0")}%</div>
            </div>
            <div class="metric-card {return_class}">
                <div class="metric-label">CAGR</div>
                <div class="metric-value">{performance.get("cagr", "0")}%</div>
            </div>
            <div class="metric-card {sharpe_class}">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{performance.get("sharpe_ratio", "0")}</div>
            </div>
            <div class="metric-card {dd_class}">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value">{performance.get("max_drawdown_pct", "0")}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Volatility (Annual)</div>
                <div class="metric-value">{performance.get("volatility_annual_pct", "0")}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{performance.get("total_trades", 0)}</div>
            </div>
        </div>
        """

    def _build_run_info(
        self,
        performance: dict[str, Any],
        manifest: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
        config_snapshot: dict[str, Any] | None,
    ) -> str:
        """Build run information sections."""
        sections = []

        # Section 1: Run Information
        info_items_run = []
        initial_equity = float(performance.get("initial_equity", 0))
        final_equity = float(performance.get("final_equity", 0))

        info_items_run.append(
            f'<div class="info-item"><span class="info-label">Initial Equity</span><span class="info-value">${initial_equity:,.2f}</span></div>'
        )
        info_items_run.append(
            f'<div class="info-item"><span class="info-label">Final Equity</span><span class="info-value">${final_equity:,.2f}</span></div>'
        )

        # Show actual date range processed (from performance data)
        if actual_start := performance.get("start_date"):
            info_items_run.append(
                f'<div class="info-item"><span class="info-label">Actual Start Date</span><span class="info-value">{actual_start}</span></div>'
            )

        if actual_end := performance.get("end_date"):
            info_items_run.append(
                f'<div class="info-item"><span class="info-label">Actual End Date</span><span class="info-value">{actual_end}</span></div>'
            )

        info_items_run.append(
            f'<div class="info-item"><span class="info-label">Duration</span><span class="info-value">{performance.get("duration_days", 0)} days</span></div>'
        )

        if manifest:
            if timestamp := manifest.get("timestamp"):
                info_items_run.append(
                    f'<div class="info-item"><span class="info-label">Run Time</span><span class="info-value">{timestamp}</span></div>'
                )
            if git_info := manifest.get("git"):
                if branch := git_info.get("branch"):
                    info_items_run.append(
                        f'<div class="info-item"><span class="info-label">Git Branch</span><span class="info-value">{branch}</span></div>'
                    )

        sections.append(
            f"""
        <div class="info-section">
            <h3>📋 Run Information</h3>
            <div class="info-grid">
                {"".join(info_items_run)}
            </div>
        </div>
        """
        )

        # Section 2: Configuration (from metadata)
        if metadata:
            info_items_config = []

            # Backtest metadata
            if backtest := metadata.get("backtest"):
                if backtest_id := backtest.get("backtest_id"):
                    info_items_config.append(
                        f'<div class="info-item"><span class="info-label">Backtest ID</span><span class="info-value">{backtest_id}</span></div>'
                    )

                if start_date := backtest.get("start_date"):
                    info_items_config.append(
                        f'<div class="info-item"><span class="info-label">Requested Start</span><span class="info-value">{start_date[:10]}</span></div>'
                    )

                if end_date := backtest.get("end_date"):
                    info_items_config.append(
                        f'<div class="info-item"><span class="info-label">Requested End</span><span class="info-value">{end_date[:10]}</span></div>'
                    )

                if strategy_adj := backtest.get("strategy_adjustment_mode"):
                    info_items_config.append(
                        f'<div class="info-item"><span class="info-label">Strategy Adjustment</span><span class="info-value">{strategy_adj}</span></div>'
                    )

                if portfolio_adj := backtest.get("portfolio_adjustment_mode"):
                    info_items_config.append(
                        f'<div class="info-item"><span class="info-label">Portfolio Adjustment</span><span class="info-value">{portfolio_adj}</span></div>'
                    )

                if risk_policy := backtest.get("risk_policy"):
                    if isinstance(risk_policy, dict):
                        policy_name = risk_policy.get("name", "N/A")
                    else:
                        policy_name = str(risk_policy)
                    info_items_config.append(
                        f'<div class="info-item"><span class="info-label">Risk Policy</span><span class="info-value">{policy_name}</span></div>'
                    )

                # Data sources from backtest.data.sources
                if data_config := backtest.get("data"):
                    if sources := data_config.get("sources"):
                        if isinstance(sources, list) and sources:
                            source_names = [s.get("name", "") for s in sources if isinstance(s, dict)]
                            sources_str = ", ".join(source_names) if source_names else "N/A"
                            info_items_config.append(
                                f'<div class="info-item"><span class="info-label">Data Sources</span><span class="info-value">{sources_str}</span></div>'
                            )

                # Reporting config
                if reporting := backtest.get("reporting"):
                    if risk_free := reporting.get("risk_free_rate"):
                        info_items_config.append(
                            f'<div class="info-item"><span class="info-label">Risk-Free Rate</span><span class="info-value">{risk_free:.2%}</span></div>'
                        )

            if info_items_config:
                sections.append(
                    f"""
        <div class="info-section">
            <h3>⚙️ Configuration</h3>
            <div class="info-grid">
                {"".join(info_items_config)}
            </div>
        </div>
        """
                )

        # Section 3: Strategy & Universe (from config_snapshot)
        if config_snapshot:
            info_items_strategy = []

            # Get universe from data.sources
            if data_config := config_snapshot.get("data"):
                if sources := data_config.get("sources"):
                    if isinstance(sources, list):
                        all_symbols = set()
                        for source in sources:
                            if isinstance(source, dict) and (universe := source.get("universe")):
                                if isinstance(universe, list):
                                    all_symbols.update(universe)

                        if all_symbols:
                            symbols_str = ", ".join(sorted(all_symbols))
                            info_items_strategy.append(
                                f'<div class="info-item"><span class="info-label">Universe</span><span class="info-value">{symbols_str}</span></div>'
                            )

            if strategies := config_snapshot.get("strategies"):
                if isinstance(strategies, list) and strategies:
                    strategy = strategies[0]
                    if strategy_id := strategy.get("strategy_id"):
                        info_items_strategy.append(
                            f'<div class="info-item"><span class="info-label">Strategy ID</span><span class="info-value">{strategy_id}</span></div>'
                        )

                    # Strategy universe
                    if strat_universe := strategy.get("universe"):
                        if isinstance(strat_universe, list):
                            strat_symbols = ", ".join(strat_universe)
                            info_items_strategy.append(
                                f'<div class="info-item"><span class="info-label">Strategy Universe</span><span class="info-value">{strat_symbols}</span></div>'
                            )

                    # Strategy data sources
                    if strat_data := strategy.get("data_sources"):
                        if isinstance(strat_data, list):
                            data_str = ", ".join(strat_data)
                            info_items_strategy.append(
                                f'<div class="info-item"><span class="info-label">Strategy Data</span><span class="info-value">{data_str}</span></div>'
                            )

                    # Strategy config parameters (all of them)
                    if config := strategy.get("config"):
                        if isinstance(config, dict):
                            for key, value in config.items():
                                # Format the key nicely (convert snake_case to Title Case)
                                label = key.replace("_", " ").title()
                                info_items_strategy.append(
                                    f'<div class="info-item"><span class="info-label">{label}</span><span class="info-value">{value}</span></div>'
                                )

            if info_items_strategy:
                sections.append(
                    f"""
        <div class="info-section">
            <h3>🎯 Strategy & Universe</h3>
            <div class="info-grid">
                {"".join(info_items_strategy)}
            </div>
        </div>
        """
                )

        return "".join(sections)

    def _create_combined_chart(self, equity_curve: pd.DataFrame | None, drawdowns: pd.DataFrame | None) -> str:
        """Create combined equity curve and drawdown chart."""
        if equity_curve is None:
            return "<p>Equity curve data not available</p>"

        # Reset index after deduplication to avoid gaps
        equity_curve = equity_curve.reset_index(drop=True)

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=["Equity Curve", "Drawdown"],
            vertical_spacing=0.1,
            shared_xaxes=True,
        )

        # Equity curve - convert to lists to avoid pandas index issues
        fig.add_trace(
            go.Scatter(
                x=equity_curve["timestamp"].tolist(),
                y=equity_curve["equity"].tolist(),
                mode="lines",
                name="Portfolio Value",
                line=dict(color="#667eea", width=2),
                fill="tozeroy",
                fillcolor="rgba(102, 126, 234, 0.1)",
            ),
            row=1,
            col=1,
        )

        # Drawdown - use drawdown_pct from equity_curve (not drawdowns DataFrame)
        if "drawdown_pct" in equity_curve.columns:
            fig.add_trace(
                go.Scatter(
                    x=equity_curve["timestamp"].tolist(),
                    y=equity_curve["drawdown_pct"].tolist(),
                    mode="lines",
                    name="Drawdown %",
                    line=dict(color="#ef4444", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(239, 68, 68, 0.2)",
                ),
                row=2,
                col=1,
            )

        # Update layout
        fig.update_layout(
            height=600,
            showlegend=True,
            hovermode="x unified",
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=50, r=50, t=50, b=50),
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f0f0f0")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f0f0f0")

        # Format axes
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)

        html: str = fig.to_html(include_plotlyjs=False, div_id="combined-chart")
        return html

    def _create_equity_chart(self, equity_curve: pd.DataFrame) -> str:
        """Create equity curve chart."""
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=equity_curve["timestamp"],
                y=equity_curve["equity"],
                mode="lines",
                name="Portfolio Value",
                line=dict(color="#667eea", width=2),
                fill="tozeroy",
                fillcolor="rgba(102, 126, 234, 0.1)",
            )
        )

        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode="x unified",
            height=400,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f0f0f0")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f0f0f0")

        html: str = fig.to_html(include_plotlyjs=False, div_id="equity-chart")
        return html

    def _create_drawdown_chart(self, drawdowns: pd.DataFrame) -> str:
        """Create drawdown chart."""
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=drawdowns["timestamp"],
                y=drawdowns["drawdown_pct"],
                mode="lines",
                name="Drawdown",
                line=dict(color="#ef4444", width=2),
                fill="tozeroy",
                fillcolor="rgba(239, 68, 68, 0.2)",
            )
        )

        fig.update_layout(
            title="Drawdown Over Time",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode="x unified",
            height=300,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f0f0f0")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f0f0f0")

        html: str = fig.to_html(include_plotlyjs=False, div_id="drawdown-chart")
        return html

    def _create_monthly_returns_chart(self, performance: dict[str, Any]) -> str:
        """Create monthly returns bar chart."""
        monthly_returns = performance.get("monthly_returns", [])
        if not monthly_returns:
            return ""

        periods = [m["period"] for m in monthly_returns]
        returns = [float(m["return_pct"]) for m in monthly_returns]

        colors = ["#10b981" if r >= 0 else "#ef4444" for r in returns]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=periods,
                y=returns,
                marker_color=colors,
                name="Monthly Return",
                hovertemplate="<b>%{x}</b><br>Return: %{y:.2f}%<extra></extra>",
            )
        )

        fig.update_layout(
            title="Monthly Returns",
            xaxis_title="Month",
            yaxis_title="Return (%)",
            height=350,
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
        )

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(
            showgrid=True, gridwidth=1, gridcolor="#f0f0f0", zeroline=True, zerolinewidth=2, zerolinecolor="#9ca3af"
        )

        return f"""
        <div class="chart-container">
            <div class="chart-title">📅 Monthly Performance</div>
            {fig.to_html(include_plotlyjs=False, div_id="monthly-chart")}
        </div>
        """

    def _build_performance_table(self, performance: dict[str, Any]) -> str:
        """Build comprehensive performance metrics table with multi-column layout."""

        # Format returns with proper precision
        best_day = float(performance.get("best_day_return_pct", 0))
        worst_day = float(performance.get("worst_day_return_pct", 0))

        metrics = [
            (
                "Returns",
                [
                    ("Total Return", f"{performance.get('total_return_pct', '0')}%"),
                    ("CAGR", f"{performance.get('cagr', '0')}%"),
                    ("Best Day", f"{best_day:.2f}%"),
                    ("Worst Day", f"{worst_day:.2f}%"),
                ],
            ),
            (
                "Risk Metrics",
                [
                    ("Volatility (Annual)", f"{performance.get('volatility_annual_pct', '0')}%"),
                    ("Max Drawdown", f"{performance.get('max_drawdown_pct', '0')}%"),
                    ("Max DD Duration", f"{performance.get('max_drawdown_duration_days', 0)} days"),
                    ("Avg Drawdown", f"{float(performance.get('avg_drawdown_pct', 0)):.2f}%"),
                    ("Current Drawdown", f"{performance.get('current_drawdown_pct', '0')}%"),
                ],
            ),
            (
                "Risk-Adjusted Returns",
                [
                    ("Sharpe Ratio", performance.get("sharpe_ratio", "0")),
                    ("Sortino Ratio", performance.get("sortino_ratio", "0")),
                    ("Calmar Ratio", performance.get("calmar_ratio", "0")),
                    ("Risk-Free Rate", f"{float(performance.get('risk_free_rate', 0)) * 100:.2f}%"),
                ],
            ),
            (
                "Trade Statistics",
                [
                    ("Total Trades", performance.get("total_trades", 0)),
                    ("Winning Trades", performance.get("winning_trades", 0)),
                    ("Losing Trades", performance.get("losing_trades", 0)),
                    ("Win Rate", f"{performance.get('win_rate', '0')}%"),
                    (
                        "Profit Factor",
                        performance.get("profit_factor", "N/A") if performance.get("profit_factor") else "N/A",
                    ),
                    ("Avg Win", f"${float(performance.get('avg_win', 0)):,.2f}"),
                    ("Avg Loss", f"${float(performance.get('avg_loss', 0)):,.2f}"),
                    ("Expectancy", f"${float(performance.get('expectancy', 0)):,.2f}"),
                ],
            ),
        ]

        # Build sections with 2-column grid layout
        sections_html = []
        for i in range(0, len(metrics), 2):
            section_pair = metrics[i : i + 2]
            pair_html = '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">'

            for section_title, items in section_pair:
                rows = "".join(
                    [f"<tr><td>{label}</td><td><strong>{value}</strong></td></tr>" for label, value in items]
                )
                pair_html += f"""
                    <div class="chart-container" style="margin-bottom: 0;">
                        <div class="chart-title">{section_title}</div>
                        <table>
                            <tbody>
                                {rows}
                            </tbody>
                        </table>
                    </div>
                """

            pair_html += "</div>"
            sections_html.append(pair_html)

        return "".join(sections_html)

    def _build_monthly_breakdown_table(self, performance: dict[str, Any]) -> str:
        """Build monthly returns heatmap table (years × months)."""
        monthly_returns = performance.get("monthly_returns", [])
        if not monthly_returns:
            return ""

        # Organize data by year and month
        from collections import defaultdict

        data_by_year: dict[str, dict[str, float]] = defaultdict(dict)

        for period in monthly_returns:
            # Parse period like "2020-03" into year and month
            year, month = period["period"].split("-")
            return_pct = float(period["return_pct"])
            data_by_year[year][month] = return_pct

        # Build table
        months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        # Header row
        header = "<tr><th>Year</th>" + "".join([f"<th>{name}</th>" for name in month_names]) + "<th>YTD</th></tr>"

        # Data rows
        rows = []
        for year in sorted(data_by_year.keys()):
            row_cells = [f"<td><strong>{year}</strong></td>"]
            ytd_return = 1.0  # Multiplicative YTD

            for month in months:
                if month in data_by_year[year]:
                    ret = data_by_year[year][month]
                    ytd_return *= 1 + ret / 100

                    # Color coding
                    if ret > 0:
                        cell_class = "positive"
                    elif ret < 0:
                        cell_class = "negative"
                    else:
                        cell_class = "neutral"

                    row_cells.append(f'<td class="{cell_class}"><strong>{ret:.2f}%</strong></td>')
                else:
                    row_cells.append('<td class="neutral">—</td>')

            # YTD column
            ytd_pct = (ytd_return - 1) * 100
            ytd_class = "positive" if ytd_pct > 0 else "negative" if ytd_pct < 0 else "neutral"
            row_cells.append(f'<td class="{ytd_class}"><strong>{ytd_pct:.2f}%</strong></td>')

            rows.append("<tr>" + "".join(row_cells) + "</tr>")

        return f"""
        <div class="chart-container">
            <div class="chart-title">📊 Monthly Returns Heatmap</div>
            <table class="heatmap-table">
                <thead>
                    {header}
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        </div>
        """

    def _build_price_chart(self) -> str:
        """Build instrument price chart from bar events with overlay indicators."""
        import json

        events_path = self.output_dir / "events.parquet"
        if not events_path.exists():
            return ""

        try:
            events_df = pd.read_parquet(events_path)
            bar_events = events_df[events_df["event_type"] == "bar"]

            if len(bar_events) == 0:
                return ""

            # Extract price data by symbol
            price_data: dict[str, list[dict[str, Any]]] = {}
            for _, row in bar_events.iterrows():
                payload = json.loads(row["payload"])
                symbol = payload.get("symbol", "")
                if symbol not in price_data:
                    price_data[symbol] = []
                price_data[symbol].append(
                    {
                        "timestamp": payload.get("timestamp", ""),
                        "close": float(payload.get("close", 0)),
                    }
                )

            if not price_data:
                return ""

            # Extract overlay indicators by symbol
            overlay_indicators: dict[str, dict[str, list[tuple[str, Any]]]] = {}
            indicator_colors: dict[str, dict[str, str]] = {}
            indicator_events = events_df[events_df["event_type"] == "indicator"]

            if len(indicator_events) > 0:
                for _, row in indicator_events.iterrows():
                    payload = json.loads(row["payload"])
                    symbol = payload.get("symbol", "")
                    timestamp = payload.get("timestamp", "")
                    indicators = payload.get("indicators", {})
                    metadata = payload.get("metadata", {})
                    placements = metadata.get("placements", {})
                    colors_meta = metadata.get("colors", {})

                    # Only process overlay indicators
                    for name, value in indicators.items():
                        if placements.get(name, "subplot") == "overlay":
                            if symbol not in overlay_indicators:
                                overlay_indicators[symbol] = {}
                                indicator_colors[symbol] = {}

                            if name not in overlay_indicators[symbol]:
                                overlay_indicators[symbol][name] = []

                            overlay_indicators[symbol][name].append((timestamp, value))

                            # Store color if provided
                            if name not in indicator_colors[symbol] and name in colors_meta:
                                indicator_colors[symbol][name] = colors_meta[name]

            num_symbols = len(price_data)

            # Define color palette for multiple symbols
            colors = [
                "#667eea",
                "#764ba2",
                "#f093fb",
                "#4facfe",
                "#43e97b",
                "#fa709a",
                "#fee140",
                "#30cfd0",
            ]

            # For multiple symbols with very different price scales, use subplots
            # Check if price ranges differ significantly
            price_ranges = {}
            for symbol, data in price_data.items():
                prices = [d["close"] for d in data]
                price_ranges[symbol] = (min(prices), max(prices))

            # Calculate if we need separate subplots (price ranges differ by >10x)
            use_subplots = False
            if num_symbols > 1:
                max_range = max(r[1] for r in price_ranges.values())
                min_range = min(r[0] for r in price_ranges.values())
                if max_range / max(min_range, 0.01) > 10:
                    use_subplots = True

            if use_subplots and num_symbols > 1:
                # Create subplots for symbols with very different price scales
                fig = make_subplots(
                    rows=num_symbols,
                    cols=1,
                    subplot_titles=[symbol for symbol in sorted(price_data.keys())],
                    vertical_spacing=0.08,
                    shared_xaxes=True,
                )

                for idx, (symbol, data) in enumerate(sorted(price_data.items()), start=1):
                    data_sorted = sorted(data, key=lambda x: x["timestamp"])
                    timestamps = [d["timestamp"] for d in data_sorted]
                    prices = [d["close"] for d in data_sorted]

                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=prices,
                            mode="lines",
                            name=symbol,
                            line=dict(width=2, color=colors[(idx - 1) % len(colors)]),
                            showlegend=False,
                        ),
                        row=idx,
                        col=1,
                    )

                    # Add overlay indicators for this symbol
                    if symbol in overlay_indicators:
                        default_indicator_colors = ["#667eea", "#764ba2", "#f093fb", "#fa709a"]
                        for ind_idx, (ind_name, ind_data) in enumerate(sorted(overlay_indicators[symbol].items())):
                            ind_data_sorted = sorted(ind_data, key=lambda x: x[0])
                            ind_timestamps = [d[0] for d in ind_data_sorted]
                            ind_values = [d[1] for d in ind_data_sorted]

                            # Get color from metadata or use default
                            ind_color = indicator_colors[symbol].get(
                                ind_name, default_indicator_colors[ind_idx % len(default_indicator_colors)]
                            )

                            fig.add_trace(
                                go.Scatter(
                                    x=ind_timestamps,
                                    y=ind_values,
                                    mode="lines",
                                    name=ind_name,
                                    line=dict(width=1.5, color=ind_color, dash="dash"),
                                    showlegend=False,
                                ),
                                row=idx,
                                col=1,
                            )

                    fig.update_yaxes(title_text="Price ($)", row=idx, col=1)

                fig.update_xaxes(title_text="Date", row=num_symbols, col=1)
                height = 250 * num_symbols
            else:
                # Single chart for one symbol or multiple symbols with similar scales
                fig = go.Figure()

                for idx, (symbol, data) in enumerate(sorted(price_data.items())):
                    data_sorted = sorted(data, key=lambda x: x["timestamp"])
                    timestamps = [d["timestamp"] for d in data_sorted]
                    prices = [d["close"] for d in data_sorted]

                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=prices,
                            mode="lines",
                            name=symbol,
                            line=dict(width=2, color=colors[idx % len(colors)]),
                        )
                    )

                    # Add overlay indicators for this symbol
                    if symbol in overlay_indicators:
                        default_indicator_colors = ["#667eea", "#764ba2", "#f093fb", "#fa709a"]
                        for ind_idx, (ind_name, ind_data) in enumerate(sorted(overlay_indicators[symbol].items())):
                            ind_data_sorted = sorted(ind_data, key=lambda x: x[0])
                            ind_timestamps = [d[0] for d in ind_data_sorted]
                            ind_values = [d[1] for d in ind_data_sorted]

                            # Get color from metadata or use default
                            ind_color = indicator_colors[symbol].get(
                                ind_name, default_indicator_colors[ind_idx % len(default_indicator_colors)]
                            )

                            fig.add_trace(
                                go.Scatter(
                                    x=ind_timestamps,
                                    y=ind_values,
                                    mode="lines",
                                    name=ind_name,
                                    line=dict(width=1.5, color=ind_color, dash="dash"),
                                )
                            )

                fig.update_xaxes(title_text="Date")
                fig.update_yaxes(title_text="Price ($)")
                height = 400

            # Update layout
            # Show legend if multiple symbols OR if we have overlay indicators
            has_overlay = any(symbol in overlay_indicators for symbol in price_data.keys())
            show_legend = (num_symbols > 1 or has_overlay) and not use_subplots

            fig.update_layout(
                title="",
                hovermode="x unified",
                height=height,
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=show_legend,
                margin=dict(l=50, r=50, t=30 if not use_subplots else 50, b=50),
            )

            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f0f0f0")
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f0f0f0")

            html: str = fig.to_html(include_plotlyjs=False, div_id="price-chart")

            symbol_text = ", ".join(sorted(price_data.keys()))
            symbol_count_text = f"{num_symbols} Asset{'s' if num_symbols > 1 else ''}"
            return f"""
            <div class="chart-container">
                <div class="chart-title">📈 Instrument Prices · {symbol_count_text} ({symbol_text})</div>
                {html}
            </div>
            """
        except Exception:
            return ""

    def _build_indicator_charts(self) -> str:
        """Build indicator overlay and subplot charts from IndicatorEvents."""
        import json

        from plotly.subplots import make_subplots

        events_path = self.output_dir / "events.parquet"
        if not events_path.exists():
            return ""

        try:
            events_df = pd.read_parquet(events_path)
            indicator_events = events_df[events_df["event_type"] == "indicator"]

            if len(indicator_events) == 0:
                return ""

            # Group by symbol
            symbols = set()
            for _, row in indicator_events.iterrows():
                payload = json.loads(row["payload"])
                symbols.add(payload.get("symbol", ""))

            # Build charts for each symbol
            charts_html = []
            for symbol in sorted(symbols):
                symbol_events = [
                    json.loads(row["payload"])
                    for _, row in indicator_events.iterrows()
                    if json.loads(row["payload"]).get("symbol") == symbol
                ]

                if not symbol_events:
                    continue

                # Organize indicators by placement
                overlay_indicators: dict[str, list[tuple[str, Any]]] = {}
                subplot_indicators: dict[str, list[tuple[str, Any]]] = {}
                colors: dict[str, str] = {}
                default_colors = ["#667eea", "#764ba2", "#f093fb", "#fa709a", "#43e97b", "#4facfe"]

                for event in symbol_events:
                    timestamp = event.get("timestamp", "")
                    indicators = event.get("indicators", {})
                    metadata = event.get("metadata", {})
                    placements = metadata.get("placements", {})
                    colors_meta = metadata.get("colors", {})

                    for name, value in indicators.items():
                        placement = placements.get(name, "subplot")

                        # Store color if provided
                        if name not in colors and name in colors_meta:
                            colors[name] = colors_meta[name]

                        # Group by placement
                        if placement == "overlay":
                            if name not in overlay_indicators:
                                overlay_indicators[name] = []
                            overlay_indicators[name].append((timestamp, value))
                        else:  # subplot
                            if name not in subplot_indicators:
                                subplot_indicators[name] = []
                            subplot_indicators[name].append((timestamp, value))

                # Skip if no indicators to plot
                if not overlay_indicators and not subplot_indicators:
                    continue

                # Determine subplot configuration
                num_subplots = len(subplot_indicators)
                if num_subplots == 0:
                    # Only overlay indicators - no chart needed (they go on price chart)
                    continue

                # Create subplots for indicator charts
                subplot_titles = list(subplot_indicators.keys())
                fig = make_subplots(
                    rows=num_subplots,
                    cols=1,
                    subplot_titles=subplot_titles,
                    vertical_spacing=0.08,
                    shared_xaxes=True,
                )

                # Add subplot indicators
                for idx, (name, data) in enumerate(sorted(subplot_indicators.items()), start=1):
                    data_sorted = sorted(data, key=lambda x: x[0])
                    timestamps = [d[0] for d in data_sorted]
                    values = [d[1] for d in data_sorted]

                    color = colors.get(name, default_colors[(idx - 1) % len(default_colors)])

                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=values,
                            mode="lines",
                            name=name,
                            line=dict(width=2, color=color),
                            showlegend=False,
                        ),
                        row=idx,
                        col=1,
                    )

                    # Update y-axis for this subplot
                    fig.update_yaxes(title_text=name, row=idx, col=1)

                # Update x-axis for bottom subplot
                fig.update_xaxes(title_text="Date", row=num_subplots, col=1)

                # Update layout
                height = 200 * num_subplots
                fig.update_layout(
                    title="",
                    hovermode="x unified",
                    height=height,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    showlegend=False,
                    margin=dict(l=60, r=50, t=50, b=50),
                )

                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f0f0f0")
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f0f0f0")

                chart_html = fig.to_html(include_plotlyjs=False, div_id=f"indicator-chart-{symbol}")

                indicator_count = len(subplot_indicators)

                charts_html.append(
                    f"""
            <div class="chart-container">
                <div class="chart-title">📊 Technical Indicators · {symbol} ({indicator_count} indicator{"s" if indicator_count > 1 else ""})</div>
                {chart_html}
            </div>
            """
                )

            return "\n".join(charts_html) if charts_html else ""

        except Exception as e:
            logger.warning("html_reporter.indicator_charts.failed", error=str(e))
            return ""

    def _calculate_total_dividends(
        self,
        trades_by_id: dict[str, dict[str, dict[str, Any]]],
        equity_curve: pd.DataFrame | None,
        performance: dict[str, Any],
    ) -> float:
        """Calculate total dividends from equity accounting.

        Dividends = Final Equity - Initial Equity - Realized P&L from Trades - Unrealized P&L
        """
        # Get equity values
        initial_equity = float(performance.get("initial_equity", 100_000))
        final_equity = float(performance.get("final_equity", initial_equity))
        equity_change = final_equity - initial_equity

        # Calculate realized P&L from closed trades
        total_realized_pnl = sum(
            float(trade_events.get("closed", {}).get("realized_pnl", 0))
            for trade_events in trades_by_id.values()
            if "closed" in trade_events and trade_events["closed"].get("realized_pnl") is not None
        )

        # Calculate unrealized P&L from open positions
        total_unrealized_pnl = 0.0
        if equity_curve is not None and not equity_curve.empty:
            last_row = equity_curve.iloc[-1]
            current_market_value = float(last_row.get("positions_value", 0))

            # Sum entry costs of all open positions
            open_entry_cost = 0.0
            for trade_events in trades_by_id.values():
                if "open" in trade_events and "closed" not in trade_events:
                    open_trade = trade_events["open"]
                    qty = int(open_trade.get("current_quantity", 0))
                    entry_price = float(open_trade.get("entry_price", 0))
                    commission = float(open_trade.get("commission_total", 0))
                    open_entry_cost += qty * entry_price + commission

            total_unrealized_pnl = current_market_value - open_entry_cost

        # Dividends are the residual
        total_dividends = equity_change - total_realized_pnl - total_unrealized_pnl

        return total_dividends

    def _build_trades_table(
        self, trades: list[dict[str, Any]] | None, equity_curve: pd.DataFrame | None, performance: dict[str, Any]
    ) -> str:
        """Build trades table showing one row per trade (grouped by trade_id)."""
        if not trades:
            return ""

        # Group trades by trade_id, keeping both open and closed events
        # We need the open event to get original quantity
        from collections import defaultdict

        trades_by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

        for trade in trades:
            trade_id = trade.get("trade_id", "")
            status = trade.get("status", "")
            trades_by_id[trade_id][status] = trade

        # Sort by entry timestamp (use closed event if available, else open)
        sorted_trade_ids = sorted(
            trades_by_id.keys(),
            key=lambda tid: trades_by_id[tid]
            .get("closed", trades_by_id[tid].get("open", {}))
            .get("entry_timestamp", ""),
        )

        # Calculate total dividends from equity accounting
        total_dividends = self._calculate_total_dividends(trades_by_id, equity_curve, performance)

        rows = []
        total_unrealized_pnl = 0.0
        for trade_id in sorted_trade_ids:
            trade_events = trades_by_id[trade_id]

            # Use closed event if available, otherwise open
            trade = trade_events.get("closed", trade_events.get("open", {}))
            open_trade = trade_events.get("open", {})

            symbol = trade.get("symbol", "")
            strategy = trade.get("strategy_id", "")
            side = trade.get("side", "")
            status = trade.get("status", "")
            entry_price = float(trade.get("entry_price", 0))
            exit_price = trade.get("exit_price")
            commission = float(trade.get("commission_total", 0))
            realized_pnl = trade.get("realized_pnl")
            entry_time = trade.get("entry_timestamp", "")[:10] if trade.get("entry_timestamp") else ""
            exit_time = trade.get("exit_timestamp", "")[:10] if trade.get("exit_timestamp") else ""

            # Get original quantity from open event
            original_quantity = int(open_trade.get("current_quantity", 0))

            # Format entry/exit prices
            entry_str = f"${entry_price:.2f}"
            exit_str = f"${float(exit_price):.2f}" if exit_price else "—"

            # Calculate PnL
            if status == "closed" and realized_pnl is not None:
                pnl = float(realized_pnl)
                # Calculate percentage based on entry cost (entry_price × original_quantity)
                entry_cost = entry_price * original_quantity
                pnl_pct = (pnl / entry_cost) * 100 if entry_cost > 0 else 0
                pnl_str = f"${pnl:,.2f}"
                pnl_pct_str = f"({pnl_pct:+.2f}%)"
                pnl_class = "positive" if pnl >= 0 else "negative"
                status_badge = '<span class="badge success">Closed</span>'
                quantity_str = f"{original_quantity:,}"  # Show original quantity
            else:
                # Open trade - calculate mark-to-market unrealized P&L
                if equity_curve is not None and not equity_curve.empty:
                    last_row = equity_curve.iloc[-1]
                    current_market_value = float(last_row.get("positions_value", 0))

                    # Calculate current market price from market value and quantity
                    current_price = current_market_value / original_quantity if original_quantity > 0 else 0
                    exit_str = f"${current_price:.2f}*"  # Show current market price with asterisk

                    # Calculate MTM unrealized P&L
                    entry_cost = entry_price * original_quantity + commission
                    mtm_pnl = current_market_value - entry_cost
                    total_unrealized_pnl += mtm_pnl

                    pnl = mtm_pnl
                    pnl_pct = (pnl / (entry_price * original_quantity)) * 100 if original_quantity > 0 else 0
                    pnl_str = f"${pnl:,.2f}*"  # Add asterisk for unrealized/MTM
                else:
                    # Fallback if no equity curve data
                    pnl = -commission
                    pnl_pct = (pnl / (entry_price * original_quantity)) * 100 if original_quantity > 0 else 0
                    pnl_str = f"${pnl:,.2f}*"

                pnl_pct_str = f"({pnl_pct:+.2f}%)"
                pnl_class = "positive" if pnl >= 0 else "negative"
                status_badge = '<span class="badge warning">Open</span>'
                quantity_str = f"{original_quantity:,}"

            row = f"""
            <tr style="font-size: 0.85em;">
                <td>{trade_id}</td>
                <td><strong>{symbol}</strong></td>
                <td>{strategy}</td>
                <td>{status_badge}</td>
                <td style="text-align: center;">{side.upper()}</td>
                <td>{entry_time}</td>
                <td style="text-align: right;">{entry_str}</td>
                <td>{exit_time if exit_time else "—"}</td>
                <td style="text-align: right;">{exit_str}</td>
                <td style="text-align: right;">{quantity_str}</td>
                <td style="text-align: right;">${commission:.2f}</td>
                <td style="text-align: right;" class="{pnl_class}"><strong>{pnl_str}</strong> <small>{pnl_pct_str}</small></td>
            </tr>
            """
            rows.append(row)

        # Calculate total P&L from trades
        total_realized_pnl = sum(
            float(trade_events.get("closed", {}).get("realized_pnl", 0))
            for trade_events in trades_by_id.values()
            if "closed" in trade_events and trade_events["closed"].get("realized_pnl") is not None
        )

        # Add dividend row if dividends exist
        dividend_row = ""
        if abs(total_dividends) >= 0.01:
            div_class = "positive" if total_dividends >= 0 else "negative"
            dividend_row = f"""
            <tr style="font-size: 0.85em; background: #f0fdf4;">
                <td>—</td>
                <td><strong>DIVIDENDS</strong></td>
                <td colspan="8">Total dividend income received</td>
                <td style="text-align: right;">—</td>
                <td style="text-align: right;" class="{div_class}"><strong>${total_dividends:,.2f}</strong></td>
            </tr>
            """

        # Total P&L includes realized, unrealized, and dividends
        total_pnl = total_realized_pnl + total_unrealized_pnl + total_dividends

        # Get initial and final equity for comparison
        initial_equity = float(performance.get("initial_equity", 0))
        final_equity = float(performance.get("final_equity", 0))
        equity_change = final_equity - initial_equity

        # Determine if total matches
        pnl_matches = abs(total_pnl - equity_change) < 1.0  # Within $1 tolerance
        match_icon = "✓" if pnl_matches else "⚠"

        # Summary row
        total_pnl_class = "positive" if total_pnl >= 0 else "negative"

        # Build summary breakdown
        summary_parts = []
        if abs(total_realized_pnl) >= 0.01:
            summary_parts.append(f"Realized: ${total_realized_pnl:,.2f}")
        if abs(total_unrealized_pnl) >= 0.01:
            summary_parts.append(f"Unrealized: ${total_unrealized_pnl:,.2f}*")
        if abs(total_dividends) >= 0.01:
            summary_parts.append(f"Dividends: ${total_dividends:,.2f}")

        summary_details = (
            f"<br>Equity: ${initial_equity:,.0f} → ${final_equity:,.0f} ({match_icon}${equity_change:,.2f})"
        )
        if summary_parts:
            summary_details += "<br>" + " + ".join(summary_parts)

        summary_row = f"""
            <tr style="font-weight: bold; border-top: 2px solid #667eea; background: #f9fafb;">
                <td colspan="11" style="text-align: right; padding-right: 1rem;">Total P&L:</td>
                <td style="text-align: right;" class="{total_pnl_class}">
                    <strong>${total_pnl:,.2f}</strong>
                    <small style="color: #6b7280; font-weight: normal;">
                        {summary_details}
                    </small>
                </td>
            </tr>
        """

        # Add footnote if there are unrealized P&L values
        footnote = ""
        if abs(total_unrealized_pnl) >= 1.0:
            footnote = '<p style="font-size: 0.85em; color: #6b7280; margin-top: 0.5rem; font-style: italic;">* Mark-to-market valuation at end of backtest period</p>'

        return f"""
        <div class="chart-container">
            <div class="chart-title">💼 Trades</div>
            <table style="font-size: 0.9em;">
                <thead>
                    <tr>
                        <th>Trade ID</th>
                        <th>Symbol</th>
                        <th>Strategy</th>
                        <th>Status</th>
                        <th style="text-align: center;">Side</th>
                        <th>Entry Date</th>
                        <th style="text-align: right;">Entry Price</th>
                        <th>Exit Date</th>
                        <th style="text-align: right;">Exit Price</th>
                        <th style="text-align: right;">Quantity</th>
                        <th style="text-align: right;">Commission</th>
                        <th style="text-align: right;">P&L</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                    {dividend_row}
                    {summary_row}
                </tbody>
            </table>
            {footnote}
        </div>
        """
