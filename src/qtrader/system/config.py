"""
System Configuration Management.

Minimal configuration for QTrader engine focused on DataService.
Configs are added incrementally as services are refactored.

Currently supports:
- DataService configuration (HOW to handle data system-wide)
- Event store / output directory
- Logging configuration

Note: This is different from BacktestConfig.DataSelectionConfig which
specifies WHAT data to load for a specific backtest run.

Usage:
    >>> from qtrader.system import get_system_config
    >>> config = get_system_config()
    >>> print(config.data.sources_config)
    config/data_sources.yaml
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import yaml


@dataclass
class DataServiceConfig:
    """Data service configuration.

    System-wide configuration for HOW DataService handles data.
    Specifies where to find data sources and basic data handling preferences.
    """

    sources_config: str = "config/data_sources.yaml"
    default_timezone: str = "America/New_York"
    price_decimals: int = 4
    validate_on_load: bool = True


@dataclass
class EventStoreConfig:
    """Event store configuration."""

    backend: Literal["sqlite", "parquet", "memory"] = "memory"
    filename: str = "events.{backend}"


@dataclass
class OutputConfig:
    """Output and results configuration for experiment-based organization.

    Directory structure: experiments/{experiment_id}/runs/{run_id}/
    Example: experiments/momentum_strategy/runs/20251119_143022/

    The run_id is generated using run_id_format (timestamp-based by default).
    Each run is isolated with its own directory containing all artifacts.
    """

    experiments_root: str = "experiments"
    run_id_format: str = "%Y%m%d_%H%M%S"
    display_format: Literal["line", "table"] = "line"
    event_store: EventStoreConfig = field(default_factory=EventStoreConfig)

    # Metadata capture toggles
    capture_git_info: bool = True
    capture_environment: bool = True


@dataclass
class CustomLibrariesConfig:
    """Custom library paths configuration.

    Specifies directories to scan for custom implementations of:
    - Data adapters
    - Strategies
    - Risk policies
    - Indicators
    - Metrics

    Set to None to skip custom discovery and use only built-in components.
    When set to a path, system will auto-discover and register implementations.

    Example:
        # Use built-in components only (safe for pip install)
        CustomLibrariesConfig(adapters=None, strategies=None, ...)

        # Point to custom libraries
        CustomLibrariesConfig(
            adapters="~/qtrader-custom/adapters",
            strategies="~/qtrader-custom/strategies",
            ...
        )
    """

    adapters: Optional[str] = None
    strategies: Optional[str] = None
    risk_policies: Optional[str] = None
    indicators: Optional[str] = None
    metrics: Optional[str] = None


@dataclass
class LoggingConfig:
    """Logging configuration (maps to log_system.LoggingConfig)."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["console", "json"] = "console"
    timestamp_format: Literal["iso", "compact", "time", "short"] = "compact"
    enable_file: bool = True
    file_path: str = "logs/qtrader.log"
    file_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING"
    file_rotation: bool = True
    max_file_size_mb: int = 10
    backup_count: int = 3
    console_width: int = 0

    def to_logger_config(self):
        """Convert to log_system.LoggingConfig for LoggerFactory."""
        from qtrader.system.log_system import LoggingConfig as LogSystemConfig

        return LogSystemConfig(
            level=self.level,
            format=self.format,
            timestamp_format=self.timestamp_format,
            enable_file=self.enable_file,
            file_path=Path(self.file_path) if self.file_path else None,
            file_level=self.file_level,
            file_rotation=self.file_rotation,
            max_file_size_mb=self.max_file_size_mb,
            backup_count=self.backup_count,
            console_width=self.console_width,
        )


@dataclass
class SystemConfig:
    """
    Minimal system configuration for QTrader.

    Currently supports DataService-only engine refactor.
    Additional service configs will be added incrementally.

    What's defined here (current):
        - Data service configuration (HOW to handle data)
        - Output/results directory
        - Logging settings
        - Custom library paths

    What will be added later (as services are refactored):
        - Portfolio service config
        - Execution service config
        - Risk service config
        - Strategy service config

    Example:
        >>> config = SystemConfig.load()
        >>> print(config.data.sources_config)
        config/data_sources.yaml
    """

    data: DataServiceConfig = field(default_factory=DataServiceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    custom_libraries: CustomLibrariesConfig = field(default_factory=CustomLibrariesConfig)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "SystemConfig":
        """
        Load system configuration from YAML file.

        Configuration search order:
        1. Explicit config_path if provided
        2. ./config/qtrader.yaml (project-relative)
        3. ./config/system.yaml (legacy fallback)
        4. ~/.qtrader/qtrader.yaml (user home)
        5. Built-in defaults (if no files found)

        Args:
            config_path: Explicit path to config file (overrides search)

        Returns:
            SystemConfig instance with loaded settings

        Example:
            >>> # Load from default location
            >>> config = SystemConfig.load()
            >>>
            >>> # Load from specific file
            >>> config = SystemConfig.load(Path("my_config.yaml"))
        """
        config_dict: dict[str, Any] = {}

        # Search for config files
        search_paths: list[Path] = []

        if config_path:
            search_paths.append(config_path)
        else:
            # Project-relative (primary location)
            project_config = Path("config/qtrader.yaml")
            if project_config.exists():
                search_paths.append(project_config)

            # Project-relative (legacy fallback)
            legacy_config = Path("config/system.yaml")
            if legacy_config.exists():
                search_paths.append(legacy_config)

            # User home
            home_config = Path.home() / ".qtrader" / "qtrader.yaml"
            if home_config.exists():
                search_paths.append(home_config)

        # Load config files (later files override earlier)
        for path in search_paths:
            if path.exists():
                with open(path) as f:
                    loaded = yaml.safe_load(f)
                    if loaded:
                        config_dict = _deep_merge(config_dict, loaded)

        # Substitute environment variables
        config_dict = _substitute_env_vars(config_dict)

        # Build config from dict
        return cls._from_dict(config_dict)

    @classmethod
    def _from_dict(cls, config_dict: dict[str, Any]) -> "SystemConfig":
        """Build SystemConfig from nested dictionary."""
        # Data service configuration
        data_dict = config_dict.get("data", {})
        data = DataServiceConfig(
            sources_config=data_dict.get("sources_config", "config/data_sources.yaml"),
            default_timezone=data_dict.get("default_timezone", "America/New_York"),
            price_decimals=data_dict.get("price_decimals", 4),
            validate_on_load=data_dict.get("validate_on_load", True),
        )

        # Output
        output_dict = config_dict.get("output", {})

        # Event store configuration
        event_store_dict = output_dict.get("event_store", {})
        event_store = EventStoreConfig(
            backend=event_store_dict.get("backend", "memory"),
            filename=event_store_dict.get("filename", "events.{backend}"),
        )

        output = OutputConfig(
            experiments_root=output_dict.get("experiments_root", "experiments"),
            run_id_format=output_dict.get("run_id_format", "%Y%m%d_%H%M%S"),
            display_format=output_dict.get("display_format", "line"),
            event_store=event_store,
            capture_git_info=output_dict.get("capture_git_info", True),
            capture_environment=output_dict.get("capture_environment", True),
        )

        # Logging
        logging_dict = config_dict.get("logging", {})
        logging = LoggingConfig(
            level=logging_dict.get("level", "INFO"),
            format=logging_dict.get("format", "console"),
            timestamp_format=logging_dict.get("timestamp_format", "compact"),
            enable_file=logging_dict.get("enable_file", True),
            file_path=logging_dict.get("file_path") or "logs/qtrader.log",
            file_level=logging_dict.get("file_level", "WARNING"),
            file_rotation=logging_dict.get("file_rotation", True),
            max_file_size_mb=logging_dict.get("max_file_size_mb", 10),
            backup_count=logging_dict.get("backup_count", 3),
            console_width=logging_dict.get("console_width", 0),
        )

        # Custom libraries
        custom_libraries_dict = config_dict.get("custom_libraries", {})

        # Helper to expand ~ and handle None/null
        def _expand_path(path_value: Any) -> Optional[str]:
            if path_value is None or path_value == "null":
                return None
            path_str = str(path_value).strip()
            if not path_str or path_str.lower() == "none":
                return None
            # Expand ~ to home directory
            return os.path.expanduser(path_str)

        custom_libraries = CustomLibrariesConfig(
            adapters=_expand_path(custom_libraries_dict.get("adapters")),
            strategies=_expand_path(custom_libraries_dict.get("strategies")),
            risk_policies=_expand_path(custom_libraries_dict.get("risk_policies")),
            indicators=_expand_path(custom_libraries_dict.get("indicators")),
            metrics=_expand_path(custom_libraries_dict.get("metrics")),
        )

        return cls(
            data=data,
            output=output,
            logging=logging,
            custom_libraries=custom_libraries,
        )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def _substitute_env_vars(config: Any) -> Any:
    """Recursively substitute ${VAR_NAME} with environment variables."""
    import re

    if isinstance(config, dict):
        return {k: _substitute_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_substitute_env_vars(item) for item in config]
    elif isinstance(config, str):
        pattern = r"\$\{([^}]+)\}"

        def replace_var(match: re.Match[str]) -> str:
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        return re.sub(pattern, replace_var, config)
    else:
        return config


# Global config instance (loaded on first access)
_config: Optional[SystemConfig] = None


def get_system_config(config_path: Optional[Path] = None) -> SystemConfig:
    """
    Get the system configuration singleton.

    Loads config on first access and caches for subsequent calls.
    Use reload_system_config() to force reload.

    Args:
        config_path: Optional explicit path to config file

    Returns:
        SystemConfig instance

    Example:
        >>> config = get_system_config()
        >>> print(config.execution.commission.per_share)
        0.0005
    """
    global _config
    if _config is None or config_path is not None:
        _config = SystemConfig.load(config_path)
    return _config


def reload_system_config(config_path: Optional[Path] = None) -> SystemConfig:
    """
    Reload system configuration from files.

    Forces a fresh load of configuration, clearing the cached singleton.

    Args:
        config_path: Optional explicit path to config file

    Returns:
        Reloaded SystemConfig instance

    Example:
        >>> # Modify config file
        >>> config = reload_system_config()
        >>> # Now uses new settings
    """
    global _config
    _config = SystemConfig.load(config_path)
    return _config
