"""
Experiment Management Models and Utilities.

Provides experiment-centric organization for backtests:
- Experiments: Logical groupings of related runs
- Runs: Individual backtest executions with metadata
- Experiment Resolution: Directory → YAML lookup
"""

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class GitInfo:
    """Git repository information at time of run."""

    commit: Optional[str] = None
    branch: Optional[str] = None
    dirty: bool = False
    diff_files_count: int = 0


@dataclass
class EnvironmentInfo:
    """Python environment information."""

    python_version: str
    qtrader_version: str
    packages: dict[str, str] = field(default_factory=dict)


@dataclass
class RunMetadata:
    """Metadata for a single experiment run."""

    experiment_id: str
    run_id: str
    started_at: str  # ISO format timestamp
    finished_at: Optional[str] = None
    status: str = "running"  # running, success, failed, partial
    config_sha256: str = ""
    git: Optional[GitInfo] = None
    environment: Optional[EnvironmentInfo] = None
    metrics: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunMetadata":
        """Create from dictionary."""
        # Handle nested dataclasses
        if data.get("git"):
            data["git"] = GitInfo(**data["git"])
        if data.get("environment"):
            data["environment"] = EnvironmentInfo(**data["environment"])
        return cls(**data)


class ExperimentResolver:
    """
    Resolves experiment directories to configuration files.

    Handles:
    - Directory path → YAML file resolution
    - YAML file validation
    - Experiment ID extraction and validation
    """

    @staticmethod
    def resolve_config_path(path: Path) -> Path:
        """
        Resolve a path (directory or file) to a backtest configuration file.

        If path is a directory:
            1. Look for {directory_name}.yaml
            2. If not found, error

        If path is a file:
            Return as-is

        Args:
            path: Directory or file path

        Returns:
            Path to config file

        Raises:
            ValueError: If resolution fails
        """
        path = path.resolve()

        # If it's a file, return it
        if path.is_file():
            return path

        # If it's a directory, look for canonical YAML
        if path.is_dir():
            experiment_id = path.name
            canonical_config = path / f"{experiment_id}.yaml"

            if canonical_config.exists():
                return canonical_config

            # Check for common alternatives
            alternatives = [
                path / "config.yaml",
                path / "backtest.yaml",
                path / "experiment.yaml",
            ]

            existing_yamls = [p for p in alternatives if p.exists()]

            if existing_yamls:
                raise ValueError(
                    f"Experiment directory '{path}' must contain '{experiment_id}.yaml' (canonical). "
                    f"Found: {[p.name for p in existing_yamls]}. "
                    f"Rename one to '{experiment_id}.yaml' or pass file path directly."
                )

            raise ValueError(
                f"No configuration file found in experiment directory: {path}\n"
                f"Expected: {canonical_config}\n"
                f"Tip: Create {experiment_id}.yaml in this directory"
            )

        raise ValueError(f"Path does not exist: {path}")

    @staticmethod
    def validate_experiment_structure(experiment_dir: Path, config_path: Path) -> None:
        """
        Validate experiment directory structure.

        Ensures:
        - Directory name matches config basename
        - Config is inside the experiment directory

        Args:
            experiment_dir: Experiment directory path
            config_path: Configuration file path

        Raises:
            ValueError: If validation fails
        """
        experiment_dir = experiment_dir.resolve()
        config_path = config_path.resolve()

        # Ensure config is inside experiment dir
        try:
            config_path.relative_to(experiment_dir)
        except ValueError:
            raise ValueError(
                f"Configuration file must be inside experiment directory.\n"
                f"Config: {config_path}\n"
                f"Experiment dir: {experiment_dir}"
            )

        # Validate naming convention (directory matches config basename)
        expected_basename = experiment_dir.name
        config_basename = config_path.stem

        if expected_basename != config_basename:
            raise ValueError(
                f"Experiment naming mismatch:\n"
                f"Directory name: {expected_basename}\n"
                f"Config basename: {config_basename}\n"
                f"Tip: Rename config to '{expected_basename}.yaml' or rename directory"
            )

    @staticmethod
    def get_experiment_dir(config_path: Path) -> Path:
        """
        Get experiment directory from config path.

        Assumes config is at: experiments/{experiment_id}/{experiment_id}.yaml

        Args:
            config_path: Path to config file

        Returns:
            Experiment directory path
        """
        return config_path.parent

    @staticmethod
    def create_run_dir(experiment_dir: Path, run_id: str) -> Path:
        """
        Create run directory for this experiment execution.

        Creates: {experiment_dir}/runs/{run_id}/

        Args:
            experiment_dir: Experiment directory
            run_id: Run identifier (typically timestamp)

        Returns:
            Created run directory path
        """
        run_dir = experiment_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @staticmethod
    def generate_run_id(timestamp_format: str = "%Y%m%d_%H%M%S") -> str:
        """
        Generate unique run ID based on current timestamp.

        If a run with this ID already exists, appends a counter suffix.

        Args:
            timestamp_format: Timestamp format string

        Returns:
            Unique run ID
        """
        base_id = datetime.now().strftime(timestamp_format)
        return base_id


class ExperimentMetadata:
    """Helper for working with experiment metadata."""

    @staticmethod
    def compute_config_hash(config_path: Path) -> str:
        """
        Compute SHA256 hash of configuration file.

        Args:
            config_path: Path to config file

        Returns:
            Hex-encoded SHA256 hash
        """
        with open(config_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    @staticmethod
    def capture_git_info(repo_path: Optional[Path] = None) -> Optional[GitInfo]:
        """
        Capture git repository information.

        Args:
            repo_path: Path to git repository (default: current directory)

        Returns:
            GitInfo or None if not a git repo
        """
        try:
            cwd = str(repo_path) if repo_path else None

            # Get commit hash
            commit_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=cwd,
            )
            commit = commit_result.stdout.strip()

            # Get branch name
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=cwd,
            )
            branch = branch_result.stdout.strip()

            # Check for uncommitted changes
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                cwd=cwd,
            )
            dirty = bool(status_result.stdout.strip())

            # Count modified files
            diff_files = len([line for line in status_result.stdout.splitlines() if line.strip()])

            return GitInfo(
                commit=commit,
                branch=branch,
                dirty=dirty,
                diff_files_count=diff_files,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    @staticmethod
    def capture_environment() -> EnvironmentInfo:
        """
        Capture Python environment information.

        Returns:
            EnvironmentInfo with version details
        """
        import sys

        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        # Get qtrader version
        try:
            import qtrader

            qtrader_version = getattr(qtrader, "__version__", "unknown")
        except Exception:
            qtrader_version = "unknown"

        # Optionally capture key package versions
        packages = {}
        try:
            import importlib.metadata

            for pkg in ["numpy", "pandas", "polars", "pydantic", "rich"]:
                try:
                    packages[pkg] = importlib.metadata.version(pkg)
                except importlib.metadata.PackageNotFoundError:
                    pass
        except Exception:
            pass

        return EnvironmentInfo(
            python_version=python_version,
            qtrader_version=qtrader_version,
            packages=packages,
        )

    @staticmethod
    def write_run_metadata(run_dir: Path, metadata: RunMetadata) -> Path:
        """
        Write run metadata to manifest file.

        Args:
            run_dir: Run directory
            metadata: Run metadata

        Returns:
            Path to manifest file
        """
        manifest_path = run_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
        return manifest_path

    @staticmethod
    def read_run_metadata(run_dir: Path) -> Optional[RunMetadata]:
        """
        Read run metadata from manifest file.

        Args:
            run_dir: Run directory

        Returns:
            RunMetadata or None if not found
        """
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            return None

        with open(manifest_path) as f:
            data = json.load(f)
        return RunMetadata.from_dict(data)

    @staticmethod
    def save_config_snapshot(config_path: Path, run_dir: Path) -> Path:
        """
        Save a snapshot of the configuration file in the run directory.

        Args:
            config_path: Original config path
            run_dir: Run directory

        Returns:
            Path to config snapshot
        """
        import shutil

        snapshot_path = run_dir / "config_snapshot.yaml"
        shutil.copy2(config_path, snapshot_path)
        return snapshot_path

    @staticmethod
    def create_latest_symlink(experiment_dir: Path, run_id: str) -> Optional[Path]:
        """
        Create or update 'latest' symlink pointing to most recent run.

        Args:
            experiment_dir: Experiment directory
            run_id: Run ID

        Returns:
            Path to symlink or None if creation failed
        """
        try:
            runs_dir = experiment_dir / "runs"
            latest_link = runs_dir / "latest"
            target = Path(run_id)  # Relative path

            # Remove existing symlink/file
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()

            # Create new symlink
            latest_link.symlink_to(target)
            return latest_link
        except Exception as e:
            logger.warning("experiment.symlink_failed", error=str(e))
            return None
