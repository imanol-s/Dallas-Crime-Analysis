"""Typer CLI entrypoint for the Dallas crime analysis project."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Annotated

import typer

from dallas_crime.acquire.utils import AcquisitionError
from dallas_crime.config import Settings


app = typer.Typer(help="Dallas crime and housing analysis pipeline.")


def _settings(project_root: Path | None) -> Settings:
    settings = Settings.from_env(project_root=project_root)
    settings.ensure_directories()
    return settings


def _echo_settings(settings: Settings) -> None:
    typer.echo("Configuration")
    for key, value in settings.describe().items():
        typer.echo(f"  {key}: {value}")


def _run_hook(
    module_name: str, function_name: str, settings: Settings
) -> dict[str, Path] | Path | None:
    try:
        module = import_module(module_name)
    except ModuleNotFoundError:
        typer.echo(
            f"Scaffold only: {module_name} is not available yet. "
            "Add the downstream pipeline modules to enable this command."
        )
        return None

    handler = getattr(module, function_name, None)
    if handler is None:
        typer.echo(f"Scaffold only: {module_name}.{function_name} is not implemented yet.")
        return None
    return handler(settings)


@app.command()
def acquire(
    project_root: Annotated[
        Path | None,
        typer.Option(
            "--project-root",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Override the project root used for data and report paths.",
        ),
    ] = None,
    dry_run: Annotated[bool, typer.Option(help="Print actions without fetching data.")] = False,
) -> None:
    """Fetch raw source data for the project."""

    settings = _settings(project_root)
    if dry_run:
        _echo_settings(settings)
        typer.echo("Acquire dry run complete.")
        return

    _echo_settings(settings)
    try:
        result = _run_hook("dallas_crime.acquire", "run_acquire", settings)
    except AcquisitionError as exc:
        typer.secho(f"Acquire failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    if isinstance(result, dict):
        for label, path in result.items():
            typer.echo(f"{label}: {path}")
    elif result is not None:
        typer.echo(str(result))


@app.command()
def build(
    project_root: Annotated[
        Path | None,
        typer.Option(
            "--project-root",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Override the project root used for data and report paths.",
        ),
    ] = None,
) -> None:
    """Build processed datasets from raw sources."""

    settings = _settings(project_root)
    _echo_settings(settings)
    outputs = _run_hook("dallas_crime.pipeline.build", "build_all", settings)
    if isinstance(outputs, dict):
        for label, path in outputs.items():
            typer.echo(f"{label}: {path}")
    elif outputs is not None:
        typer.echo(str(outputs))


@app.command()
def analyze(
    project_root: Annotated[
        Path | None,
        typer.Option(
            "--project-root",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Override the project root used for data and report paths.",
        ),
    ] = None,
) -> None:
    """Run the statistical analysis and write report artifacts."""

    settings = _settings(project_root)
    _echo_settings(settings)
    outputs = _run_hook("dallas_crime.pipeline.analyze", "run_analysis", settings)
    if isinstance(outputs, dict):
        for label, path in outputs.items():
            typer.echo(f"{label}: {path}")
    elif outputs is not None:
        typer.echo(str(outputs))


@app.command("show-config")
def show_config(
    project_root: Annotated[
        Path | None,
        typer.Option(
            "--project-root",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Override the project root used for data and report paths.",
        ),
    ] = None,
) -> None:
    """Print the resolved project configuration."""

    _echo_settings(_settings(project_root))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
