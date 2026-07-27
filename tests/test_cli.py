"""The grl-snam CLI wiring: commands registered, help works, demos resolvable."""

from click.testing import CliRunner

from grl_snam import demos
from grl_snam.cli import main


def test_cli_help_lists_pipeline():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "pipeline" in result.output


def test_cli_registers_core_commands():
    names = set(main.commands)
    for cmd in (
        "selftest",
        "obstacles",
        "build-sdf",
        "train",
        "capture",
        "pipeline",
        "demo",
        "lab-demo",
        "eval",
        "train-coef",
    ):
        assert cmd in names, f"missing command: {cmd}"


def test_capture_subcommands():
    assert {"drive", "multigoal"} <= set(main.commands["capture"].commands)


def test_demo_registry_and_paths_resolve():
    reg = demos.registry()
    assert "austin-freedrive" in reg and "lab" in reg
    for name in reg:
        path = demos.demo_path(name)
        assert path and path.endswith(".py")


def test_demo_path_unknown_is_none():
    assert demos.demo_path("does-not-exist") is None
