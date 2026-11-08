from pathlib import Path
from agentic_app.core import config


def test_legacy_parent_setting_falls_back_to_sibling_data_folder(tmp_path, monkeypatch):
    project_parent = tmp_path / "agentic_platform_updated"
    platform_root = project_parent / "agentic_platform"
    data_dir = project_parent / "data"
    platform_root.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    (data_dir / "test_operational_readouts.csv").write_text("vehicle_id,time_step\n", encoding="utf-8")
    monkeypatch.setattr(config, "PLATFORM_ROOT", platform_root)

    settings = config.Settings.from_env(environ={"SCANIA_DATA_DIR": str(project_parent)})

    assert settings.data_dir == data_dir.resolve()


def test_default_prefers_sibling_data_folder_when_it_contains_splits(tmp_path, monkeypatch):
    project_parent = tmp_path / "agentic_platform_updated"
    platform_root = project_parent / "agentic_platform"
    data_dir = project_parent / "data"
    platform_root.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    (data_dir / "validation_operational_readouts.csv").write_text("vehicle_id,time_step\n", encoding="utf-8")
    monkeypatch.setattr(config, "PLATFORM_ROOT", platform_root)

    settings = config.Settings.from_env(environ={})

    assert settings.data_dir == data_dir.resolve()
