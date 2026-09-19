from railguard.project_paths import participant_root


def test_participant_root_finds_workspace_data(tmp_path, monkeypatch):
    monkeypatch.delenv("RAILGUARD_PARTICIPANT_ROOT", raising=False)
    app = tmp_path / "cluelessbunch/app"
    app.mkdir(parents=True)
    dataset = tmp_path / "NebulaX-Hackathon-ProblemStatement/PS3"
    dataset.mkdir(parents=True)
    assert participant_root(app) == dataset
    local = app / "NebulaX-Hackathon-ProblemStatement/PS3"
    local.mkdir(parents=True)
    assert participant_root(app) == local
    monkeypatch.setenv("RAILGUARD_PARTICIPANT_ROOT", str(dataset))
    assert participant_root(app) == dataset
