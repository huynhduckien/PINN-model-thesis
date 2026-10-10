import json

from lpbfpinn.config import Material
from lpbfpinn.manifest import write_manifest


def test_manifest_roundtrip(tmp_path):
    p = write_manifest(tmp_path / "m.json", "unit", Material(), extra={"k": 1})
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["run"] == "unit"
    assert data["config"]["T_sol"] == 1878.0
    assert data["extra"] == {"k": 1}
    assert "created_utc" in data
