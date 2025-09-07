from pathlib import Path

_p = Path(__file__)

repo_dir = _p.parent.parent.parent
data_dir = repo_dir / "data"
dist_data_dir = repo_dir / "dist_data"
