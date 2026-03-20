from __future__ import annotations

import json
from pathlib import Path


def _md_to_comment_lines(md: str) -> list[str]:
    lines = md.splitlines()
    if not lines:
        return ["#"]
    return [("# " + l) if l.strip() else "#" for l in lines]


def convert_notebook_to_vscode_script(nb_path: Path, out_path: Path) -> None:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    parts: list[str] = []
    parts.append("# -*- coding: utf-8 -*-")
    parts.append(f"# Auto-generated from {nb_path.name} (VS Code Python notebook script format)")
    parts.append("")

    for cell in nb.get("cells", []):
        ctype = cell.get("cell_type")
        src = "".join(cell.get("source", []))

        if ctype == "markdown":
            parts.append("# %% [markdown]")
            parts.extend(_md_to_comment_lines(src))
            parts.append("")
        elif ctype == "code":
            parts.append("# %%")
            parts.append(src.rstrip("\n"))
            parts.append("")
        else:
            parts.append("# %%")
            parts.append(f"# [Unsupported cell_type: {ctype}]")
            parts.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "py_helpers" / "vs_code_jupyter_notebook_scripts"
    out_dir.mkdir(parents=True, exist_ok=True)

    notebooks = [
        repo_root / "0_config_and_pipeline.ipynb",
        repo_root / "1_cohort_workflow.ipynb",
        repo_root / "2_feature_importance.ipynb",
        repo_root / "3_model_train_shap_ffa.ipynb",
        repo_root / "4_dashboard_visuals.ipynb",
        repo_root / "5_build_and_deploy.ipynb",
    ]

    for nb_path in notebooks:
        if not nb_path.exists():
            raise FileNotFoundError(f"Notebook not found: {nb_path}")
        out_path = out_dir / f"{nb_path.stem}.py"
        convert_notebook_to_vscode_script(nb_path, out_path)
        print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
