from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_SLUG = "pgx-analysis"
NOTEBOOK_METADATA_BUCKET = os.environ.get("PGX_NOTEBOOK_METADATA_BUCKET", "mushin-solutions-project-metadata")
NOTEBOOK_METADATA_PREFIX = os.environ.get("PGX_NOTEBOOK_METADATA_PREFIX", "notebooks")
DATALAKE_BUCKET = os.environ.get("PGX_DATALAKE_BUCKET", "pgxdatalake")
GITHUB_ARTIFACT_ROOT = os.environ.get("PGX_GITHUB_ARTIFACT_ROOT", "reports/notebook_artifacts")


@dataclass(frozen=True)
class NotebookArtifactContext:
    project_root: Path
    notebook_path: Path
    notebook_name: str
    step_dir: Path
    github_dir: Path
    local_output_dir: Path
    s3_metadata_prefix: str
    s3_artifact_prefix: str
    datalake_bucket: str = DATALAKE_BUCKET
    metadata_bucket: str = NOTEBOOK_METADATA_BUCKET

    def manifest_path(self) -> Path:
        return self.local_output_dir / "artifact_manifest.json"

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("project_root", "notebook_path", "step_dir", "github_dir", "local_output_dir"):
            data[key] = str(data[key])
        return data


def find_project_root(start: Path | str | None = None) -> Path:
    path = Path(start or Path.cwd()).resolve()
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists() or (candidate / "py_helpers").exists():
            return candidate
    return path


def infer_step_dir(project_root: Path, notebook_path: Path) -> Path:
    try:
        rel = notebook_path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return project_root
    if len(rel.parts) > 1:
        return project_root / rel.parts[0]
    return project_root


def setup_notebook_artifacts(
    notebook_file: str | Path,
    step_name: str | None = None,
    run_label: str | None = None,
    create_dirs: bool = True,
) -> NotebookArtifactContext:
    project_root = find_project_root()
    notebook_path = Path(notebook_file).resolve()
    notebook_name = notebook_path.stem
    step_dir = project_root / step_name if step_name else infer_step_dir(project_root, notebook_path)
    step_slug = step_dir.name if step_dir != project_root else "root"
    safe_run_label = run_label or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    github_dir = project_root / GITHUB_ARTIFACT_ROOT / step_slug / notebook_name
    local_output_dir = step_dir / "outputs" / "notebook_artifacts" / notebook_name / safe_run_label
    s3_metadata_prefix = f"{NOTEBOOK_METADATA_PREFIX}/{PROJECT_SLUG}/{step_slug}/{notebook_name}/"
    s3_artifact_prefix = f"gold/notebook_artifacts/{step_slug}/{notebook_name}/{safe_run_label}/"

    if create_dirs:
        github_dir.mkdir(parents=True, exist_ok=True)
        local_output_dir.mkdir(parents=True, exist_ok=True)

    context = NotebookArtifactContext(
        project_root=project_root,
        notebook_path=notebook_path,
        notebook_name=notebook_name,
        step_dir=step_dir,
        github_dir=github_dir,
        local_output_dir=local_output_dir,
        s3_metadata_prefix=s3_metadata_prefix,
        s3_artifact_prefix=s3_artifact_prefix,
    )
    write_artifact_manifest(context)
    return context


def write_artifact_manifest(context: NotebookArtifactContext, extra: dict[str, Any] | None = None) -> Path:
    payload = context.as_dict()
    payload["s3_metadata_uri"] = f"s3://{context.metadata_bucket}/{context.s3_metadata_prefix}"
    payload["s3_artifact_uri"] = f"s3://{context.datalake_bucket}/{context.s3_artifact_prefix}"
    payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    if extra:
        payload.update(extra)
    path = context.manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def s3_artifact_uri(context: NotebookArtifactContext, filename: str) -> str:
    return f"s3://{context.datalake_bucket}/{context.s3_artifact_prefix}{filename.lstrip('/')}"


def github_artifact_path(context: NotebookArtifactContext, filename: str) -> Path:
    context.github_dir.mkdir(parents=True, exist_ok=True)
    return context.github_dir / filename


def local_artifact_path(context: NotebookArtifactContext, filename: str) -> Path:
    context.local_output_dir.mkdir(parents=True, exist_ok=True)
    return context.local_output_dir / filename
