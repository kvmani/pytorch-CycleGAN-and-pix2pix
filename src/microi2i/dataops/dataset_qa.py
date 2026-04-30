"""Dataset quality assurance for microscopy image translation workflows."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
from pathlib import Path
import re
from typing import Any

from PIL import Image, UnidentifiedImageError

from microi2i.core.contracts import DatasetQAConfig, LeakageGroupPolicy


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _repo_path(value: str | Path, *, repo_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def _image_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"source root does not exist: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _leakage_group(path: Path, root: Path, policy: LeakageGroupPolicy) -> str:
    relative = path.relative_to(root)
    text = str(relative).replace("\\", "/")
    if policy.regex:
        match = re.search(policy.regex, text)
        if match:
            if match.groupdict():
                return next(iter(match.groupdict().values()))
            return match.group(1) if match.groups() else match.group(0)
        if policy.required:
            return ""
    if policy.mode in {"parent", "folder_or_specimen_id"}:
        return relative.parts[0] if len(relative.parts) > 1 else path.stem
    if policy.mode == "stem":
        return path.stem
    return path.stem


def _inspect_image(path: Path, root: Path, policy: LeakageGroupPolicy) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": str(path),
        "relative_path": str(path.relative_to(root)),
        "suffix": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
        "sha256": "",
        "readable": False,
        "width": None,
        "height": None,
        "mode": "",
        "channels": None,
        "leakage_group": _leakage_group(path, root, policy),
        "errors": [],
    }
    if not row["leakage_group"]:
        row["errors"].append("missing_leakage_group")
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            row["readable"] = True
            row["width"], row["height"] = img.size
            row["mode"] = img.mode
            row["channels"] = len(img.getbands())
        row["sha256"] = _sha256(path)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        row["errors"].append(f"unreadable_image: {exc}")
    return row


def _contact_sheet(samples: list[dict[str, Any]], output_path: Path, *, max_images: int, thumb_size: int) -> dict[str, Any]:
    readable = [row for row in samples if row["readable"]][:max_images]
    if not readable:
        return {"status": "skipped", "reason": "no readable images"}
    cols = min(5, max(1, len(readable)))
    rows = (len(readable) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_size, rows * thumb_size), "white")
    for index, row in enumerate(readable):
        with Image.open(row["path"]) as img:
            thumb = img.convert("RGB")
            thumb.thumbnail((thumb_size, thumb_size))
            x = (index % cols) * thumb_size + (thumb_size - thumb.width) // 2
            y = (index // cols) * thumb_size + (thumb_size - thumb.height) // 2
            sheet.paste(thumb, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return {"status": "written", "path": str(output_path), "image_count": len(readable)}


def _write_html_report(report: dict[str, Any], output_path: Path) -> None:
    summary = report["summary"]
    contact = report.get("contact_sheet", {})
    contact_html = ""
    if contact.get("status") == "written":
        contact_html = f"<h2>Contact Sheet</h2><img src='{Path(contact['path']).name}' style='max-width:100%;border:1px solid #ccd;border-radius:10px;'>"
    output_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>MicroI2I Dataset QA</title>"
        "<style>body{font-family:Segoe UI,Arial,sans-serif;max-width:980px;margin:2rem auto;line-height:1.5}"
        "code,pre{background:#f4f7fa;padding:.2rem .35rem;border-radius:4px}"
        ".ok{color:#146c43}.fail{color:#b42318}</style></head><body>"
        "<h1>MicroI2I Dataset QA Report</h1>"
        f"<p>Status: <strong class=\"{'ok' if report['status'] == 'passed' else 'fail'}\">{report['status']}</strong></p>"
        f"<p>Dataset: <code>{report['dataset_id']}</code> | Task: <code>{report['task_type']}</code></p>"
        "<h2>Summary</h2><ul>"
        f"<li>Total images: {summary['total_images']}</li>"
        f"<li>Readable images: {summary['readable_images']}</li>"
        f"<li>Unreadable images: {summary['unreadable_images']}</li>"
        f"<li>Duplicate groups: {summary['duplicate_groups']}</li>"
        f"<li>Shape groups: {len(summary['shape_groups'])}</li>"
        f"<li>Issues: {len(report['issues'])}</li>"
        "</ul>"
        f"{contact_html}"
        "<h2>Issues</h2><pre>"
        f"{report['issues']}"
        "</pre></body></html>",
        encoding="utf-8",
    )


def run_dataset_qa(config: DatasetQAConfig, *, repo_root: Path) -> dict[str, Any]:
    """Validate dataset roots and write QA artifacts."""

    output_dir = _repo_path(config.output_dir, repo_root=repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    root_summaries: list[dict[str, Any]] = []
    roots = [_repo_path(root, repo_root=repo_root) for root in config.source_roots]

    for root in roots:
        files = _image_files(root)
        if not files:
            issues.append({"severity": "error", "code": "empty_root", "path": str(root)})
        samples = [_inspect_image(path, root, config.leakage_group_policy) for path in files]
        all_samples.extend(samples)
        root_summaries.append({"root": str(root), "image_count": len(files)})
        for sample in samples:
            for error in sample["errors"]:
                issues.append({"severity": "error", "code": error, "path": sample["path"]})

    if config.task_type == "unpaired_translation" and len(roots) < 2:
        issues.append({"severity": "error", "code": "unpaired_requires_two_roots", "roots": [str(root) for root in roots]})

    duplicate_groups = [
        [row["path"] for row in rows]
        for _, rows in _group_by_hash(all_samples).items()
        if len(rows) > 1
    ]
    for group in duplicate_groups:
        issues.append({"severity": "warning", "code": "duplicate_files", "paths": group})

    shapes = Counter(
        f"{row['width']}x{row['height']}x{row['channels']}"
        for row in all_samples
        if row["readable"]
    )
    if len(shapes) > 1:
        issues.append({"severity": "warning", "code": "shape_mismatch", "shape_groups": dict(shapes)})

    groups_by_root: dict[str, set[str]] = defaultdict(set)
    for row in all_samples:
        root_name = _owning_root(row["path"], roots)
        if row["leakage_group"]:
            groups_by_root[root_name].add(str(row["leakage_group"]))
    leakage_overlap: list[str] = []
    if len(groups_by_root) > 1:
        sets = list(groups_by_root.values())
        leakage_overlap = sorted(set.intersection(*sets)) if all(sets) else []
        if leakage_overlap:
            issues.append({"severity": "warning", "code": "cross_domain_leakage_group_overlap", "groups": leakage_overlap})

    contact_cfg = config.contact_sheet
    contact = _contact_sheet(
        all_samples,
        output_dir / "contact_sheet.jpg",
        max_images=int(contact_cfg.get("max_images", 25)),
        thumb_size=int(contact_cfg.get("thumb_size", 128)),
    )

    report = {
        "schema_version": "microi2i.dataset_qa_report.v1",
        "dataset_id": config.dataset_id,
        "task_type": config.task_type,
        "status": "failed" if any(issue["severity"] == "error" for issue in issues) else "passed",
        "metadata": config.metadata,
        "source_roots": [str(root) for root in roots],
        "root_summaries": root_summaries,
        "summary": {
            "total_images": len(all_samples),
            "readable_images": sum(1 for row in all_samples if row["readable"]),
            "unreadable_images": sum(1 for row in all_samples if not row["readable"]),
            "duplicate_groups": len(duplicate_groups),
            "shape_groups": dict(shapes),
            "leakage_group_overlap": leakage_overlap,
        },
        "leakage_group_policy": asdict(config.leakage_group_policy),
        "samples": all_samples,
        "issues": issues,
        "contact_sheet": contact,
    }
    (output_dir / "dataset_qa_report.json").write_text(_json_dumps(report), encoding="utf-8")
    _write_html_report(report, output_dir / "dataset_qa_report.html")
    return report


def _group_by_hash(samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        digest = str(row.get("sha256", ""))
        if digest:
            grouped[digest].append(row)
    return grouped


def _owning_root(path: str, roots: list[Path]) -> str:
    p = Path(path)
    for root in roots:
        try:
            p.relative_to(root)
            return str(root)
        except ValueError:
            continue
    return ""


def _json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True)
