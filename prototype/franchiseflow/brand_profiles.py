"""Small local brand asset catalog used by the FranchiseFlow prototype."""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROFILE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _profile_dir(root: str | Path, profile_id: str) -> Path:
    profile_id = str(profile_id or "")
    if not PROFILE_ID_RE.fullmatch(profile_id):
        raise ValueError("invalid brand profile id")
    return Path(root) / profile_id


def _profile_path(root: str | Path, profile_id: str) -> Path:
    return _profile_dir(root, profile_id) / "profile.json"


def load_profile(root: str | Path, profile_id: str) -> dict[str, Any] | None:
    path = _profile_path(root, profile_id)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"brand profile is unreadable: {profile_id}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"brand profile is invalid: {profile_id}")
    return value


def list_profiles(root: str | Path) -> list[dict[str, Any]]:
    base = Path(root)
    if not base.exists():
        return []
    profiles: list[dict[str, Any]] = []
    for path in sorted(base.glob("*/profile.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            profiles.append(value)
    return profiles


def create_profile(
    root: str | Path,
    profile_id: str,
    *,
    brand_name: str,
    target_hint: str | None,
    branch_labels: list[str],
    branch_sources: Iterable[Path],
    default_fit: str = "cover",
    surface_policy: str = "review",
) -> dict[str, Any]:
    """Copy validated branch assets and write one durable profile manifest."""
    if not brand_name or len(brand_name) > 120:
        raise ValueError("brand_name must contain 1 to 120 characters")
    if target_hint is not None and len(target_hint) > 500:
        raise ValueError("target_hint must be at most 500 characters")
    if default_fit not in {"cover", "contain"}:
        raise ValueError("default_fit must be cover or contain")
    if surface_policy not in {"review", "all_surfaces"}:
        raise ValueError("surface_policy must be review or all_surfaces")
    sources = [Path(item) for item in branch_sources]
    if not sources or len(sources) != len(branch_labels):
        raise ValueError("brand profile requires one label per branch asset")
    directory = _profile_dir(root, profile_id)
    if directory.exists():
        raise ValueError(f"brand profile already exists: {profile_id}")
    directory.mkdir(parents=True, exist_ok=False)
    branch_files: list[str] = []
    try:
        for index, source in enumerate(sources, start=1):
            if not source.is_file():
                raise ValueError(f"brand profile asset is missing: {source.name}")
            suffix = source.suffix.lower()
            if suffix not in {".png", ".jpg", ".jpeg"}:
                raise ValueError(f"brand profile asset has unsupported extension: {suffix}")
            filename = f"branch_{index}{suffix}"
            shutil.copyfile(source, directory / filename)
            branch_files.append(filename)
        profile = {
            "profile_id": profile_id,
            "brand_name": brand_name,
            "target_hint": target_hint,
            "branch_labels": list(branch_labels),
            "branch_files": branch_files,
            "default_fit": default_fit,
            "surface_policy": surface_policy,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        path = directory / "profile.json"
        fd, temporary = tempfile.mkstemp(prefix=".profile-", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(profile, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        return profile
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise


def profile_asset_paths(root: str | Path, profile: dict[str, Any]) -> list[Path]:
    profile_id = str(profile.get("profile_id") or "")
    directory = _profile_dir(root, profile_id)
    filenames = profile.get("branch_files")
    if not isinstance(filenames, list) or not filenames:
        raise ValueError("brand profile has no branch assets")
    result: list[Path] = []
    for filename in filenames:
        name = Path(str(filename)).name
        path = directory / name
        if path.parent != directory or not path.is_file():
            raise ValueError(f"brand profile asset is missing: {name}")
        result.append(path)
    return result
