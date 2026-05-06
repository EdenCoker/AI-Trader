from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path


class ModelPromoter:
    """Versioned, atomic promotion and rollback for local calibrator models."""

    def __init__(
        self,
        *,
        models_dir: Path = Path("data/models"),
        production_path: Path | None = None,
        registry_path: Path | None = None,
        archive_dir: Path | None = None,
    ) -> None:
        self.models_dir = models_dir
        self.production_path = production_path or models_dir / "production.json"
        self.registry_path = registry_path or models_dir / "version_registry.json"
        self.archive_dir = archive_dir or models_dir / "archive"

    def promote(
        self,
        *,
        candidate: Path,
        reason: str,
        gate_result: dict | None = None,
    ) -> dict:
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        registry = self._load_registry()
        version = self._next_version(registry)
        stamp = datetime.now(UTC).strftime("%Y-%m-%d")
        archive_path = self.archive_dir / f"{version}_{stamp}.json"

        self._atomic_copy(candidate, archive_path)
        self._archive_unregistered_previous(registry)
        self._atomic_copy(archive_path, self.production_path)

        model_payload = _read_json(candidate)
        entry = {
            "version": version,
            "path": str(archive_path),
            "promoted_at": datetime.now(UTC).isoformat(),
            "reason": reason,
            "model_sha256": _sha256(archive_path),
            "training_count": int(model_payload.get("training_count", 0)),
            "metrics": model_payload.get("metrics", {}),
            "gate_result": gate_result or {},
        }
        registry["current"] = version
        registry.setdefault("history", []).append(entry)
        self._save_registry(registry)
        return entry

    def reject_candidate(
        self,
        *,
        candidate: Path,
        reason: str,
        gate_result: dict | None = None,
    ) -> dict:
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        registry = self._load_registry()
        version = self._next_version(registry)
        stamp = datetime.now(UTC).strftime("%Y-%m-%d")
        archive_path = self.archive_dir / f"{version}_candidate_REJECTED_{stamp}.json"
        self._atomic_copy(candidate, archive_path)
        entry = {
            "version": version,
            "path": str(archive_path),
            "rejected_at": datetime.now(UTC).isoformat(),
            "reason": reason,
            "model_sha256": _sha256(archive_path),
            "gate_result": gate_result or {},
        }
        registry.setdefault("rejections", []).append(entry)
        self._save_registry(registry)
        return entry

    def rollback(self, *, to_version: str) -> dict:
        registry = self._load_registry()
        entry = self._find_version(registry, to_version)
        if entry is None:
            raise ValueError(f"Unknown model version: {to_version}")
        source = Path(entry["path"])
        if not source.exists():
            raise FileNotFoundError(source)
        self._atomic_copy(source, self.production_path)
        rollback = {
            "version": to_version,
            "rolled_back_at": datetime.now(UTC).isoformat(),
            "source_path": str(source),
            "model_sha256": _sha256(source),
        }
        registry["current"] = to_version
        registry.setdefault("rollbacks", []).append(rollback)
        self._save_registry(registry)
        return rollback

    def _archive_unregistered_previous(self, registry: dict) -> None:
        if not self.production_path.exists() or registry.get("current"):
            return
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self._atomic_copy(
            self.production_path,
            self.archive_dir / f"v0000_previous_{stamp}.json",
        )

    def _load_registry(self) -> dict:
        if not self.registry_path.exists():
            return {"current": None, "history": [], "rejections": [], "rollbacks": []}
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _save_registry(self, registry: dict) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.registry_path.with_name(self.registry_path.name + ".tmp")
        tmp.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.registry_path)

    def _next_version(self, registry: dict) -> str:
        numbers = []
        for section in ("history", "rejections"):
            for item in registry.get(section, []):
                version = str(item.get("version", ""))
                if version.startswith("v") and version[1:5].isdigit():
                    numbers.append(int(version[1:5]))
        return f"v{(max(numbers, default=0) + 1):04d}"

    def _find_version(self, registry: dict, version: str) -> dict | None:
        for item in registry.get("history", []):
            if item.get("version") == version:
                return item
        return None

    @staticmethod
    def _atomic_copy(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_name(destination.name + ".tmp")
        shutil.copy2(source, tmp)
        tmp.replace(destination)


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
