from __future__ import annotations

import importlib
import json
import logging
import os
import sys
from pathlib import Path

from deepgit.github.schema import RepoRecord, TreeEntry

logger = logging.getLogger("deepgit.semantic")


def _ensure_real_numpy() -> None:
    """Materialize the real ``numpy`` before fastembed imports it.

    dspy installs a lazy ``_LazyModule`` proxy for ``numpy``. When fastembed later
    triggers it, dspy's loader re-enters ``exec_module`` mid-initialization and
    numpy blows up with ``data type 'bool' not understood``. Replacing the proxy in
    ``sys.modules`` with a freshly imported real module sidesteps the re-entrancy;
    dspy's dangling proxy self-heals via its ``loaded is not self`` early-return.
    """
    mod = sys.modules.get("numpy")
    if mod is not None and mod.__class__.__name__ == "_LazyModule":
        del sys.modules["numpy"]
        # Drop any partially-swapped submodules so the fresh import is clean.
        for name in [m for m in sys.modules if m == "numpy" or m.startswith("numpy.")]:
            sys.modules.pop(name, None)
        importlib.import_module("numpy")


_TRUST_INJECTED = False


def _use_os_trust_store() -> None:
    """Route TLS verification through the OS cert store (once).

    The model is downloaded from HuggingFace on first use, and the corporate
    network intercepts TLS with a Zscaler root CA that certifi doesn't ship.
    ``truststore`` makes Python's ``ssl`` module trust the Windows certificate
    store, which already contains that root, so the download verifies cleanly
    without disabling verification or hand-managing CA bundles.
    """
    global _TRUST_INJECTED
    if _TRUST_INJECTED:
        return
    try:
        import truststore

        truststore.inject_into_ssl()
        _TRUST_INJECTED = True
        logger.debug("injected OS trust store for HTTPS")
    except Exception as exc:  # pragma: no cover - truststore optional
        logger.debug("truststore unavailable (%s); relying on certifi", exc)

# CPU/ONNX embedding model — small, fast, strong for retrieval, no GPU needed.
_MODEL_NAME = os.getenv("DEEPGIT_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
_TABLE = "repos"
_MAX_README_STORE = 3000  # chars of README to persist per repo
_MAX_EMBED_CHARS = 1400   # keep embedding input within the model's token budget


def _index_dir() -> Path:
    override = os.getenv("DEEPGIT_INDEX_DIR")
    base = Path(override) if override else Path.home() / ".deepgit" / "lancedb"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _embed_text(rec: RepoRecord) -> str:
    """The document text we embed for a repo — its searchable semantic surface."""
    topics = " ".join(rec.topics[:12])
    readme = " ".join((rec.readme or "").split())[:_MAX_EMBED_CHARS]
    parts = [rec.name_with_owner, rec.description or "", topics, readme]
    return "\n".join(p for p in parts if p).strip()[: _MAX_EMBED_CHARS + 400]


class SemanticIndex:
    """A persistent LanceDB vector index over repositories.

    Safe to construct even when the optional dependencies are missing: it simply
    reports ``available == False`` and turns every method into a no-op.
    """

    def __init__(self) -> None:
        self.available = False
        self._model = None
        self._db = None
        self._table = None
        try:
            _ensure_real_numpy()
            import lancedb  # noqa: F401
            from fastembed import TextEmbedding
        except Exception as exc:  # pragma: no cover - optional deps
            logger.info("semantic index disabled (deps missing): %s", exc)
            return
        try:
            self._TextEmbedding = TextEmbedding
            self._lancedb = lancedb
            self._db = lancedb.connect(str(_index_dir()))
            if _TABLE in self._db.table_names():
                self._table = self._db.open_table(_TABLE)
            self.available = True
        except Exception as exc:  # pragma: no cover - env issues
            logger.warning("semantic index init failed: %s", exc)
            self.available = False

    # -- embedding -----------------------------------------------------------
    def _ensure_model(self):
        if self._model is None:
            _use_os_trust_store()
            cache_dir = _index_dir().parent / "models"
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("loading embedding model %s (first use downloads it)", _MODEL_NAME)
            self._model = self._TextEmbedding(
                model_name=_MODEL_NAME, cache_dir=str(cache_dir)
            )
        return self._model

    def _embed_docs(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        return [v.tolist() for v in model.embed(texts)]

    def _embed_query(self, text: str) -> list[float]:
        model = self._ensure_model()
        # bge models retrieve better with the query instruction that
        # ``query_embed`` adds; fall back to plain embed if unavailable.
        try:
            return list(model.query_embed(text))[0].tolist()
        except Exception:
            return list(model.embed([text]))[0].tolist()

    # -- persistence ---------------------------------------------------------
    def _row(self, rec: RepoRecord, vector: list[float]) -> dict:
        tree = [
            {"name": e.name, "type": e.type, "size": e.size} for e in rec.root_tree
        ]
        return {
            "vector": vector,
            "name_with_owner": rec.name_with_owner,
            "name": rec.name,
            "owner": rec.owner,
            "url": rec.url,
            "description": rec.description or "",
            "homepage": rec.homepage or "",
            "stars": int(rec.stars),
            "forks": int(rec.forks),
            "watchers": int(rec.watchers),
            "open_issues": int(rec.open_issues),
            "disk_usage_kb": int(rec.disk_usage_kb),
            "primary_language": rec.primary_language or "",
            "license_spdx": rec.license_spdx or "",
            "created_at": rec.created_at or "",
            "updated_at": rec.updated_at or "",
            "pushed_at": rec.pushed_at or "",
            "default_branch": rec.default_branch or "main",
            "is_archived": bool(rec.is_archived),
            "is_fork": bool(rec.is_fork),
            "languages_json": json.dumps(rec.languages or []),
            "topics_json": json.dumps(rec.topics or []),
            "root_tree_json": json.dumps(tree),
            "readme": (rec.readme or "")[:_MAX_README_STORE],
        }

    def _record(self, row: dict) -> RepoRecord:
        tree = [
            TreeEntry(name=e.get("name", ""), type=e.get("type", ""),
                      size=int(e.get("size", 0) or 0), path=e.get("name", ""))
            for e in json.loads(row.get("root_tree_json") or "[]")
        ]
        return RepoRecord(
            name_with_owner=row.get("name_with_owner", ""),
            name=row.get("name", ""),
            owner=row.get("owner", ""),
            url=row.get("url", ""),
            description=row.get("description", ""),
            homepage=row.get("homepage", ""),
            stars=int(row.get("stars", 0) or 0),
            forks=int(row.get("forks", 0) or 0),
            watchers=int(row.get("watchers", 0) or 0),
            open_issues=int(row.get("open_issues", 0) or 0),
            disk_usage_kb=int(row.get("disk_usage_kb", 0) or 0),
            primary_language=row.get("primary_language", ""),
            languages=json.loads(row.get("languages_json") or "[]"),
            topics=json.loads(row.get("topics_json") or "[]"),
            license_spdx=row.get("license_spdx", ""),
            created_at=row.get("created_at", ""),
            updated_at=row.get("updated_at", ""),
            pushed_at=row.get("pushed_at", ""),
            is_archived=bool(row.get("is_archived", False)),
            is_fork=bool(row.get("is_fork", False)),
            default_branch=row.get("default_branch", "main") or "main",
            readme=row.get("readme", ""),
            root_tree=tree,
        )

    def upsert(self, records: list[RepoRecord]) -> int:
        """Embed and persist repos, updating any that were seen before."""
        if not self.available or not records:
            return 0
        # De-dup within the batch, keeping the richer record per repo.
        by_name: dict[str, RepoRecord] = {}
        for r in records:
            if r.name_with_owner:
                by_name[r.name_with_owner] = r
        recs = list(by_name.values())
        try:
            vectors = self._embed_docs([_embed_text(r) for r in recs])
        except Exception as exc:
            logger.warning("embedding failed, skipping upsert: %s", exc)
            return 0
        rows = [self._row(r, v) for r, v in zip(recs, vectors)]
        try:
            if self._table is None:
                self._table = self._db.create_table(_TABLE, data=rows)
            else:
                (
                    self._table.merge_insert("name_with_owner")
                    .when_matched_update_all()
                    .when_not_matched_insert_all()
                    .execute(rows)
                )
            return len(rows)
        except Exception as exc:
            logger.warning("lancedb upsert failed: %s", exc)
            return 0

    def search(
        self, query_text: str, *, k: int = 10, exclude: set[str] | None = None
    ) -> list[RepoRecord]:
        """Return the ``k`` semantically nearest repos not already in ``exclude``."""
        if not self.available or self._table is None:
            return []
        exclude = {e.lower() for e in (exclude or set())}
        try:
            vec = self._embed_query(query_text)
            rows = self._table.search(vec).limit(k + len(exclude) + 5).to_list()
        except Exception as exc:
            logger.warning("semantic search failed: %s", exc)
            return []
        out: list[RepoRecord] = []
        for row in rows:
            name = (row.get("name_with_owner") or "").lower()
            if not name or name in exclude:
                continue
            out.append(self._record(row))
            if len(out) >= k:
                break
        return out

    def count(self) -> int:
        if not self.available or self._table is None:
            return 0
        try:
            return self._table.count_rows()
        except Exception:
            return 0


_INDEX: SemanticIndex | None = None


def get_index() -> SemanticIndex:
    """Process-wide lazy singleton for the semantic index."""
    global _INDEX
    if _INDEX is None:
        _INDEX = SemanticIndex()
    return _INDEX
