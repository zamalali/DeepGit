from __future__ import annotations

from dataclasses import dataclass, field

from deepgit.state import Repository


@dataclass(kw_only=True)
class TreeEntry:
    """A single entry in a repository's root tree."""

    name: str
    type: str  # "blob" | "tree"
    size: int = 0
    path: str = ""


@dataclass(kw_only=True)
class RepoRecord:
    """Rich, single-round-trip view of a repository.

    This is the ingestion-layer model. It carries everything a downstream
    agent needs to *decide what to read next* without any further API calls:
    metadata, README text, and the root file tree.
    """

    name_with_owner: str = ""
    name: str = ""
    owner: str = ""
    url: str = ""
    description: str = ""
    homepage: str = ""

    stars: int = 0
    forks: int = 0
    watchers: int = 0
    open_issues: int = 0

    primary_language: str = ""
    languages: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    license_spdx: str = ""

    created_at: str = ""
    updated_at: str = ""
    pushed_at: str = ""
    is_archived: bool = False
    is_fork: bool = False
    default_branch: str = "main"

    disk_usage_kb: int = 0

    readme: str = ""
    root_tree: list[TreeEntry] = field(default_factory=list)

    def has_tests(self) -> bool:
        names = {e.name.lower() for e in self.root_tree}
        return bool(names & {"tests", "test", "spec", "__tests__"})

    def has_ci(self) -> bool:
        return any(e.name == ".github" for e in self.root_tree)

    def manifest_files(self) -> list[str]:
        """Dependency/build manifests present at the repo root."""
        manifests = {
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "requirements.txt",
            "package.json",
            "cargo.toml",
            "go.mod",
            "pom.xml",
            "build.gradle",
            "gemfile",
            "composer.json",
        }
        return [e.name for e in self.root_tree if e.name.lower() in manifests]

    def to_repository(self) -> Repository:
        """Bridge to the legacy ``Repository`` dataclass for compatibility."""
        combined = self.description
        if self.readme:
            combined = f"{self.description}\n\n{self.readme}"
        return Repository(
            title=self.name,
            full_name=self.name_with_owner,
            link=self.url,
            clone_url=f"{self.url}.git" if self.url else "",
            description=self.description,
            combined_doc=combined,
            stars=self.stars,
            forks=self.forks,
            open_issues_count=self.open_issues,
            language=self.primary_language,
            topics=list(self.topics),
            license=self.license_spdx,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )
