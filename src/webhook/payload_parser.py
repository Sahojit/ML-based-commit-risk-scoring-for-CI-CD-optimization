import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ParsedCommit:
    def __init__(
        self,
        commit_hash: str,
        author: str,
        author_email: str,
        timestamp: datetime,
        message: str,
        files_added: List[str],
        files_modified: List[str],
        files_removed: List[str],
        lines_added: int,
        lines_deleted: int,
        repo_name: str,
        branch: str,
    ):
        self.commit_hash = commit_hash
        self.author = author
        self.author_email = author_email
        self.timestamp = timestamp
        self.message = message
        self.files_added = files_added
        self.files_modified = files_modified
        self.files_removed = files_removed
        self.lines_added = lines_added
        self.lines_deleted = lines_deleted
        self.repo_name = repo_name
        self.branch = branch

    @property
    def files_changed(self) -> int:
        return len(self.files_added) + len(self.files_modified) + len(self.files_removed)

    @property
    def all_files(self) -> List[str]:
        return self.files_added + self.files_modified + self.files_removed

    @property
    def touches_tests(self) -> int:
        test_patterns = ("test", "spec", "_test.", ".test.")
        return int(any(p in f.lower() for f in self.all_files for p in test_patterns))

    @property
    def touches_core(self) -> int:
        core_patterns = ("src/core", "src/api", "src/models", "src/inference")
        return int(any(p in f.lower() for f in self.all_files for p in core_patterns))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "commit_hash": self.commit_hash,
            "author": self.author,
            "author_email": self.author_email,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "files_added": self.files_added,
            "files_modified": self.files_modified,
            "files_removed": self.files_removed,
            "files_changed": self.files_changed,
            "lines_added": self.lines_added,
            "lines_deleted": self.lines_deleted,
            "touches_tests": self.touches_tests,
            "touches_core": self.touches_core,
            "repo_name": self.repo_name,
            "branch": self.branch,
        }


def parse_github_payload(payload: Dict[str, Any]) -> Optional[ParsedCommit]:
    head = payload.get("head_commit")
    if not head:
        logger.warning("Payload has no head_commit — not a push event or empty push.")
        return None

    try:
        commit_hash = head["id"]
        message = head.get("message", "")
        timestamp = datetime.fromisoformat(
            head["timestamp"].replace("Z", "+00:00")
        )

        author_info = head.get("author", {})
        author = author_info.get("name", "unknown")
        author_email = author_info.get("email", "unknown")

        files_added = head.get("added", [])
        files_modified = head.get("modified", [])
        files_removed = head.get("removed", [])

        stats = _extract_line_stats(payload, head)
        lines_added = stats["lines_added"]
        lines_deleted = stats["lines_deleted"]

        repo_name = payload.get("repository", {}).get("full_name", "unknown")
        branch = payload.get("ref", "").replace("refs/heads/", "")

        commit = ParsedCommit(
            commit_hash=commit_hash,
            author=author,
            author_email=author_email,
            timestamp=timestamp,
            message=message,
            files_added=files_added,
            files_modified=files_modified,
            files_removed=files_removed,
            lines_added=lines_added,
            lines_deleted=lines_deleted,
            repo_name=repo_name,
            branch=branch,
        )

        logger.info(
            f"Parsed commit {commit_hash[:8]} by {author} "
            f"(+{lines_added}/-{lines_deleted}, {commit.files_changed} files)"
        )
        return commit

    except KeyError as e:
        logger.error(f"Missing required field in payload: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to parse GitHub payload: {e}")
        return None


def _extract_line_stats(
    payload: Dict[str, Any], head_commit: Dict[str, Any]
) -> Dict[str, int]:
    lines_added = head_commit.get("added_lines", 0)
    lines_deleted = head_commit.get("removed_lines", 0)

    if lines_added == 0 and lines_deleted == 0:
        all_files = (
            head_commit.get("added", [])
            + head_commit.get("modified", [])
            + head_commit.get("removed", [])
        )
        lines_added = len(all_files) * 10
        lines_deleted = max(len(head_commit.get("removed", [])) * 10, 0)

    return {"lines_added": lines_added, "lines_deleted": lines_deleted}
