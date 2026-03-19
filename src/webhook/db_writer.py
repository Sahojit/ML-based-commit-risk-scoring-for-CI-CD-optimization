import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_engine = None


def _get_engine():
    global _engine
    if _engine is not None:
        return _engine

    try:
        from sqlalchemy import create_engine
        db_url = _build_db_url()
        _engine = create_engine(db_url, pool_pre_ping=True, pool_size=5)
        logger.info("Database engine created.")
        return _engine
    except Exception as e:
        logger.error(f"Could not create DB engine: {e}")
        return None


def _build_db_url() -> str:
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        return database_url

    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    name = os.environ.get("DB_NAME", "commit_risk_db")
    user = os.environ.get("DB_USERNAME", "postgres")
    password = os.environ.get("DB_PASSWORD", "")
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


def ensure_predictions_table():
    engine = _get_engine()
    if engine is None:
        return

    ddl = """
    CREATE TABLE IF NOT EXISTS webhook_predictions (
        id              SERIAL PRIMARY KEY,
        commit_hash     VARCHAR(40)  NOT NULL,
        repo_name       VARCHAR(255),
        branch          VARCHAR(255),
        author          VARCHAR(255),
        author_email    VARCHAR(255),
        commit_message  TEXT,
        risk_score      FLOAT        NOT NULL,
        risk_level      VARCHAR(10)  NOT NULL,
        decision        TEXT         NOT NULL,
        files_changed   INTEGER,
        lines_added     INTEGER,
        lines_deleted   INTEGER,
        touches_core    INTEGER,
        touches_tests   INTEGER,
        source          VARCHAR(20)  DEFAULT 'webhook',
        created_at      TIMESTAMP    DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_wp_commit_hash ON webhook_predictions(commit_hash);
    CREATE INDEX IF NOT EXISTS idx_wp_created_at  ON webhook_predictions(created_at);
    """

    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text(ddl))
            conn.commit()
        logger.info("webhook_predictions table ensured.")
    except Exception as e:
        logger.error(f"Failed to create table: {e}")


def write_prediction(
    commit_data: Dict[str, Any],
    prediction: Dict[str, Any],
) -> Optional[int]:
    engine = _get_engine()
    if engine is None:
        logger.warning("No DB engine — skipping write.")
        return None

    sql = """
    INSERT INTO webhook_predictions
        (commit_hash, repo_name, branch, author, author_email, commit_message,
         risk_score, risk_level, decision,
         files_changed, lines_added, lines_deleted, touches_core, touches_tests)
    VALUES
        (:commit_hash, :repo_name, :branch, :author, :author_email, :commit_message,
         :risk_score, :risk_level, :decision,
         :files_changed, :lines_added, :lines_deleted, :touches_core, :touches_tests)
    ON CONFLICT DO NOTHING
    RETURNING id;
    """

    row = {
        "commit_hash":    commit_data.get("commit_hash", ""),
        "repo_name":      commit_data.get("repo_name", ""),
        "branch":         commit_data.get("branch", ""),
        "author":         commit_data.get("author", ""),
        "author_email":   commit_data.get("author_email", ""),
        "commit_message": commit_data.get("message", "")[:500],
        "risk_score":     prediction.get("risk_score", 0.0),
        "risk_level":     prediction.get("risk_level", "UNKNOWN"),
        "decision":       prediction.get("recommendation", ""),
        "files_changed":  commit_data.get("files_changed", 0),
        "lines_added":    commit_data.get("lines_added", 0),
        "lines_deleted":  commit_data.get("lines_deleted", 0),
        "touches_core":   commit_data.get("touches_core", 0),
        "touches_tests":  commit_data.get("touches_tests", 0),
    }

    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text(sql), row)
            conn.commit()
            inserted_id = result.scalar()
            logger.info(
                f"Stored prediction for {row['commit_hash'][:8]}: "
                f"{row['risk_level']} ({row['risk_score']:.3f}) → id={inserted_id}"
            )
            return inserted_id
    except Exception as e:
        logger.error(f"Failed to write prediction to DB: {e}")
        return None
