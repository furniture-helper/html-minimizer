import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Optional

import boto3
import psycopg
from boilerplate_remover.BoilerplateRemover import BoilerplateRemover
from botocore.exceptions import BotoCoreError, ClientError


def _env(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value is not None and value != "":
            return value
    return default


def _require_env(*names: str) -> str:
    value = _env(*names)
    if value is None:
        raise RuntimeError(f"Missing required environment variable. Expected one of: {', '.join(names)}")
    return value


def _safe_rel_key(s3_key: str) -> Path:
    pure = PurePosixPath(s3_key.lstrip("/"))
    parts = [p for p in pure.parts if p not in ("", ".", "..")]
    if not parts:
        raise RuntimeError(f"Invalid S3 key: {s3_key!r}")
    return Path(*parts)


def _join_s3_key(prefix: Optional[str], key: str) -> str:
    if not prefix:
        return key
    return f"{prefix.rstrip('/')}/{key.lstrip('/')}"


def get_db_connection():
    database_url = _env("DATABASE_URL")
    connect_timeout = int(_env("PGCONNECT_TIMEOUT", default="10"))

    if database_url:
        return psycopg.connect(database_url, connect_timeout=connect_timeout)

    required_settings = {
        "host": ("PGHOST", "PG_HOST"),
        "port": ("PGPORT", "PG_PORT"),
        "dbname": ("PGDATABASE", "PG_DATABASE"),
        "user": ("PGUSER", "PG_USER"),
        "password": ("PGPASSWORD", "PG_PASSWORD"),
    }

    connection_kwargs = {}
    missing = []
    for conn_key, env_names in required_settings.items():
        default = "5432" if conn_key == "port" else None
        value = _env(*env_names, default=default)
        if not value:
            missing.append("/".join(env_names))
        else:
            connection_kwargs[conn_key] = value

    if missing:
        raise RuntimeError(
            "Missing PostgreSQL settings. Set DATABASE_URL or provide: "
            "PGHOST/PG_HOST, PGPORT/PG_PORT, PGDATABASE/PG_DATABASE, "
            "PGUSER/PG_USER, PGPASSWORD/PG_PASSWORD. "
            f"Missing: {', '.join(missing)}"
        )

    return psycopg.connect(**connection_kwargs, connect_timeout=connect_timeout)

def _get_batch_limit() -> int:
    raw = _env("MINIMIZE_BATCH_LIMIT", default="10")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise RuntimeError(f"Invalid MINIMIZE_BATCH_LIMIT: {raw!r}. Expected a positive integer.")

    if value <= 0:
        raise RuntimeError(f"Invalid MINIMIZE_BATCH_LIMIT: {value}. Must be > 0.")
    return value

def get_s3_keys_to_minimize() -> list[tuple[str, str]]:
    limit = _get_batch_limit()
    query = """
        SELECT pages.s3_key, pages.url
        FROM pages
        LEFT JOIN minimized_pages ON pages.url = minimized_pages.url
        WHERE (
            pages.last_crawled_at > minimized_pages.last_minimized_at
            OR minimized_pages.last_minimized_at IS NULL
        )
          AND pages.is_active = true
          AND pages.s3_key != 'NOT_CRAWLED'
        ORDER BY pages.last_crawled_at ASC
        LIMIT %s
    """

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (limit,))
            return [(row[0], row[1]) for row in cur.fetchall()]

def download_anchor_tree_from_s3() -> Path:
    cache_path = Path(".cache") / "anchor_tree.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"Anchor tree cache exists at {cache_path}, skipping download")
        return cache_path

    s3_bucket = _require_env("ANCHOR_TREE_S3_BUCKET")
    s3_key = _require_env("ANCHOR_TREE_S3_KEY")
    s3_prefix = _env("ANCHOR_TREE_S3_PREFIX")
    full_key = _join_s3_key(s3_prefix, s3_key)

    try:
        s3 = boto3.client("s3")
        response = s3.get_object(Bucket=s3_bucket, Key=full_key)
        cache_path.write_bytes(response["Body"].read())
        size_bytes = response.get("ContentLength", 0)
        print(f"Anchor tree downloaded: {size_bytes / 1024 / 1024:.2f} MB -> {cache_path}")
        return cache_path
    except ClientError as error:
        error_code = error.response["Error"].get("Code", "Unknown")
        if error_code == "NoSuchKey":
            raise FileNotFoundError(f"S3 object not found: s3://{s3_bucket}/{full_key}") from error
        if error_code in ("NoSuchBucket", "AccessDenied"):
            raise PermissionError(f"S3 access error ({error_code}): s3://{s3_bucket}/{full_key}") from error
        raise
    except BotoCoreError as error:
        raise RuntimeError(f"S3 download failed: {error}") from error


def generate_minimized_html(s3_key: str, s3_client, boilerplate_remover: BoilerplateRemover) -> str:
    raw_bucket = _require_env("RAW_HTML_S3_BUCKET")
    rel_key = _safe_rel_key(s3_key)
    local_path = Path("tmp") / rel_key
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        s3_client.download_file(raw_bucket, s3_key, str(local_path))
        minimized_tree = boilerplate_remover.get_minimized_tree(str(local_path))
        return minimized_tree.to_html()
    finally:
        try:
            local_path.unlink(missing_ok=True)
        except TypeError:
            if local_path.exists():
                local_path.unlink()


def upload_minimized_html_to_s3(s3_client, html: str, source_s3_key: str) -> str:
    bucket = _require_env("MINIMIZED_HTML_S3_BUCKET")
    prefix = _env("MINIMIZED_HTML_S3_PREFIX")

    output_key = _join_s3_key(prefix, source_s3_key)

    s3_client.upload_fileobj(
        Fileobj=BytesIO(html.encode("utf-8")),
        Bucket=bucket,
        Key=output_key,
    )
    return output_key


def update_minimized_database(url: str, minimized_s3_key: str) -> None:
    query = """
        INSERT INTO minimized_pages (url, s3_key, last_minimized_at)
        VALUES (%s, %s, NOW())
        ON CONFLICT (url)
        DO UPDATE
        SET s3_key = EXCLUDED.s3_key,
            last_minimized_at = EXCLUDED.last_minimized_at
    """

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (url, minimized_s3_key))
        conn.commit()


_thread_local = threading.local()


def _get_thread_resources():
    if not hasattr(_thread_local, "boilerplate_remover"):
        _thread_local.boilerplate_remover = BoilerplateRemover()
    if not hasattr(_thread_local, "s3_client"):
        _thread_local.s3_client = boto3.client("s3")
    return _thread_local.s3_client, _thread_local.boilerplate_remover


def _minimize_worker(item: tuple[str, str]) -> tuple[str, str]:
    s3_key, url = item
    s3_client, boilerplate_remover = _get_thread_resources()
    minimized_html = generate_minimized_html(s3_key, s3_client, boilerplate_remover)
    minimized_s3_key = upload_minimized_html_to_s3(s3_client, minimized_html, s3_key)
    update_minimized_database(url, minimized_s3_key)
    return minimized_s3_key, url


if __name__ == "__main__":
    try:
        # Fail fast for required worker buckets before spawning threads.
        _require_env("RAW_HTML_S3_BUCKET")
        _require_env("MINIMIZED_HTML_S3_BUCKET")

        items = get_s3_keys_to_minimize()
        pending_count = len(items)
        print(f"{pending_count} pages to minimize")

        download_anchor_tree_from_s3()

        max_workers = int(_env("MAX_WORKERS", default="4"))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_minimize_worker, item): item for item in items}

            for future in as_completed(futures):
                source_s3_key, url = futures[future]
                pending_count -= 1
                try:
                    minimized_s3_key, _ = future.result()
                    print(f"Minimized: {url} ({source_s3_key} -> {minimized_s3_key}) ({pending_count} remaining)")
                except Exception as error:
                    print(f"Failed to minimize {url} ({source_s3_key}): {error}", file=sys.stderr)

    except (RuntimeError, psycopg.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
