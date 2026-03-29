import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Optional

import boto3
import psycopg
from boilerplate_remover.BoilerplateRemover import BoilerplateRemover
from botocore.exceptions import BotoCoreError, ClientError


VALID_LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


def _get_log_level() -> int:
    raw_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    resolved_level = VALID_LOG_LEVELS.get(raw_level)

    if resolved_level is None:
        sys.stderr.write(
            f"Invalid LOG_LEVEL={raw_level!r}. Falling back to INFO. "
            f"Valid values: {', '.join(sorted(VALID_LOG_LEVELS))}\n"
        )
        return logging.INFO

    return resolved_level


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s [%(processName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(_get_log_level())


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


def download_anchor_trees_from_s3() -> list[Path]:
    """
    Download all anchor tree files from the S3 bucket and store them in .cache/anchor_trees/.
    """
    cache_dir = Path(".cache") / "anchor_trees"
    cache_dir.mkdir(parents=True, exist_ok=True)

    s3_bucket = _require_env("ANCHOR_TREE_S3_BUCKET")
    s3_prefix = _env("ANCHOR_TREE_S3_PREFIX")
    # Download all objects under the prefix (or all if prefix is None)
    try:
        s3 = boto3.client("s3", region_name="eu-west-1")
        paginator = s3.get_paginator("list_objects_v2")
        list_kwargs = {"Bucket": s3_bucket}
        if s3_prefix:
            list_kwargs["Prefix"] = s3_prefix
        downloaded_paths = []
        for page in paginator.paginate(**list_kwargs):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Use relative path under prefix for local filename
                rel_key = key[len(s3_prefix):].lstrip("/") if s3_prefix and key.startswith(s3_prefix) else key
                local_path = cache_dir / rel_key.replace("/", "_")
                if local_path.exists():
                    logger.info("Anchor tree cache exists at %s, skipping download", local_path)
                else:
                    try:
                        response = s3.get_object(Bucket=s3_bucket, Key=key)
                        local_path.write_bytes(response["Body"].read())
                        size_bytes = response.get("ContentLength", 0)
                        logger.info(
                            "Anchor tree downloaded: %.2f MB -> %s",
                            size_bytes / 1024 / 1024,
                            local_path,
                        )
                    except ClientError as error:
                        error_code = error.response["Error"].get("Code", "Unknown")
                        if error_code == "NoSuchKey":
                            logger.error(f"S3 object not found: s3://{s3_bucket}/{key}")
                            continue
                        if error_code in ("NoSuchBucket", "AccessDenied"):
                            logger.error(f"S3 access error ({error_code}): s3://{s3_bucket}/{key}")
                            continue
                        logger.error(f"S3 error for key {key}: {error}")
                        continue
                    except BotoCoreError as error:
                        logger.error(f"S3 download failed for key {key}: {error}")
                        continue
                downloaded_paths.append(local_path)
        if not downloaded_paths:
            logger.warning("No anchor tree files found in s3://%s/%s", s3_bucket, s3_prefix or "")
        return downloaded_paths
    except Exception as error:
        raise RuntimeError(f"S3 anchor tree download failed: {error}") from error


def generate_minimized_html(
    s3_key: str,
    url: str,
    s3_client,
    boilerplate_remover: BoilerplateRemover,
) -> str:
    raw_bucket = _require_env("RAW_HTML_S3_BUCKET")
    rel_key = _safe_rel_key(s3_key)

    logger.debug("Downloading raw HTML for %s", url)
    html_content = s3_client.get_object(Bucket=raw_bucket, Key=s3_key)["Body"].read().decode("utf-8")
    logger.debug("Downloaded raw HTML for %s", url)

    minimized_tree = boilerplate_remover.get_minimized_tree_from_string(html_content)
    logger.debug("Generated minimized tree for %s", url)

    return minimized_tree.to_html()


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
    bulk_update_minimized_database([(url, minimized_s3_key)])


def bulk_update_minimized_database(results: list[tuple[str, str]]) -> None:
    """Upsert multiple pages in a single DB round-trip."""
    if not results:
        return
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
            cur.executemany(query, results)
        conn.commit()



# Module-level globals — one instance per worker process
_process_s3_client = None
_process_boilerplate_removers: dict[str, BoilerplateRemover] = {}


def _init_process_worker():
    """Initializer called once per worker process before any tasks run."""
    global _process_s3_client, _process_boilerplate_removers
    logger.debug("Initializing process worker globals")
    _process_boilerplate_removers = {}
    logger.debug("BoilerplateRemover dict initialized for process")
    logger.debug("Creating S3 client for process")
    _process_s3_client = boto3.client("s3", region_name="eu-west-1")
    logger.debug("S3 client created for process")


def _get_worker_resources(domain: str):
    """Return process-level resources, initializing lazily if needed."""
    global _process_s3_client, _process_boilerplate_removers
    if domain not in _process_boilerplate_removers:
        logger.debug(f"Creating BoilerplateRemover instance for domain '{domain}' in process")
        anchor_tree_cache_path = f".cache/anchor_trees/anchor_tree_{domain.replace('/', '_')}.pkl"
        logger.debug(f"Reading cached anchor_tree from %s", anchor_tree_cache_path)
        _process_boilerplate_removers[domain] = BoilerplateRemover(cache_path=anchor_tree_cache_path)
        logger.debug(f"BoilerplateRemover instance created for domain '{domain}' in process")
    if _process_s3_client is None:
        logger.debug("Creating S3 client for process")
        _process_s3_client = boto3.client("s3", region_name="eu-west-1")
        logger.debug("S3 client created for process")
    return _process_s3_client, _process_boilerplate_removers[domain]


def _minimize_worker(item: tuple[str, str]) -> tuple[str, str]:
    s3_key, url = item
    domain = get_domain_from_url(url)
    s3_client, boilerplate_remover = _get_worker_resources(domain)

    logger.debug("Minimizing %s", url)
    minimized_html = generate_minimized_html(s3_key, url, s3_client, boilerplate_remover)
    logger.debug("Minimized %s", url)

    minimized_s3_key = upload_minimized_html_to_s3(s3_client, minimized_html, s3_key)
    logger.debug("Uploaded %s", url)

    return minimized_s3_key, url

def get_domain_from_url(url: str) -> str:
    # Simple extraction of domain from URL for logging purposes
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return "unknown_domain"

if __name__ == "__main__":
    try:
        _require_env("RAW_HTML_S3_BUCKET")
        _require_env("MINIMIZED_HTML_S3_BUCKET")

        items = get_s3_keys_to_minimize()
        pending_count = len(items)
        logger.debug("%s pages to minimize", pending_count)

        download_anchor_trees_from_s3()

        completed: list[tuple[str, str]] = []  # (url, minimized_s3_key)

        max_workers = int(_env("MAX_WORKERS", default="4"))
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_process_worker) as executor:
            futures = {executor.submit(_minimize_worker, item): item for item in items}

            for future in as_completed(futures):
                source_s3_key, url = futures[future]
                pending_count -= 1
                try:
                    minimized_s3_key, _ = future.result()
                    completed.append((url, minimized_s3_key))
                    logger.info(
                        "Minimized: %s (%s -> %s) (%s remaining)",
                        url,
                        source_s3_key,
                        minimized_s3_key,
                        pending_count,
                    )
                except Exception:
                    logger.exception(
                        "Failed to minimize %s (%s)",
                        url,
                        source_s3_key,
                    )

        if completed:
            bulk_update_minimized_database(completed)
            logger.debug("Bulk updated minimized database for %d pages", len(completed))

    except (RuntimeError, psycopg.Error):
        logger.exception("Error while running minimizer")
        raise SystemExit(1)
