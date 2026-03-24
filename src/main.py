import logging
import os
import sys
import time
import uuid
from collections import OrderedDict
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


def _normalize_domain(domain: Optional[str]) -> str:
    if not domain:
        return ""
    normalized = domain.strip().lower()
    if normalized.startswith("www."):
        normalized = normalized[4:]
    return normalized


def _get_anchor_tree_cache_dir() -> Path:
    cache_dir = Path(_env("ANCHOR_TREE_CACHE_DIR", default=".cache/anchor_tree"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_anchor_tree_candidates(domain: Optional[str]) -> list[tuple[str, str]]:
    """Return ordered candidate keys as (label, full_s3_key)."""
    s3_prefix = _env("ANCHOR_TREE_S3_PREFIX")
    key_template = _env("ANCHOR_TREE_S3_KEY_TEMPLATE", default="anchor_tree_{domain}.pkl")
    default_key = _env("ANCHOR_TREE_DEFAULT_S3_KEY", "ANCHOR_TREE_S3_KEY")

    candidates: list[tuple[str, str]] = []
    normalized_domain = _normalize_domain(domain)

    if normalized_domain:
        rendered = key_template.format(domain=normalized_domain)
        candidates.append((f"domain:{normalized_domain}", _join_s3_key(s3_prefix, rendered)))

    if default_key:
        default_full_key = _join_s3_key(s3_prefix, default_key)
        if all(default_full_key != full_key for _, full_key in candidates):
            candidates.append(("default", default_full_key))

    if not candidates:
        raise RuntimeError(
            "Anchor tree key configuration is missing. Set ANCHOR_TREE_S3_KEY_TEMPLATE or "
            "ANCHOR_TREE_DEFAULT_S3_KEY/ANCHOR_TREE_S3_KEY."
        )

    return candidates


def _anchor_cache_path_for_key(full_s3_key: str) -> Path:
    rel_key = _safe_rel_key(full_s3_key)
    return _get_anchor_tree_cache_dir() / rel_key


def _download_anchor_tree_for_domain(s3_client, domain: Optional[str]) -> Path:
    bucket = _require_env("ANCHOR_TREE_S3_BUCKET")
    candidates = _get_anchor_tree_candidates(domain)
    errors: list[str] = []

    for label, full_key in candidates:
        cache_path = _anchor_cache_path_for_key(full_key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            logger.debug("Anchor tree cache hit for %s at %s", label, cache_path)
            return cache_path

        try:
            logger.debug("Downloading anchor tree (%s) from s3://%s/%s", label, bucket, full_key)
            response = s3_client.get_object(Bucket=bucket, Key=full_key)
            tmp_path = cache_path.with_suffix(
                f"{cache_path.suffix}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            payload = response["Body"].read()
            tmp_path.write_bytes(payload)

            # On Windows, replacing a file can fail if another process has it open.
            # If another process already published the cache file, treat that as a cache hit.
            published = False
            for _ in range(5):
                if cache_path.exists():
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    return cache_path
                try:
                    os.replace(tmp_path, cache_path)
                    published = True
                    break
                except PermissionError:
                    if cache_path.exists():
                        try:
                            tmp_path.unlink(missing_ok=True)
                        except OSError:
                            pass
                        return cache_path
                    time.sleep(0.05)

            if not published:
                if cache_path.exists():
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    return cache_path
                raise PermissionError(f"Unable to publish anchor tree cache file: {cache_path}")

            logger.info("Anchor tree downloaded for %s -> %s", label, cache_path)
            return cache_path
        except ClientError as error:
            error_code = error.response["Error"].get("Code", "Unknown")
            if error_code == "NoSuchKey":
                errors.append(f"{label}=s3://{bucket}/{full_key} (not found)")
                continue
            if error_code in ("NoSuchBucket", "AccessDenied"):
                raise PermissionError(f"S3 access error ({error_code}): s3://{bucket}/{full_key}") from error
            raise
        except BotoCoreError as error:
            raise RuntimeError(f"S3 download failed for s3://{bucket}/{full_key}: {error}") from error

    raise FileNotFoundError(
        "No anchor tree found for domain candidates. Tried: " + "; ".join(errors)
    )


def _get_domain_remover_cache_size() -> int:
    raw = _env("ANCHOR_TREE_PROCESS_CACHE_SIZE", default="64")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise RuntimeError(f"Invalid ANCHOR_TREE_PROCESS_CACHE_SIZE: {raw!r}. Expected a positive integer.")
    if value <= 0:
        raise RuntimeError(f"Invalid ANCHOR_TREE_PROCESS_CACHE_SIZE: {value}. Must be > 0.")
    return value


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


def get_s3_keys_to_minimize() -> list[tuple[str, str, Optional[str]]]:
    limit = _get_batch_limit()
    query = """
        SELECT pages.s3_key, pages.url, pages.domain
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
            return [(row[0], row[1], row[2]) for row in cur.fetchall()]


def generate_minimized_html(
    s3_key: str,
    url: str,
    s3_client,
    boilerplate_remover: BoilerplateRemover,
) -> str:
    raw_bucket = _require_env("RAW_HTML_S3_BUCKET")
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


# Module-level globals — one set per worker process
_process_s3_client = None
_process_boilerplate_removers: OrderedDict[str, BoilerplateRemover] = OrderedDict()


def _init_process_worker():
    """Initializer called once per worker process before any tasks run."""
    global _process_s3_client
    logger.debug("Creating S3 client for process")
    _process_s3_client = boto3.client("s3", region_name="eu-west-1")
    logger.debug("S3 client created for process")


def _get_worker_resources():
    """Return process-level resources, initializing lazily if needed."""
    global _process_s3_client
    if _process_s3_client is None:
        logger.debug("Creating S3 client for process")
        _process_s3_client = boto3.client("s3", region_name="eu-west-1")
        logger.debug("S3 client created for process")
    return _process_s3_client


def _get_or_create_remover_for_domain(s3_client, domain: Optional[str]) -> BoilerplateRemover:
    normalized_domain = _normalize_domain(domain)
    cache_key = normalized_domain or "__default__"

    remover = _process_boilerplate_removers.get(cache_key)
    if remover is not None:
        _process_boilerplate_removers.move_to_end(cache_key)
        return remover

    anchor_tree_cache_path = _download_anchor_tree_for_domain(s3_client, normalized_domain)
    logger.debug("Creating BoilerplateRemover instance for domain=%s", normalized_domain or "default")
    remover = BoilerplateRemover(cache_path=str(anchor_tree_cache_path))
    _process_boilerplate_removers[cache_key] = remover
    _process_boilerplate_removers.move_to_end(cache_key)

    max_size = _get_domain_remover_cache_size()
    while len(_process_boilerplate_removers) > max_size:
        evicted_domain, _ = _process_boilerplate_removers.popitem(last=False)
        logger.debug("Evicted domain remover cache entry: %s", evicted_domain)

    return remover


def _minimize_worker(item: tuple[str, str, Optional[str]]) -> tuple[str, str]:
    s3_key, url, domain = item
    s3_client = _get_worker_resources()
    boilerplate_remover = _get_or_create_remover_for_domain(s3_client, domain)

    logger.debug("Minimizing %s", url)
    minimized_html = generate_minimized_html(s3_key, url, s3_client, boilerplate_remover)
    logger.debug("Minimized %s", url)

    minimized_s3_key = upload_minimized_html_to_s3(s3_client, minimized_html, s3_key)
    logger.debug("Uploaded %s", url)

    return minimized_s3_key, url


if __name__ == "__main__":
    try:
        _require_env("RAW_HTML_S3_BUCKET")
        _require_env("MINIMIZED_HTML_S3_BUCKET")
        _require_env("ANCHOR_TREE_S3_BUCKET")

        items = get_s3_keys_to_minimize()
        pending_count = len(items)
        logger.debug("%s pages to minimize", pending_count)

        _get_anchor_tree_cache_dir()

        completed: list[tuple[str, str]] = []  # (url, minimized_s3_key)

        max_workers = int(_env("MAX_WORKERS", default="4"))
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_process_worker) as executor:
            futures = {executor.submit(_minimize_worker, item): item for item in items}

            for future in as_completed(futures):
                source_s3_key, url, _ = futures[future]
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
