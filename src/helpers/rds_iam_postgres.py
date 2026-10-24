import os
import time
from typing import Any, Optional

import boto3


class RdsIamPostgresCredentials:
    """Resolve PostgreSQL credentials with optional RDS IAM auth token support."""

    DEFAULT_PG_HOST = "localhost"
    DEFAULT_PG_PORT = 5432
    DEFAULT_PG_USER = "furniture_crawler"
    DEFAULT_PG_PASSWORD = "Test@123"
    DEFAULT_PG_DATABASE = "furniture_crawler"

    RDS_IAM_TOKEN_TTL_SECONDS = 15 * 60
    RDS_IAM_TOKEN_REFRESH_BUFFER_SECONDS = 60

    def __init__(self) -> None:
        self._cached_token: Optional[str] = None
        self._token_expires_at: float = 0.0

    def get_connection_kwargs(self) -> dict[str, Any]:
        host = os.getenv("PGHOST") or os.getenv("PG_HOST") or self.DEFAULT_PG_HOST
        port = int(os.getenv("PGPORT") or os.getenv("PG_PORT") or self.DEFAULT_PG_PORT)
        user = os.getenv("RDS_IAM_USER") or os.getenv("PGUSER") or os.getenv("PG_USER") or self.DEFAULT_PG_USER
        database = os.getenv("PGDATABASE") or os.getenv("PG_DATABASE") or self.DEFAULT_PG_DATABASE

        password = os.getenv("PGPASSWORD") or os.getenv("PG_PASSWORD")
        should_use_rds_iam_token = bool(os.getenv("RDS_IAM_USER")) or host != self.DEFAULT_PG_HOST

        if not password:
            password = (
                self._get_rds_iam_token(host, port, user)
                if should_use_rds_iam_token
                else self.DEFAULT_PG_PASSWORD
            )

        return {
            "host": host,
            "port": port,
            "dbname": database,
            "user": user,
            "password": password,
        }

    def _get_rds_iam_token(self, host: str, port: int, username: str) -> str:
        now = time.time()
        refresh_at = self._token_expires_at - self.RDS_IAM_TOKEN_REFRESH_BUFFER_SECONDS

        if self._cached_token and now < refresh_at:
            return self._cached_token

        region = os.getenv("AWS_REGION") or "eu-west-1"
        rds_client = boto3.client("rds", region_name=region)
        token = rds_client.generate_db_auth_token(
            DBHostname=host,
            Port=port,
            DBUsername=username,
            Region=region,
        )

        self._cached_token = token
        self._token_expires_at = now + self.RDS_IAM_TOKEN_TTL_SECONDS
        return token

