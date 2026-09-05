import asyncio
import json
import os
from typing import cast
from urllib.error import HTTPError
from urllib.request import urlopen


_host: str | None = None
_UNSET = object()
_ecs_task_id: str | None | object = _UNSET


async def get_host() -> str:
    global _host

    if _host is not None:
        return _host

    _host = os.getenv("RUNNING_MODE")
    if os.getenv("RUNNING_MODE") == "ecs":
        ecs_task_id = await get_ecs_task_id()
        if ecs_task_id:
            _host = ecs_task_id

    return _host or "unknown"


async def get_region() -> str:
    return os.getenv("AWS_REGION") or "unknown"


async def get_ecs_task_id() -> str | None:
    global _ecs_task_id

    if _ecs_task_id is not _UNSET:
        return cast(str | None, _ecs_task_id)

    metadata_uri = os.getenv("ECS_CONTAINER_METADATA_URI_V4")
    if not metadata_uri:
        _ecs_task_id = None
        return None

    task_url = f"{metadata_uri.rstrip('/')}/task"

    def _fetch_task_metadata() -> dict:
        with urlopen(task_url, timeout=5) as response:
            status = getattr(response, "status", 200)
            reason = getattr(response, "reason", "")
            if status < 200 or status >= 300:
                raise RuntimeError(f"Failed to read ECS task metadata: {status} {reason}")

            return json.loads(response.read().decode("utf-8"))

    try:
        data = await asyncio.to_thread(_fetch_task_metadata)
    except HTTPError as error:
        raise RuntimeError(
            f"Failed to read ECS task metadata: {error.code} {error.reason}"
        ) from error

    task_arn = data.get("TaskARN")
    if not isinstance(task_arn, str):
        _ecs_task_id = None
        return None

    _ecs_task_id = task_arn.split("/")[-1]
    return cast(str | None, _ecs_task_id)

