"""Notifications to *myself*: desktop toast, Slack DM, and the JSONL audit log.

These are the only "writes" that happen in every mode, including ``observe``; they never target
GitHub or the tracker. ``Notifier.notify`` never raises: each channel reports success or an error
string, the audit record is always written, and the caller decides what to do when nothing
user-facing was delivered (e.g. refine: no notification -> no edit).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from .config import Profile

SLACK_API = "https://slack.com/api"
SLACK_TIMEOUT_S = 15.0
TOAST_TIMEOUT_S = 10.0
APP_NAME = "software-factory"


class Audit:
    """Append-only JSONL log: ``{"ts", "profile", "event", ...fields}`` per line."""

    def __init__(self, path: Path, profile: str = "") -> None:
        self.path = Path(path)
        self.profile = profile
        self._lock = threading.Lock()

    def write(self, event: str, **fields: Any) -> None:
        record = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), "profile": self.profile, "event": event}
        record.update(fields)
        line = json.dumps(record, default=str, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def read(self) -> list[dict]:
        """Every record (for tests, ``status`` and ``report``)."""
        if not self.path.is_file():
            return []
        records: list[dict] = []
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if raw:
                records.append(json.loads(raw))
        return records


class Delivered(BaseModel):
    channels: list[str] = Field(default_factory=list)  # subset of "audit", "toast", "slack"
    errors: list[str] = Field(default_factory=list)

    @property
    def user_facing(self) -> bool:
        """True when the toast or the Slack DM was delivered (the audit log alone does not count)."""
        return "toast" in self.channels or "slack" in self.channels


class Notifier:
    """``toaster`` is any object with ``send(title: str, message: str) -> Any`` (the default is a lazily
    built ``desktop_notifier.DesktopNotifierSync``); ``http`` is the Slack client (injectable with
    ``httpx.MockTransport`` in tests)."""

    def __init__(
        self,
        profile: Profile,
        audit: Audit,
        http: httpx.Client | None = None,
        toaster: Any | None = None,
    ) -> None:
        self.profile = profile
        self.audit = audit
        self._http = http
        self._toaster = toaster
        self._toaster_failed: str | None = None

    # --------------------------------------------------------------------------- public
    def notify(self, title: str, body: str, *, target: str = "", kind: str = "", level: str = "info") -> Delivered:
        """Deliver on every enabled channel, always write the audit record, never raise."""
        channels = ["audit"]
        errors: list[str] = []
        toast_title = f"[{level}] {title}" if level and level != "info" else title
        if self.profile.notify.desktop:
            error = self._toast(toast_title, body)
            if error is None:
                channels.append("toast")
            else:
                errors.append(f"toast: {error}")
        if self.profile.notify.slack:
            error = self._slack_dm(_slack_text(title, body, target=target, kind=kind, level=level))
            if error is None:
                channels.append("slack")
            else:
                errors.append(f"slack: {error}")
        try:
            self.audit.write(
                "notify",
                title=title,
                body=body,
                target=target,
                kind=kind,
                level=level,
                channels=channels,
                errors=errors,
            )
        except Exception as exc:  # noqa: BLE001 - the audit log must never take the tick down
            errors.append(f"audit: {exc}")
            channels.remove("audit")
        return Delivered(channels=channels, errors=errors)

    def slack_dm(self, text: str) -> bool:
        """``conversations.open(users=SLACK_USER_ID)`` then ``chat.postMessage``; False on any problem."""
        return self._slack_dm(text) is None

    def toast(self, title: str, body: str) -> bool:
        return self._toast(title, body) is None

    # ---------------------------------------------------------------------------- slack
    def _slack_dm(self, text: str) -> str | None:
        token = self.profile.secret("SLACK_BOT_TOKEN")
        user_id = self.profile.secret("SLACK_USER_ID")
        if not token:
            return "SLACK_BOT_TOKEN missing from the profile .env"
        if not user_id:
            return "SLACK_USER_ID missing from the profile .env"
        try:
            opened = self._slack_call(token, "conversations.open", {"users": user_id})
            if isinstance(opened, str):
                return f"conversations.open failed: {opened}"
            channel = (opened.get("channel") or {}).get("id")
            if not channel:
                return "conversations.open returned no channel id"
            posted = self._slack_call(token, "chat.postMessage", {"channel": channel, "text": text})
            if isinstance(posted, str):
                return f"chat.postMessage failed: {posted}"
        except Exception as exc:  # noqa: BLE001 - network/JSON errors become a recorded error
            return f"{type(exc).__name__}: {exc}"
        return None

    def _slack_call(self, token: str, method: str, payload: dict) -> dict | str:
        """POST a Slack Web API method; returns the JSON body on ``ok`` or an error string."""
        client = self._client()
        response = client.post(
            f"{SLACK_API}/{method}",
            json=payload,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"},
            timeout=SLACK_TIMEOUT_S,
        )
        if response.status_code != 200:
            return f"HTTP {response.status_code}"
        data = response.json()
        if not isinstance(data, dict) or not data.get("ok"):
            return str((data or {}).get("error", "unknown_error")) if isinstance(data, dict) else "bad json"
        return data

    def _client(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(timeout=SLACK_TIMEOUT_S)
        return self._http

    # ---------------------------------------------------------------------------- toast
    def _toast(self, title: str, body: str) -> str | None:
        toaster = self._get_toaster()
        if toaster is None:
            return self._toaster_failed or "desktop notifier unavailable"
        return _call_with_timeout(lambda: toaster.send(title, body), TOAST_TIMEOUT_S)

    def _get_toaster(self) -> Any | None:
        if self._toaster is not None:
            return self._toaster
        if self._toaster_failed is not None:
            return None
        try:
            from desktop_notifier import DesktopNotifierSync

            self._toaster = DesktopNotifierSync(app_name=APP_NAME)
        except Exception as exc:  # noqa: BLE001 - missing backend, no display, etc.
            self._toaster_failed = f"{type(exc).__name__}: {exc}"
            return None
        return self._toaster


def _slack_text(title: str, body: str, *, target: str, kind: str, level: str) -> str:
    prefix = f"[{level}] " if level and level != "info" else ""
    header = f"*{prefix}{title}*"
    meta = " ".join(part for part in (f"`{kind}`" if kind else "", target) if part)
    lines = [header]
    if meta:
        lines.append(meta)
    if body:
        lines.append(body)
    return "\n".join(lines)


def _call_with_timeout(fn: Callable[[], Any], timeout_s: float) -> str | None:
    """Run ``fn`` on a daemon thread (desktop backends can block or need their own event loop);
    return ``None`` on success, else the error string. A hung backend becomes a timeout error."""
    result: dict[str, Any] = {}

    def run() -> None:
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001
            result["error"] = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target=run, name="factory-toast", daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        return f"timed out after {timeout_s:g}s"
    return result.get("error")
