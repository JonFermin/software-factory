"""notify.py: audit JSONL, Slack DM via MockTransport, injected toaster, never raises."""

from __future__ import annotations

import json
import threading
import time

import httpx
import pytest

from software_factory import notify as notify_mod
from software_factory.config import Profile
from software_factory.notify import Audit, Delivered, Notifier


class FakeToaster:
    def __init__(self, fail: Exception | None = None, hang_s: float = 0.0) -> None:
        self.sent: list[tuple[str, str]] = []
        self.fail = fail
        self.hang_s = hang_s

    def send(self, title: str, message: str) -> str:
        if self.hang_s:
            time.sleep(self.hang_s)
        if self.fail:
            raise self.fail
        self.sent.append((title, message))
        return "id-1"


class SlackServer:
    """MockTransport handler recording every call; configurable failures."""

    def __init__(self, *, open_ok: bool = True, post_ok: bool = True, status: int = 200, channel: str = "D123") -> None:
        self.calls: list[httpx.Request] = []
        self.open_ok, self.post_ok, self.status, self.channel = open_ok, post_ok, status, channel

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if self.status != 200:
            return httpx.Response(self.status, text="nope")
        if request.url.path == "/api/conversations.open":
            body = {"ok": True, "channel": {"id": self.channel}} if self.open_ok else {"ok": False, "error": "user_not_found"}
        elif request.url.path == "/api/chat.postMessage":
            body = {"ok": True, "ts": "1.2"} if self.post_ok else {"ok": False, "error": "not_in_channel"}
        else:
            body = {"ok": False, "error": "unknown_method"}
        return httpx.Response(200, json=body)

    def payload(self, index: int) -> dict:
        return json.loads(self.calls[index].content)


@pytest.fixture
def slack() -> SlackServer:
    return SlackServer()


@pytest.fixture
def notifier(profile: Profile, tmp_path, slack: SlackServer):
    audit = Audit(tmp_path / "logs dir" / "audit.jsonl", profile=profile.name)
    toaster = FakeToaster()
    http = httpx.Client(transport=httpx.MockTransport(slack))
    n = Notifier(profile, audit, http=http, toaster=toaster)
    n.fake_toaster = toaster  # type: ignore[attr-defined]
    return n


# ----------------------------------------------------------------------------------- audit


def test_audit_appends_jsonl_with_ts_and_profile(tmp_path):
    audit = Audit(tmp_path / "deep" / "audit.jsonl", profile="personal")
    audit.write("push", repo="o/r", number=1, shas=["a"], path=tmp_path)
    audit.write("flag", reason="x")
    lines = audit.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["event"] == "push" and first["profile"] == "personal" and first["repo"] == "o/r"
    assert first["ts"].endswith("+00:00") and first["path"] == str(tmp_path)  # non-JSON types stringified
    assert audit.read()[1] == {**json.loads(lines[1])}
    assert Audit(tmp_path / "missing.jsonl").read() == []


# ---------------------------------------------------------------------------------- notify


def test_notify_delivers_on_every_channel_and_audits(notifier, slack):
    delivered = notifier.notify("CI fixed", "pushed abc", target="o/r#1", kind="ci")

    assert delivered == Delivered(channels=["audit", "toast", "slack"], errors=[])
    assert delivered.user_facing
    assert notifier.fake_toaster.sent == [("CI fixed", "pushed abc")]
    assert [c.url.path for c in slack.calls] == ["/api/conversations.open", "/api/chat.postMessage"]
    assert all(c.headers["Authorization"] == "Bearer xoxb-test" for c in slack.calls)
    assert slack.payload(0) == {"users": "U123"}
    post = slack.payload(1)
    assert post["channel"] == "D123"
    assert post["text"] == "*CI fixed*\n`ci` o/r#1\npushed abc"
    record = notifier.audit.read()[-1]
    assert record["event"] == "notify" and record["channels"] == ["audit", "toast", "slack"]
    assert record["title"] == "CI fixed" and record["target"] == "o/r#1" and record["kind"] == "ci"
    assert record["level"] == "info" and record["errors"] == []


def test_notify_level_prefixes_titles(notifier, slack):
    notifier.notify("Auth failed", "gh", level="error")
    assert notifier.fake_toaster.sent == [("[error] Auth failed", "gh")]
    assert slack.payload(1)["text"].startswith("*[error] Auth failed*")


def test_notify_respects_disabled_channels(notifier, slack):
    notifier.profile.notify.desktop = False
    notifier.profile.notify.slack = False
    delivered = notifier.notify("t", "b")
    assert delivered.channels == ["audit"] and delivered.errors == []
    assert not delivered.user_facing
    assert notifier.fake_toaster.sent == [] and slack.calls == []
    assert notifier.audit.read()[-1]["channels"] == ["audit"]


def test_missing_slack_secrets_are_recorded_errors_not_exceptions(notifier, slack):
    notifier.profile.secrets.pop("SLACK_BOT_TOKEN")
    delivered = notifier.notify("t", "b")
    assert delivered.channels == ["audit", "toast"]
    assert delivered.errors == ["slack: SLACK_BOT_TOKEN missing from the profile .env"]
    assert delivered.user_facing  # toast still counts
    assert slack.calls == []
    notifier.profile.secrets["SLACK_BOT_TOKEN"] = "xoxb"
    notifier.profile.secrets.pop("SLACK_USER_ID")
    assert notifier.notify("t", "b").errors == ["slack: SLACK_USER_ID missing from the profile .env"]
    assert slack.calls == []


def test_slack_api_errors_are_checked_via_ok(profile, tmp_path):
    audit = Audit(tmp_path / "a.jsonl")

    def build(server: SlackServer) -> Notifier:
        return Notifier(profile, audit, http=httpx.Client(transport=httpx.MockTransport(server)), toaster=FakeToaster())

    bad_open = SlackServer(open_ok=False)
    delivered = build(bad_open).notify("t", "b")
    assert delivered.errors == ["slack: conversations.open failed: user_not_found"]
    assert len(bad_open.calls) == 1  # never posts without a channel

    bad_post = SlackServer(post_ok=False)
    assert build(bad_post).notify("t", "b").errors == ["slack: chat.postMessage failed: not_in_channel"]

    http_500 = SlackServer(status=500)
    assert build(http_500).notify("t", "b").errors == ["slack: conversations.open failed: HTTP 500"]

    no_channel = SlackServer(channel="")
    assert build(no_channel).notify("t", "b").errors == ["slack: conversations.open returned no channel id"]


def test_slack_transport_exception_is_recorded(profile, tmp_path):
    def boom(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no network")

    n = Notifier(profile, Audit(tmp_path / "a.jsonl"), http=httpx.Client(transport=httpx.MockTransport(boom)), toaster=FakeToaster())
    delivered = n.notify("t", "b")
    assert delivered.channels == ["audit", "toast"]
    assert delivered.errors == ["slack: ConnectError: no network"]
    assert n.slack_dm("x") is False


def test_toaster_exception_is_recorded_and_slack_still_delivers(profile, tmp_path, slack):
    n = Notifier(
        profile,
        Audit(tmp_path / "a.jsonl"),
        http=httpx.Client(transport=httpx.MockTransport(slack)),
        toaster=FakeToaster(fail=RuntimeError("no notification center")),
    )
    delivered = n.notify("t", "b")
    assert delivered.channels == ["audit", "slack"]
    assert delivered.errors == ["toast: RuntimeError: no notification center"]
    assert delivered.user_facing
    assert n.toast("t", "b") is False


def test_hung_toaster_times_out_instead_of_blocking(profile, tmp_path, monkeypatch):
    monkeypatch.setattr(notify_mod, "TOAST_TIMEOUT_S", 0.05)
    n = Notifier(profile, Audit(tmp_path / "a.jsonl"), toaster=FakeToaster(hang_s=2.0))
    n.profile.notify.slack = False
    started = time.monotonic()
    delivered = n.notify("t", "b")
    assert time.monotonic() - started < 1.5
    assert delivered.channels == ["audit"]
    assert delivered.errors == ["toast: timed out after 0.05s"]


def test_default_toaster_construction_failure_is_an_error_not_an_exception(profile, tmp_path, monkeypatch):
    import desktop_notifier

    def explode(*_a, **_k):
        raise OSError("no backend")

    monkeypatch.setattr(desktop_notifier, "DesktopNotifierSync", explode)
    n = Notifier(profile, Audit(tmp_path / "a.jsonl"))
    n.profile.notify.slack = False
    delivered = n.notify("t", "b")
    assert delivered.channels == ["audit"]
    assert delivered.errors == ["toast: OSError: no backend"]
    assert n.toast("again", "b") is False  # failure is remembered; no retry storm
    assert not delivered.user_facing


def test_notify_never_raises_even_when_audit_fails(profile, tmp_path, slack):
    class BrokenAudit(Audit):
        def write(self, event: str, **fields) -> None:
            raise OSError("disk full")

    n = Notifier(profile, BrokenAudit(tmp_path / "a.jsonl"), http=httpx.Client(transport=httpx.MockTransport(slack)), toaster=FakeToaster())
    delivered = n.notify("t", "b")
    assert delivered.channels == ["toast", "slack"]
    assert delivered.errors == ["audit: disk full"]


def test_slack_dm_and_toast_bool_wrappers(notifier, slack):
    assert notifier.slack_dm("hello") is True
    assert slack.payload(1) == {"channel": "D123", "text": "hello"}
    assert notifier.toast("hi", "there") is True
    assert notifier.fake_toaster.sent == [("hi", "there")]
    assert notifier.audit.read() == []  # the raw wrappers do not audit; notify() does


def test_delivered_user_facing_semantics():
    assert not Delivered(channels=["audit"]).user_facing
    assert Delivered(channels=["audit", "toast"]).user_facing
    assert Delivered(channels=["slack"], errors=["toast: x"]).user_facing
    assert not Delivered().user_facing


def test_audit_write_is_thread_safe(tmp_path):
    audit = Audit(tmp_path / "a.jsonl", profile="p")

    def worker(i: int) -> None:
        for j in range(20):
            audit.write("evt", i=i, j=j)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    records = audit.read()
    assert len(records) == 120 and all(r["event"] == "evt" for r in records)
