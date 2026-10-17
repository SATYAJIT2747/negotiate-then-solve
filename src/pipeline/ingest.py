"""Channels and the central request store (proposal L0-L1).

Every channel produces the same ``Request``: sender mapped from email to
person and role, text normalised, replies threaded, duplicates dropped.

* email: RFC 822 messages (from a file, or polled over IMAP);
* messaging: Slack/Teams/WhatsApp-shaped JSON from mock adapters;
* portal: an authenticated user id (college SSO in production).

Unknown senders are rejected here: identity comes from the directory, never
from what a message says about itself.
"""

from __future__ import annotations

import email
import email.policy
import hashlib
import imaplib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parseaddr, parsedate_to_datetime

from core.instance import Instance
from core.schemas import Channel, Request, Role

from .store import Store

DOMAIN = "college.edu"


@dataclass(frozen=True)
class Identity:
    person: str
    role: Role


class Directory:
    """Email -> person -> role. Built from the instance for synthetic runs;
    in production it comes from the college directory (Google Workspace /
    Microsoft 365)."""

    def __init__(self, entries: dict[str, Identity]) -> None:
        self.by_email = {k.lower(): v for k, v in entries.items()}

    @classmethod
    def from_instance(cls, instance: Instance, domain: str = DOMAIN) -> Directory:
        entries: dict[str, Identity] = {}
        for f in instance.faculty:
            local = re.sub(r"[^a-z0-9]+", ".", f.name.lower().removeprefix("dr. ")).strip(".")
            entries[f"{local}@{domain}"] = Identity(f.id, f.role)
        for g in instance.groups:
            entries[f"cr.{g.id.lower()}@{domain}"] = Identity(f"ST-{g.id}", Role.STUDENT)
        entries[f"labs@{domain}"] = Identity("S-LAB", Role.LAB_INCHARGE)
        entries[f"timetable@{domain}"] = Identity("C-TT", Role.COORDINATOR)
        entries[f"exams@{domain}"] = Identity("S-EXAM", Role.EXAM_CELL)
        return cls(entries)

    def lookup(self, address: str) -> Identity | None:
        return self.by_email.get(parseaddr(address)[1].lower())

    def email_of(self, person: str) -> str | None:
        return next((e for e, i in self.by_email.items() if i.person == person), None)


class UnknownSender(ValueError):
    pass


_QUOTE = re.compile(r"(?ms)^(On .+ wrote:|-----Original Message-----|From: ).*\Z")


def normalise(text: str) -> str:
    """Strip quoted replies and signatures' trailing whitespace; keep the rest verbatim."""
    text = _QUOTE.sub("", text.replace("\r\n", "\n"))
    text = "\n".join(line for line in text.splitlines() if not line.startswith(">"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def dedupe_key(sender: str, text: str) -> str:
    return hashlib.sha256(f"{sender}\n{' '.join(text.lower().split())}".encode()).hexdigest()


class Intake:
    def __init__(self, directory: Directory, store: Store) -> None:
        self.directory = directory
        self.store = store
        self._threads: dict[str, str] = {}  # message-id -> thread id

    def _make(self, channel: Channel, ident: Identity, text: str, when: datetime,
              thread: str | None) -> Request | None:
        text = normalise(text)
        r = self.store.add_new_request(
            lambda rid: Request(id=rid, channel=channel, sender_id=ident.person, role=ident.role, raw_text=text,
                                thread_id=thread or rid, received_at=when),
            dedupe_key(ident.person, text))
        if r is None:
            self.store.log(None, "duplicate_dropped", sender=ident.person, channel=channel.value)
        return r

    def from_email(self, raw: bytes | str) -> Request | None:
        msg = (email.message_from_bytes(raw, policy=email.policy.default) if isinstance(raw, bytes)
               else email.message_from_string(raw, policy=email.policy.default))
        ident = self.directory.lookup(msg.get("From", ""))
        if ident is None:
            self.store.log(None, "unknown_sender", channel="email", sender=parseaddr(msg.get("From", ""))[1])
            raise UnknownSender(msg.get("From", ""))
        body = msg.get_body(preferencelist=("plain",))
        text = body.get_content() if body else ""
        subject = str(msg.get("Subject", "")).strip()
        if subject and not re.match(r"(?i)^(re|fwd?):", subject) and subject.lower() not in text.lower():
            text = f"{subject}\n{text}"
        parent = (msg.get("In-Reply-To") or "").strip() or (str(msg.get("References") or "").split() or [""])[-1]
        thread = self._threads.get(parent)
        try:
            when = parsedate_to_datetime(msg["Date"]) if msg.get("Date") else datetime.now(UTC)
        except (TypeError, ValueError):
            when = datetime.now(UTC)
        r = self._make(Channel.EMAIL, ident, text, when, thread)
        if r and msg.get("Message-ID"):
            self._threads[msg["Message-ID"].strip()] = r.thread_id
        return r

    def from_messaging(self, payload: dict) -> Request | None:
        """Slack-shaped: {"user": {"email": ...}, "text": ..., "ts": ..., "thread_ts": ...}."""
        ident = self.directory.lookup(payload.get("user", {}).get("email", ""))
        if ident is None:
            self.store.log(None, "unknown_sender", channel="messaging")
            raise UnknownSender(str(payload.get("user")))
        when = datetime.fromtimestamp(float(payload.get("ts", 0)) or datetime.now(UTC).timestamp(), UTC)
        thread = self._threads.get(str(payload.get("thread_ts"))) if payload.get("thread_ts") else None
        r = self._make(Channel.MESSAGING, ident, payload.get("text", ""), when, thread)
        if r and payload.get("ts"):
            self._threads[str(payload["ts"])] = r.thread_id
        return r

    def from_portal(self, person: str, text: str) -> Request | None:
        """``person`` comes from the authenticated session, not from the form."""
        ident = next((i for i in self.directory.by_email.values() if i.person == person), None)
        if ident is None:
            raise UnknownSender(person)
        return self._make(Channel.PORTAL, ident, text, datetime.now(UTC), None)


def poll_imap(host: str, user: str, password: str, mailbox: str = "INBOX") -> Iterator[bytes]:
    """Yield unseen messages from a dedicated mailbox and mark them seen.
    Feed each to ``Intake.from_email``."""
    with imaplib.IMAP4_SSL(host) as imap:
        imap.login(user, password)
        imap.select(mailbox)
        _, data = imap.search(None, "UNSEEN")
        for num in data[0].split():
            _, parts = imap.fetch(num, "(RFC822)")
            for part in parts:
                if isinstance(part, tuple):
                    yield part[1]
            imap.store(num, "+FLAGS", "\\Seen")
