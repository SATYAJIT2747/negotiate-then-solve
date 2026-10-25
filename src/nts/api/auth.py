"""Who is calling: people sign in, and the server decides from the session what they may do.

    users = Users.load("runs/agentic/users.json")    # {"<dept>/<name>": "pbkdf2_sha256$<iterations>$<salt>$<hash>"}
    users.set("demo", "Dr. Iyer", "a passphrase"); users.save()
    sessions = Sessions(users)                       # Users.open(): anyone in a department's directory, no password

Roles: a teacher starts requests as themselves, sees their own requests and what waits for their answer,
and answers asks addressed to them; the timetable office (sign in as "office") sees everything, starts
requests for the office and approves or rejects drafts. A session is a random token held server-side
and sent as an HttpOnly cookie (or ``Authorization: Bearer``), so a client cannot choose who it is.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from nts.tools.draft import OFFICE

ITERATIONS = 200_000
SESSION_HOURS = 12
OFFICE_NAME = "the timetable office"


def hash_password(password: str, salt: str | None = None, iterations: int = ITERATIONS) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), iterations).hex()
    return f"pbkdf2_sha256${iterations}${salt}${digest}"


def check_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations, salt, _ = stored.split("$")
    except ValueError:
        return False
    return scheme == "pbkdf2_sha256" and hmac.compare_digest(hash_password(password, salt, int(iterations)), stored)


class Users:
    """Passwords by "<department>/<person>"; ``open`` (no file): anyone named in a directory, no password."""

    def __init__(self, hashes: dict[str, str] | None, path: Path | None = None) -> None:
        self.hashes, self.path = hashes, path

    @classmethod
    def open(cls) -> Users:
        return cls(None)

    @classmethod
    def load(cls, path: Path | str) -> Users:
        path = Path(path)
        return cls(json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}, path)

    @property
    def needs_password(self) -> bool:
        return self.hashes is not None

    def set(self, dept: str, person: str, password: str) -> None:
        assert self.hashes is not None, "an open sign-in has no passwords"
        self.hashes[f"{dept}/{person}"] = hash_password(password)

    def save(self) -> None:
        assert self.path is not None and self.hashes is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.hashes, indent=1, sort_keys=True), encoding="utf-8")

    def check(self, dept: str, person: str, password: str) -> bool:
        if self.hashes is None:
            return True
        stored = self.hashes.get(f"{dept}/{person}")
        if stored is None:
            check_password(password, hash_password("timing"))  # the same work for an unknown name
            return False
        return check_password(password, stored)


@dataclass(frozen=True)
class Session:
    token: str
    dept: str
    person: str  # a faculty id, or OFFICE
    name: str
    expires: float

    @property
    def office(self) -> bool:
        return self.person == OFFICE

    def view(self) -> dict:
        return {"dept": self.dept, "name": self.name, "role": "office" if self.office else "teacher"}


class Sessions:
    def __init__(self, users: Users) -> None:
        self.users = users
        self._by_token: dict[str, Session] = {}
        self._lock = threading.Lock()

    def start(self, dept: str, person: str, name: str, password: str) -> Session | None:
        """A new session for ``name`` (as the directory spells it, or "office"), or None if the password is wrong."""
        if not self.users.check(dept, "office" if person == OFFICE else name, password):
            return None
        s = Session(secrets.token_urlsafe(32), dept, person, OFFICE_NAME if person == OFFICE else name,
                    time.time() + SESSION_HOURS * 3600)
        with self._lock:
            self._by_token[s.token] = s
        return s

    def get(self, token: str | None) -> Session | None:
        if not token:
            return None
        with self._lock:
            s = self._by_token.get(token)
            if s is not None and s.expires < time.time():
                del self._by_token[token]
                return None
            return s

    def end(self, token: str) -> None:
        with self._lock:
            self._by_token.pop(token, None)
