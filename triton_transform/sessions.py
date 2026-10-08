"""Workspace-local, atomic snapshots of explicitly synchronized editor sessions."""

from __future__ import annotations

from contextlib import contextmanager
import errno
import json
import os
import re
import tempfile
import time
from pathlib import Path


_LOCK_TIMEOUT_SECONDS = 5.0


class SessionStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _path(self, session_id: str) -> Path:
        if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", session_id):
            raise ValueError("session_id must contain 1–80 letters, digits, underscores or hyphens")
        return self.root / f"{session_id}.json"

    def get(self, session_id: str) -> dict | None:
        path = self._path(session_id)
        # Missing-session queries should not create directories or lock files.
        if not path.exists():
            return None
        with self._locked(session_id):
            return self._read(session_id)

    def _read(self, session_id: str) -> dict | None:
        """Read a snapshot while the caller already holds its session lock."""
        path = self._path(session_id)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("expected a JSON object")
            json.dumps(data, allow_nan=False)
            return data
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid context for session {session_id}: {exc}") from exc

    def put(self, session_id: str, context: dict) -> bool:
        path = self._path(session_id)
        if not isinstance(context, dict) or not isinstance(context.get("document_id"), str):
            raise ValueError("context must contain a string document_id")
        version = context.get("version")
        if type(version) is not int or version < 0:
            raise ValueError("context.version must be a nonnegative integer")
        payload = json.dumps(context, ensure_ascii=False, allow_nan=False)
        with self._locked(session_id):
            previous = self._read(session_id)
            if previous and previous.get("document_id") == context["document_id"]:
                if previous.get("version", -1) > version:
                    return False
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.root,
                                                 prefix=f".{session_id}-", suffix=".tmp", delete=False) as stream:
                    temporary = stream.name
                    stream.write(payload)
                os.replace(temporary, path)
            finally:
                if temporary and os.path.exists(temporary):
                    os.unlink(temporary)
        return True

    def clear(self, session_id: str) -> None:
        with self._locked(session_id):
            self._path(session_id).unlink(missing_ok=True)

    @contextmanager
    def _locked(self, session_id: str):
        """Keep a stable lock file so all stores/processes share one lock inode.

        Readers also acquire this lock: ordinary Windows file readers prevent
        replacement while open. Put uses the unlocked _read helper under its
        existing lock, avoiding nested lock acquisition.
        """
        path = self._path(session_id).with_suffix('.lock')
        self.root.mkdir(parents=True, exist_ok=True)
        with path.open('a+b') as stream:
            if os.name == 'nt':
                import msvcrt

                if stream.tell() == 0:
                    stream.write(b'\0')
                    stream.flush()

                def acquire():
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)

                def release():
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                def acquire():
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                def release():
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

            deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
            while True:
                try:
                    acquire()
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f'Timed out waiting for session lock: {session_id}') from exc
                    time.sleep(min(0.025, remaining))
            try:
                yield
            finally:
                release()

    def list_sessions(self) -> list[dict]:
        if not self.root.exists():
            return []
        sessions = []
        for path in sorted(self.root.glob("*.json")):
            try:
                context = self.get(path.stem)
            except ValueError:
                continue
            if context is not None:
                sessions.append({"session_id": path.stem,
                                 "document_id": context.get("document_id"),
                                 "version": context.get("version"),
                                 "selected_node_id": context.get("selected_node_id")})
        return sessions
