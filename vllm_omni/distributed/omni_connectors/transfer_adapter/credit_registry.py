# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Cross-process per-stream credit counters for producer-consumer flow control.

Counts ``puts`` (Stage 0 -> connector) and ``gets`` (connector -> Stage 1)
per ``external_req_id``. Stage 0 can read ``outstanding = puts - gets`` to
decide whether to skip a stream's LM step (see WS-4 in RFC #3535).

Each request owns one fixed-size shared-memory segment. Increments use
``fcntl.flock`` to match the convention in ``SharedMemoryConnector``.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import struct
import tempfile
from multiprocessing import shared_memory as shm_pkg
from pathlib import Path

from ..utils.logging import get_connector_logger

logger = get_connector_logger(__name__)

_COUNTER_SIZE = 16  # two u64
_COUNTER_STRUCT = struct.Struct("<QQ")
# Short prefix + 24-hex-char hash keeps the full SHM name well under the
# macOS POSIX shm_open limit (31 chars incl. the leading slash) while
# avoiding collisions in practice.
_NAME_PREFIX = "vomc_"
_HASH_LEN = 24


def _safe_segment_name(external_req_id: str) -> str:
    digest = hashlib.sha256(external_req_id.encode("utf-8")).hexdigest()
    return f"{_NAME_PREFIX}{digest[:_HASH_LEN]}"


def _lock_dir() -> str:
    # /dev/shm on Linux (matches SharedMemoryConnector); tempdir elsewhere.
    if os.path.isdir("/dev/shm"):
        return "/dev/shm"
    return tempfile.gettempdir()


def _lock_path(segment_name: str) -> str:
    return os.path.join(_lock_dir(), f"{segment_name}.lock")


class StreamCreditRegistry:
    """Cross-process per-stream credit counter.

    Safe to call from either Stage 0 (producer) or Stage 1 (consumer)
    processes. Segments are created on first ``inc_put``; ``drop`` cleans
    them up after the request finishes. A registry constructed with
    ``cap=0`` short-circuits all calls (legacy behavior).
    """

    def __init__(self, cap: int = 0) -> None:
        if cap < 0:
            raise ValueError("cap must be >= 0")
        self.cap = cap
        self._enabled = cap > 0
        self._segments: dict[str, shm_pkg.SharedMemory] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def inc_put(self, external_req_id: str) -> None:
        if not self._enabled:
            return
        self._inc(external_req_id, slot=0)

    def inc_get(self, external_req_id: str) -> None:
        if not self._enabled:
            return
        self._inc(external_req_id, slot=1)

    def outstanding(self, external_req_id: str) -> int:
        """Return ``puts - gets``. 0 when the segment does not exist
        (treats unknown streams as unconstrained)."""
        if not self._enabled:
            return 0
        seg = self._open(external_req_id, create=False)
        if seg is None:
            return 0
        try:
            lock_file = _lock_path(seg.name)
            with open(lock_file, "wb+") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_SH)
                puts, gets = _COUNTER_STRUCT.unpack_from(seg.buf, 0)
                fcntl.flock(lockf, fcntl.LOCK_UN)
        except Exception:  # pragma: no cover - defensive
            logger.debug("credit_registry: outstanding read failed for %s", external_req_id, exc_info=True)
            return 0
        return max(0, int(puts) - int(gets))

    def is_blocked(self, external_req_id: str) -> bool:
        if not self._enabled:
            return False
        return self.outstanding(external_req_id) >= self.cap

    def drop(self, external_req_id: str) -> None:
        """Unlink the segment + lock for a request. Idempotent."""
        if not self._enabled:
            return
        name = _safe_segment_name(external_req_id)
        seg = self._segments.pop(external_req_id, None)
        if seg is None:
            try:
                seg = shm_pkg.SharedMemory(name=name)
            except FileNotFoundError:
                seg = None
            except Exception:  # pragma: no cover
                logger.debug("credit_registry: open-on-drop failed for %s", name, exc_info=True)
                seg = None
        if seg is not None:
            try:
                seg.close()
                seg.unlink()
            except FileNotFoundError:
                pass
            except Exception:  # pragma: no cover
                logger.debug("credit_registry: unlink failed for %s", name, exc_info=True)
        lock_file = _lock_path(name)
        try:
            os.remove(lock_file)
        except FileNotFoundError:
            pass
        except OSError:  # pragma: no cover
            pass

    def close(self) -> None:
        """Drop all segments owned by this process. Used on adapter shutdown."""
        for external_req_id in list(self._segments.keys()):
            self.drop(external_req_id)

    @staticmethod
    def gc_orphaned_segments() -> None:
        """Remove leaked credit segments from previous crashes.

        Only matches our prefix. Safe to call at adapter startup. On Linux
        the SHM data lives in ``/dev/shm`` alongside our lock files; on
        platforms without ``/dev/shm`` we still clean up the lock files.
        """
        dirs = []
        if os.path.isdir("/dev/shm"):
            dirs.append("/dev/shm")
        tmp = tempfile.gettempdir()
        if tmp not in dirs:
            dirs.append(tmp)
        for d in dirs:
            try:
                for entry in Path(d).iterdir():
                    name = entry.name
                    if not name.startswith(_NAME_PREFIX):
                        continue
                    base = name[: -len(".lock")] if name.endswith(".lock") else name
                    if not base.startswith(_NAME_PREFIX):
                        continue
                    try:
                        entry.unlink()
                    except FileNotFoundError:
                        pass
                    except OSError:
                        pass
            except FileNotFoundError:
                continue
            except Exception:  # pragma: no cover
                logger.debug("credit_registry: gc_orphaned_segments failed for %s", d, exc_info=True)

    def _inc(self, external_req_id: str, slot: int) -> None:
        seg = self._open(external_req_id, create=True)
        if seg is None:
            return
        lock_file = _lock_path(seg.name)
        try:
            with open(lock_file, "wb+") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX)
                puts, gets = _COUNTER_STRUCT.unpack_from(seg.buf, 0)
                if slot == 0:
                    puts += 1
                else:
                    gets += 1
                _COUNTER_STRUCT.pack_into(seg.buf, 0, puts, gets)
                fcntl.flock(lockf, fcntl.LOCK_UN)
        except Exception:  # pragma: no cover - defensive
            logger.debug("credit_registry: inc failed for %s slot=%d", external_req_id, slot, exc_info=True)

    def _open(self, external_req_id: str, create: bool) -> shm_pkg.SharedMemory | None:
        cached = self._segments.get(external_req_id)
        if cached is not None:
            return cached
        name = _safe_segment_name(external_req_id)
        try:
            seg = shm_pkg.SharedMemory(name=name)
        except FileNotFoundError:
            if not create:
                return None
            try:
                seg = shm_pkg.SharedMemory(name=name, create=True, size=_COUNTER_SIZE)
                _COUNTER_STRUCT.pack_into(seg.buf, 0, 0, 0)
            except FileExistsError:
                # Race: another process created it between our two opens.
                seg = shm_pkg.SharedMemory(name=name)
            except Exception:  # pragma: no cover
                logger.debug("credit_registry: create failed for %s", name, exc_info=True)
                return None
        except Exception:  # pragma: no cover
            logger.debug("credit_registry: open failed for %s", name, exc_info=True)
            return None
        self._segments[external_req_id] = seg
        return seg
