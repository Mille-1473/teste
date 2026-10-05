"""One-way periodic synchronization of a source folder onto a replica.

Arguments, in order: source folder, replica folder, interval in seconds,
number of synchronizations, log file path.
"""

import hashlib
import logging
import math
import os
import shutil
import stat
import sys
import tempfile
import time
from typing import Optional

CHUNK_SIZE = 1024 * 1024
LOGGER_NAME = "folder_sync"


def file_md5(path: str) -> str:
    """Return the MD5 hex digest of a file, reading it in chunks."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_of_first_file(source_folder: str) -> Optional[str]:
    """Return the MD5 hex digest of the first regular file in *source_folder*.

    Directories and file names are visited in sorted order. Files in a
    directory are considered before its subdirectories. Symbolic links are
    skipped. Returns None when the tree contains no regular file.
    """
    for root, dir_names, file_names in os.walk(source_folder, followlinks=False):
        dir_names.sort()
        file_names.sort()
        for name in file_names:
            candidate = os.path.join(root, name)
            if os.path.isfile(candidate) and not os.path.islink(candidate):
                return file_md5(candidate)
    return None


def paths_conflict(source: str, replica: str) -> bool:
    """Return True when the two folders are the same or one contains the other."""
    source_path = os.path.normcase(os.path.abspath(source))
    replica_path = os.path.normcase(os.path.abspath(replica))
    if source_path == replica_path:
        return True
    source_prefix = source_path.rstrip(os.sep) + os.sep
    replica_prefix = replica_path.rstrip(os.sep) + os.sep
    return source_path.startswith(replica_prefix) or replica_path.startswith(
        source_prefix
    )


def _is_real_dir(path: str) -> bool:
    return os.path.isdir(path) and not os.path.islink(path)


def _is_regular_file(path: str) -> bool:
    return os.path.isfile(path) and not os.path.islink(path)


def _unlink(path: str) -> None:
    try:
        os.remove(path)
    except PermissionError:
        os.chmod(path, stat.S_IWRITE)
        os.remove(path)


def _rmdir(path: str) -> None:
    try:
        os.rmdir(path)
    except PermissionError:
        os.chmod(path, stat.S_IWRITE)
        os.rmdir(path)


def close_logger(logger: logging.Logger) -> None:
    """Detach and close handlers so the log file is not left locked."""
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def build_logger(log_path: str) -> logging.Logger:
    """Create a logger that writes the same lines to *log_path* and stdout."""
    logger = logging.getLogger(LOGGER_NAME)
    close_logger(logger)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    parent = os.path.dirname(os.path.abspath(log_path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)

    formatter = logging.Formatter(
        "%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    for handler in (file_handler, stream_handler):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class FolderSynchronizer:
    """Copy *source* onto *replica* and delete anything the source no longer has."""

    def __init__(self, source: str, replica: str, logger: logging.Logger) -> None:
        self.source = os.path.abspath(source)
        self.replica = os.path.abspath(replica)
        self._logger = logger

    def synchronize(self) -> None:
        self._ensure_replica_root()
        self._sync_dir(self.source, self.replica)

    def remove_path(self, path: str) -> None:
        """Delete a file or directory tree, logging each removal from the bottom."""
        if os.path.islink(path):
            operation = "REMOVED directory" if os.path.isdir(path) else "REMOVED file"
            _unlink(path)
            self._log(operation, path)
            return
        if not os.path.isdir(path):
            _unlink(path)
            self._log("REMOVED file", path)
            return
        for name in os.listdir(path):
            self.remove_path(os.path.join(path, name))
        _rmdir(path)
        self._log("REMOVED directory", path)

    def _ensure_replica_root(self) -> None:
        if _is_real_dir(self.replica):
            return
        self._mkdir_logged(self.replica)

    def _mkdir_logged(self, path: str) -> None:
        parent = os.path.dirname(path)
        if parent and parent != path and not _is_real_dir(parent):
            self._mkdir_logged(parent)
        os.mkdir(path)
        self._log("CREATED directory", path)

    def _sync_dir(self, source_dir: str, replica_dir: str) -> None:
        source_names = set(os.listdir(source_dir))
        replica_names = set(os.listdir(replica_dir))
        for name in sorted(replica_names - source_names):
            self.remove_path(os.path.join(replica_dir, name))
        for name in sorted(source_names):
            self._sync_entry(
                os.path.join(source_dir, name),
                os.path.join(replica_dir, name),
            )

    def _sync_entry(self, source_path: str, replica_path: str) -> None:
        if os.path.islink(source_path):
            self._sync_symlink(source_path, replica_path)
        elif os.path.isdir(source_path):
            self._sync_directory(source_path, replica_path)
        elif os.path.isfile(source_path):
            self._sync_file(source_path, replica_path)

    def _sync_directory(self, source_path: str, replica_path: str) -> None:
        if os.path.lexists(replica_path) and not _is_real_dir(replica_path):
            self.remove_path(replica_path)
        if not _is_real_dir(replica_path):
            os.mkdir(replica_path)
            self._log("CREATED directory", replica_path)
        self._sync_dir(source_path, replica_path)

    def _sync_file(self, source_path: str, replica_path: str) -> None:
        if os.path.lexists(replica_path) and not _is_regular_file(replica_path):
            self.remove_path(replica_path)
        if _same_file_content(source_path, replica_path):
            return
        self._copy_file(source_path, replica_path)

    def _sync_symlink(self, source_path: str, replica_path: str) -> None:
        target = os.readlink(source_path)
        if os.path.islink(replica_path) and os.readlink(replica_path) == target:
            return
        if os.path.lexists(replica_path):
            self.remove_path(replica_path)
        os.symlink(
            target,
            replica_path,
            target_is_directory=os.path.isdir(source_path),
        )
        self._log("COPIED file", replica_path)

    def _copy_file(self, source_path: str, replica_path: str) -> None:
        parent = os.path.dirname(replica_path)
        descriptor, temp_path = tempfile.mkstemp(prefix=".sync-", dir=parent)
        os.close(descriptor)
        try:
            shutil.copy2(source_path, temp_path)
            if os.path.lexists(replica_path) and not os.path.islink(replica_path):
                try:
                    os.chmod(replica_path, stat.S_IWRITE | stat.S_IREAD)
                except OSError:
                    pass
            os.replace(temp_path, replica_path)
        except Exception:
            try:
                _unlink(temp_path)
            except OSError:
                pass
            raise
        self._log("COPIED file", replica_path)

    def _log(self, operation: str, path: str) -> None:
        self._logger.info("%s %s", operation, os.path.abspath(path))


def _same_file_content(source_path: str, replica_path: str) -> bool:
    if not _is_regular_file(replica_path):
        return False
    if os.path.getsize(source_path) != os.path.getsize(replica_path):
        return False
    return file_md5(source_path) == file_md5(replica_path)


def _parse_count(text: str) -> Optional[int]:
    digits = text[1:] if text[:1] in "+-" else text
    if not digits.isdigit():
        return None
    return int(text)


def parse_arguments(argv):
    """Parse the five positional arguments. Return None when they are invalid."""
    if len(argv) != 6:
        print(
            "Usage: python folder_sync.py <source> <replica> "
            "<interval> <count> <log_file>",
            file=sys.stderr,
        )
        return None

    source, replica, interval_text, count_text, log_path = argv[1:6]
    if not source or not replica or not log_path:
        print("Folder and log paths must not be empty.", file=sys.stderr)
        return None

    try:
        interval = float(interval_text)
    except ValueError:
        interval = None
    if interval is None or not math.isfinite(interval) or interval < 0:
        print("Interval must be a non-negative number of seconds.", file=sys.stderr)
        return None

    count = _parse_count(count_text)
    if count is None:
        print("Synchronization count must be an integer.", file=sys.stderr)
        return None
    if count < 0:
        print("Synchronization count must be a non-negative integer.", file=sys.stderr)
        return None

    return source, replica, interval, count, log_path


def run(source: str, replica: str, interval: float, count: int, log_path: str) -> None:
    """Synchronize *source* onto *replica* exactly *count* times."""
    source_abs = os.path.abspath(source)
    replica_abs = os.path.abspath(replica)

    if not _is_real_dir(source_abs):
        print(
            f"Source folder does not exist or is not a directory: {source_abs}",
            file=sys.stderr,
        )
        return
    if paths_conflict(source_abs, replica_abs):
        print(
            "Source and replica must be different folders and must not contain each other.",
            file=sys.stderr,
        )
        return
    if os.path.lexists(replica_abs) and not _is_real_dir(replica_abs):
        print(
            f"Replica path exists and is not a directory: {replica_abs}",
            file=sys.stderr,
        )
        return

    try:
        logger = build_logger(log_path)
    except OSError as error:
        print(f"Cannot open log file: {error}", file=sys.stderr)
        return

    try:
        synchronizer = FolderSynchronizer(source_abs, replica_abs, logger)
        for index in range(count):
            synchronizer.synchronize()
            # Sleep between runs only; stop immediately after the last one.
            if index < count - 1:
                time.sleep(interval)
    finally:
        close_logger(logger)


def main() -> None:
    """Entry point. Reads command-line arguments and runs a finite sync schedule."""
    arguments = parse_arguments(sys.argv)
    if arguments is None:
        return
    run(*arguments)


if __name__ == "__main__":
    main()
