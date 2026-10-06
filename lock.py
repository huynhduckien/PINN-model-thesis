"""Khoa GPU: mot tien trinh GPU tai mot thoi diem (prereg 01, quy tac chung 2). Dung: `with gpu_lock(): ...`.

Tep .run.lock chua PID. Neu PID do khong con song thi khoa duoc coi la cu va bi thay.
"""
import contextlib
import os
from pathlib import Path

import psutil

LOCK = Path(__file__).resolve().parent/".run.lock"


class Busy(RuntimeError):
    pass


@contextlib.contextmanager
def gpu_lock():
    if LOCK.exists():
        try:
            pid = int(LOCK.read_text().strip())
        except ValueError:
            pid = -1
        if pid > 0 and psutil.pid_exists(pid):
            raise Busy(f"GPU dang duoc giu boi PID {pid} ({LOCK}); tien trinh nay tu choi chay")
        LOCK.unlink()
    fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)       # O_EXCL: hai tien trinh khoi dong cung luc, chi mot thang
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        yield
    finally:
        if LOCK.exists() and LOCK.read_text().strip() == str(os.getpid()):
            LOCK.unlink()
