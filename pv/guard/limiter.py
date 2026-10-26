"""Simulated edge device: CPU affinity + Windows Job Object memory cap over the app and Ollama tree."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

import psutil
import pywintypes
import win32api
import win32con
import win32job

ROOT = Path(__file__).resolve().parents[2]
MB = 2**20


def physical_cpus() -> list[int]:
    """One logical CPU per physical core (SMT siblings are adjacent on this platform)."""
    phys, logical = psutil.cpu_count(logical=False) or 1, psutil.cpu_count(logical=True) or 1
    step = max(logical // phys, 1)
    return list(range(0, logical, step))


def commit_free_mb() -> float:
    """System-wide free commit (RAM + page file). Allocation failures begin when this nears zero."""
    import ctypes

    class MemStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in
                    ("total_phys", "avail_phys", "total_page", "avail_page", "total_virt", "avail_virt", "avail_ext")]
    m = MemStatus()
    m.length = ctypes.sizeof(MemStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.avail_page / MB


def ollama_pid() -> int | None:
    f = ROOT / "data" / "ollama.pid"
    return int(f.read_text().strip()) if f.exists() else None


class Limiter:
    def __init__(self, cores: int, ram_mb: int, include_ollama: bool = True, split: bool = True) -> None:
        self.include_ollama, self.split = include_ollama, split
        self.job = win32job.CreateJobObject(None, "")
        self.cores, self.ram_mb = 0, 0
        self.llm_cpus: list[int] = []
        self.audio_cpus: list[int] = []
        self.on_pressure: Callable[[float], None] | None = None
        self._stop = threading.Event()
        self._assigned: set[int] = set()
        self._tree: list[psutil.Process] = []
        self._tree_t = 0.0
        self._dirty = True
        self.tighten(cores, ram_mb)

    # ---- layout -------------------------------------------------------
    def _layout(self, cores: int) -> None:
        cpus = physical_cpus()
        cores = max(1, min(cores, len(cpus)))
        chosen = cpus[:cores]
        if cores >= 4 and self.split:
            self.llm_cpus, self.audio_cpus = chosen[cores // 2:], chosen[: cores // 2]
        else:
            self.llm_cpus = self.audio_cpus = chosen
        self.cores = cores

    def tighten(self, cores: int, ram_mb: int) -> None:
        """Live change of the declared device (used by the demo control).

        The declared limit changes at once (the governor reacts to it). The OS cap is eased down: a Job Object limit
        below what the processes already hold makes every further allocation fail, so the cap is first set to the
        target or current usage (+headroom) and ratcheted down by the watchdog as memory is released.
        """
        self._layout(cores)
        self.ram_mb = ram_mb
        self._dirty = True
        self._set_job_limit(max(ram_mb, int(self.commit_mb() + self.HEADROOM_MB)))
        self.enforce()

    HEADROOM_MB = 256

    def _set_job_limit(self, mb: int) -> None:
        info = win32job.QueryInformationJobObject(self.job, win32job.JobObjectExtendedLimitInformation)
        info["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_JOB_MEMORY
        info["JobMemoryLimit"] = mb * MB
        win32job.SetInformationJobObject(self.job, win32job.JobObjectExtendedLimitInformation, info)

    def commit_mb(self) -> float:
        """Committed (private) memory of the whole tree: what the Job Object limit counts."""
        total = 0
        for p in self.tree():
            try:
                total += p.memory_info().private
            except psutil.Error:
                pass
        return total / MB

    def ratchet(self) -> None:
        """Lower the OS cap toward the declared limit as far as current usage allows."""
        want = max(self.ram_mb, int(self.commit_mb() + self.HEADROOM_MB))
        if want < self.job_limit_mb():
            self._set_job_limit(want)

    # ---- enforcement --------------------------------------------------
    def tree(self, refresh: bool = False) -> list[psutil.Process]:
        """App + Ollama process tree. Enumerating processes is slow on Windows, so it is cached for 2 s."""
        if refresh or time.monotonic() - self._tree_t > 2.0:
            me = psutil.Process()
            procs = [me, *me.children(recursive=True)]
            pid = ollama_pid() if self.include_ollama else None
            if pid and psutil.pid_exists(pid):
                o = psutil.Process(pid)
                procs += [o, *o.children(recursive=True)]
            self._tree, self._tree_t = list({p.pid: p for p in procs}.values()), time.monotonic()
        return self._tree

    def enforce(self) -> None:
        """Pin affinity and join the Job Object. New processes always; everything again after tighten()."""
        me = psutil.Process().pid
        for p in self.tree(refresh=True):
            if p.pid in self._assigned and not self._dirty:
                continue
            cpus = self.audio_cpus if p.pid == me or not self._is_llm(p) else self.llm_cpus
            try:
                p.cpu_affinity(cpus)
                if p.pid not in self._assigned:
                    try:
                        h = win32api.GetCurrentProcess() if p.pid == me else win32api.OpenProcess(
                            win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE, False, p.pid
                        )
                    except (OSError, TypeError, ValueError, pywintypes.error):
                        h = None
                    if h is not None:
                        try:
                            win32job.AssignProcessToJobObject(self.job, h)
                        except (AttributeError, OSError, TypeError, ValueError, pywintypes.error):
                            pass
                        finally:
                            if p.pid != me:
                                try:
                                    win32api.CloseHandle(h)
                                except (OSError, TypeError, ValueError, pywintypes.error):
                                    pass
                    self._assigned.add(p.pid)
            except (psutil.Error, OSError):
                continue
        self._dirty = False

    def _is_llm(self, p: psutil.Process) -> bool:
        try:
            return "ollama" in p.name().lower()
        except psutil.Error:
            return False

    # ---- observation --------------------------------------------------
    def rss_mb(self) -> float:
        total = 0
        for p in self.tree():
            try:
                total += p.memory_info().rss
            except psutil.Error:
                pass
        return total / MB

    def job_limit_mb(self) -> int:
        info = win32job.QueryInformationJobObject(self.job, win32job.JobObjectExtendedLimitInformation)
        return info["JobMemoryLimit"] // MB

    def affinity_report(self) -> dict[str, list[int]]:
        out = {}
        for p in self.tree():
            try:
                out[f"{p.name()}:{p.pid}"] = p.cpu_affinity()
            except psutil.Error:
                pass
        return out

    def banner(self) -> str:
        return (f"DEVICE LIMIT: {self.cores} cores (LLM cpus {self.llm_cpus}, audio cpus {self.audio_cpus}), "
                f"{self.ram_mb} MB RAM (Job Object cap {self.job_limit_mb()} MB)")

    # ---- watchdog -----------------------------------------------------
    def start_watchdog(self, period_s: float = 0.5) -> None:
        """Every 0.5 s publish memory pressure (cheap); every 2 s enforce on new processes."""
        def loop() -> None:
            n = 0
            while not self._stop.wait(period_s):
                n += 1
                if n % 4 == 0:
                    self.enforce()
                if self.on_pressure:
                    self.on_pressure(self.rss_mb() / self.ram_mb)
        threading.Thread(target=loop, name="limiter-watchdog", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
