"""R4: the simulated device is enforced by the OS (affinity + Job Object memory cap), not by politeness.

Each experiment runs in its own interpreter: a process cannot leave a Job Object, so the test runner itself
must never be placed in a tiny one.
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(code: str, timeout: int = 120) -> dict:
    r = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=ROOT, capture_output=True, text=True,
                       timeout=timeout)
    assert r.returncode == 0, r.stderr[-1500:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_affinity_and_job_cap_applied_to_the_app():
    out = run("""
        import json, psutil, win32api, win32job
        from pv.guard.limiter import Limiter, physical_cpus
        lim = Limiter(2, 1234, include_ollama=False)
        print(json.dumps(dict(
            aff=psutil.Process().cpu_affinity(), phys=physical_cpus(), cap=lim.job_limit_mb(),
            in_job=bool(win32job.IsProcessInJob(win32api.GetCurrentProcess(), lim.job)), banner=lim.banner())))
    """)
    assert len(out["aff"]) == 2 and set(out["aff"]) <= set(out["phys"])
    assert out["cap"] == 1234 and out["in_job"]
    assert "2 cores" in out["banner"] and "1234 MB" in out["banner"]


def test_memory_cap_is_enforced_on_child_processes():
    """A child inside the job can allocate under the cap and is refused above it."""
    out = run("""
        import json, subprocess, sys, psutil
        from pv.guard.limiter import Limiter
        CHILD = ("import sys; sys.stdin.readline(); x = bytearray(int(sys.argv[1]) * 2**20); "
                 "x[::4096] = b'1' * len(x[::4096]); print('ALLOCATED')")
        lim = Limiter(2, 400, include_ollama=False)
        res = {}
        for mb in (100, 900):
            p = subprocess.Popen([sys.executable, "-c", CHILD, str(mb)], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            lim._tree_t = 0
            lim.enforce()
            aff = psutil.Process(p.pid).cpu_affinity()
            o, e = p.communicate("go\\n", timeout=60)
            res[mb] = dict(allocated="ALLOCATED" in o, memory_error="MemoryError" in e, aff=len(aff))
        print(json.dumps(res))
    """)
    assert out["100"] == {"allocated": True, "memory_error": False, "aff": 2}
    assert out["900"]["allocated"] is False and out["900"]["memory_error"] is True, out


def test_tighten_changes_limits_live():
    out = run("""
        import json, psutil
        from pv.guard.limiter import Limiter
        lim = Limiter(4, 4096, include_ollama=False)
        before = (lim.cores, lim.job_limit_mb(), len(psutil.Process().cpu_affinity()))
        lim.tighten(2, 1536)
        print(json.dumps(dict(before=before, after=(lim.cores, lim.job_limit_mb(), len(psutil.Process().cpu_affinity())))))
    """)
    assert out["before"] == [4, 4096, 2] or out["before"][:2] == [4, 4096]
    assert out["after"] == [2, 1536, 2]
