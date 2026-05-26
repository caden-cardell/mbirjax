"""
memory_tracker.py — JAX GPU memory tracking with call-site attribution.

Lets you tag arrays at specific points in your code and later dump a report
showing which arrays are still alive, how much device memory they hold, and
exactly where in your source they were registered.

Also exposes ``live_report()`` — a zero-setup scan of *all* live JAX arrays
using ``jax.live_arrays()`` with JAX's internal allocation tracebacks filtered
down to your own code.

Typical usage
-------------
::

    from mbirjax.memory_tracker import tracker   # global singleton

    sinogram = jax.device_put(sinogram, self.sinogram_device)
    tracker.track(sinogram, "sinogram")

    recon = jnp.zeros(shape)
    tracker.track(recon, "recon")

    # … later, to see what is consuming device memory …
    tracker.dump()

    # Or, to scan *everything* JAX has allocated without prior tracking:
    tracker.live_report()

    # Snapshot-based diff (did a loop leak memory?):
    tracker.snapshot("before_iter")
    run_one_iteration(...)
    tracker.snapshot("after_iter")
    tracker.diff("before_iter", "after_iter")
"""

from __future__ import annotations

import linecache
import sys
import threading
import weakref
from dataclasses import dataclass, field
from typing import Any, Optional

import jax


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_bytes(n: int | float) -> str:
    """Return a human-readable byte count string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


# ---------------------------------------------------------------------------
# Data class for a single tracked registration
# ---------------------------------------------------------------------------

@dataclass
class TrackedEntry:
    """One call to ``tracker.track()``."""

    label: str
    """User-supplied (or inferred) name for this array."""

    file: str
    """Source file where ``track()`` was called."""

    line: int
    """Line number of the ``track()`` call."""

    function: str
    """Enclosing function name."""

    size_bytes: int
    """Bytes on device at the time of registration."""

    shape: tuple
    dtype: str

    devices: list[str]
    """One entry per addressable shard, e.g. ``['cuda:0', 'cuda:1']``."""

    ptrs: list[Optional[int]]
    """Raw device memory pointer per shard (``None`` if unavailable)."""

    call_index: int
    """Sequential registration number (0-based)."""

    array_ref: Any = field(repr=False)
    """``weakref.ref`` to the array — may be dead by dump time."""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MemoryTracker:
    """
    Tracks JAX arrays with their call-site origin and device memory pointers.

    A module-level singleton ``tracker`` is provided for convenience.

    Parameters
    ----------
    name:
        A label printed in report headers — useful if you create multiple
        tracker instances for different subsystems.
    jax_internal_prefixes:
        Tuple of path prefixes treated as JAX-internal frames and stripped
        from ``live_report()`` tracebacks.
    """

    def __init__(
        self,
        name: str = "default",
        jax_internal_prefixes: tuple[str, ...] = (
            "/jax/",
            "/jaxlib/",
            "/site-packages/jax",
            "/site-packages/jaxlib",
            "<string>",
            "<frozen",
        ),
    ) -> None:
        self.name = name
        self._jax_prefixes = jax_internal_prefixes
        self._entries: list[TrackedEntry] = []
        self._snapshots: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._counter = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def track(
        self,
        array: Any,
        label: Optional[str] = None,
        *,
        _caller_depth: int = 1,
    ) -> Any:
        """
        Register *array* and capture the call site.

        Parameters
        ----------
        array:
            Any JAX array (``jax.Array``).  Non-JAX values are returned
            immediately without any tracking.
        label:
            Human-readable name.  If omitted the tracker tries to infer the
            LHS variable name from the call site source; falls back to
            ``"array_<N>"``.

        Returns
        -------
        The same *array* object unchanged, so the call can be inlined::

            sinogram = tracker.track(jax.device_put(sinogram, dev), "sinogram")
        """
        if not isinstance(array, jax.Array):
            return array

        # --- capture call site -------------------------------------------------
        frame = sys._getframe(_caller_depth)
        file = frame.f_code.co_filename
        line_no = frame.f_lineno
        function = frame.f_code.co_name

        if label is None:
            label = _infer_label(frame) or f"array_{self._counter}"

        # --- array metadata ----------------------------------------------------
        try:
            size_bytes = array.on_device_size_in_bytes()
        except Exception:
            size_bytes = getattr(array, "nbytes", 0)

        devices: list[str] = []
        ptrs: list[Optional[int]] = []
        for shard in array.addressable_shards:
            devices.append(str(shard.device))
            try:
                ptrs.append(shard.data.unsafe_buffer_pointer())
            except Exception:
                ptrs.append(None)

        # --- store -------------------------------------------------------------
        with self._lock:
            idx = self._counter
            self._counter += 1
            entry = TrackedEntry(
                label=label,
                file=file,
                line=line_no,
                function=function,
                size_bytes=size_bytes,
                shape=tuple(array.shape),
                dtype=str(array.dtype),
                devices=devices,
                ptrs=ptrs,
                call_index=idx,
                array_ref=weakref.ref(array),
            )
            self._entries.append(entry)

        return array

    def __call__(self, array: Any, label: Optional[str] = None) -> Any:
        """Shorthand alias for :meth:`track`.

        Usage::

            sinogram = tracker(sinogram, "sinogram")
        """
        return self.track(array, label, _caller_depth=2)

    # ------------------------------------------------------------------
    # Reporting: tracked arrays
    # ------------------------------------------------------------------

    def dump(
        self,
        *,
        only_live: bool = False,
        sort_by: str = "size",
        file=None,
    ) -> None:
        """
        Print a report of all registered arrays.

        Each entry shows:

        * Whether the array is still **live on device** (checked against the
          current set of live JAX arrays via their memory pointers).
        * The **call site** (file, line, function) where ``track()`` was called.
        * **Size, shape, dtype** of the array.
        * **Device memory pointer(s)** for cross-referencing with profiler output.

        Parameters
        ----------
        only_live:
            If ``True``, skip entries whose arrays have already been freed.
        sort_by:
            ``"size"`` (default, largest first), ``"order"`` (call order),
            or ``"device"`` (group by device then size).
        file:
            Output stream (default: ``sys.stdout``).
        """
        out = file or sys.stdout
        live_ptrs = _live_ptr_set()

        entries = list(self._entries)
        if sort_by == "size":
            entries.sort(key=lambda e: e.size_bytes, reverse=True)
        elif sort_by == "device":
            entries.sort(key=lambda e: (e.devices[0] if e.devices else "", -e.size_bytes))
        # else "order" — keep insertion order

        print(f"\n{'═'*72}", file=out)
        print(
            f"  MemoryTracker '{self.name}'  —  {len(entries)} registered arrays",
            file=out,
        )
        print(f"{'═'*72}", file=out)

        shown = 0
        total_live_bytes = 0
        for e in entries:
            ptr_in_live = any(p in live_ptrs for p in e.ptrs if p is not None)
            py_alive = e.array_ref() is not None

            if py_alive and ptr_in_live:
                # Both the Python object and device buffer are alive.
                status = "LIVE "
                total_live_bytes += e.size_bytes
            elif not py_alive:
                # Python object GC'd → array is gone regardless of ptr match.
                # (A ptr match here means the device address was reused.)
                status = "FREED"
            elif py_alive and not ptr_in_live:
                # Python object alive but JAX buffer was freed (unusual).
                status = "DEAD?"

            if only_live and not is_live:
                continue

            shown += 1
            size_str = _fmt_bytes(e.size_bytes).rjust(10)
            print(f"\n  [{e.call_index:3d}] {status}  {e.label!r}", file=out)
            print(f"        {size_str}  shape={e.shape}  dtype={e.dtype}", file=out)
            print(f"        {e.file}:{e.line}  in {e.function}()", file=out)
            for dev, ptr in zip(e.devices, e.ptrs):
                ptr_str = hex(ptr) if ptr is not None else "N/A"
                print(f"        device={dev}  ptr={ptr_str}", file=out)

        if shown == 0:
            print("\n  (no entries to show)", file=out)
        else:
            print(
                f"\n  Shown: {shown} entries  —  "
                f"live tracked memory: {_fmt_bytes(total_live_bytes)}",
                file=out,
            )

        print(f"\n{'─'*72}", file=out)
        _print_device_stats(file=out)
        print(f"{'═'*72}\n", file=out)

    # ------------------------------------------------------------------
    # Reporting: all live JAX arrays (no prior tracking needed)
    # ------------------------------------------------------------------

    def live_report(
        self,
        *,
        min_bytes: int = 0,
        show_jax_frames: bool = False,
        file=None,
    ) -> None:
        """
        Scan **all** currently live JAX arrays and report them.

        This works without any prior ``track()`` calls — useful for hunting
        down mystery memory usage after-the-fact.  Each array is shown with
        its JAX-internal allocation traceback filtered to *your* code frames.

        Parameters
        ----------
        min_bytes:
            Skip arrays smaller than this threshold (default: show all).
        show_jax_frames:
            If ``True``, include JAX-internal traceback frames (noisy but
            complete).  Default is ``False`` — only user-code frames shown.
        file:
            Output stream (default: ``sys.stdout``).
        """
        out = file or sys.stdout
        all_arrays = jax.live_arrays()
        all_arrays = sorted(all_arrays, key=_safe_size, reverse=True)

        total_bytes = sum(_safe_size(a) for a in all_arrays)
        shown = [a for a in all_arrays if _safe_size(a) >= min_bytes]

        print(f"\n{'═'*72}", file=out)
        print(
            f"  Live JAX arrays  —  {len(all_arrays)} total  "
            f"({_fmt_bytes(total_bytes)})",
            file=out,
        )
        if min_bytes > 0:
            print(f"  (showing arrays ≥ {_fmt_bytes(min_bytes)})", file=out)
        print(f"{'═'*72}", file=out)

        for i, a in enumerate(shown):
            size = _safe_size(a)
            size_str = _fmt_bytes(size).rjust(10)
            devs = [str(d) for d in a.devices()]

            # Per-shard pointer(s)
            ptr_strs: list[str] = []
            for shard in a.addressable_shards:
                try:
                    ptr_strs.append(hex(shard.data.unsafe_buffer_pointer()))
                except Exception:
                    ptr_strs.append("N/A")

            print(
                f"\n  [{i:3d}]  {size_str}  shape={a.shape}  dtype={a.dtype}",
                file=out,
            )
            for dev, ptr_s in zip(devs, ptr_strs):
                print(f"        device={dev}  ptr={ptr_s}", file=out)

            tb = getattr(a, "traceback", None)
            if tb is not None:
                tb_str = str(tb).strip()
                if show_jax_frames:
                    lines = tb_str.splitlines()
                else:
                    lines = self._filter_traceback(tb_str)

                if lines:
                    label = "full traceback" if show_jax_frames else "user frames"
                    print(f"        traceback ({label}):", file=out)
                    for ln in lines:
                        print(f"          {ln.strip()}", file=out)
                else:
                    print("        traceback: (all JAX internal — use show_jax_frames=True)", file=out)

        print(f"\n{'─'*72}", file=out)
        _print_device_stats(file=out)
        print(f"{'═'*72}\n", file=out)

    # ------------------------------------------------------------------
    # Snapshots and diff
    # ------------------------------------------------------------------

    def snapshot(self, label: str) -> None:
        """
        Capture the current device memory state under *label*.

        Call before and after a region of interest, then compare with
        :meth:`diff`.
        """
        dev_stats: dict[str, dict] = {}
        for dev in jax.local_devices():
            try:
                dev_stats[str(dev)] = dev.memory_stats()
            except Exception:
                dev_stats[str(dev)] = {}

        live = jax.live_arrays()
        # Store lightweight summary — not the actual arrays
        live_summary = [
            {
                "devices": [str(d) for d in a.devices()],
                "shape": tuple(a.shape),
                "dtype": str(a.dtype),
                "size": _safe_size(a),
                "ptrs": _array_ptrs(a),
            }
            for a in live
        ]

        with self._lock:
            self._snapshots[label] = {
                "dev_stats": dev_stats,
                "live_summary": live_summary,
            }

        total = sum(x["size"] for x in live_summary)
        print(
            f"[MemoryTracker] snapshot '{label}'  "
            f"—  {len(live)} live arrays  ({_fmt_bytes(total)})"
        )

    def diff(self, label_before: str, label_after: str, *, file=None) -> None:
        """
        Compare two snapshots and report what changed.

        Shows per-device ``bytes_in_use`` delta and new / freed arrays.
        """
        out = file or sys.stdout
        s1 = self._snapshots.get(label_before)
        s2 = self._snapshots.get(label_after)

        if s1 is None:
            print(f"[MemoryTracker] snapshot '{label_before}' not found.", file=out)
            return
        if s2 is None:
            print(f"[MemoryTracker] snapshot '{label_after}' not found.", file=out)
            return

        print(f"\n{'═'*72}", file=out)
        print(f"  Diff  '{label_before}'  →  '{label_after}'", file=out)
        print(f"{'═'*72}", file=out)

        # Device-level delta
        all_devs = sorted(set(s1["dev_stats"]) | set(s2["dev_stats"]))
        for dev in all_devs:
            st1 = s1["dev_stats"].get(dev, {})
            st2 = s2["dev_stats"].get(dev, {})
            b1 = st1.get("bytes_in_use", 0)
            b2 = st2.get("bytes_in_use", 0)
            delta = b2 - b1
            sign = "+" if delta >= 0 else ""
            peak2 = st2.get("peak_bytes_in_use", 0)
            print(
                f"\n  {dev}",
                file=out,
            )
            print(
                f"    bytes_in_use:  {_fmt_bytes(b1)}  →  {_fmt_bytes(b2)}"
                f"  (Δ {sign}{_fmt_bytes(delta)})",
                file=out,
            )
            print(f"    peak_bytes_in_use (after):  {_fmt_bytes(peak2)}", file=out)

        # Array-level delta based on pointer sets
        ptrs1 = {p for entry in s1["live_summary"] for p in entry["ptrs"]}
        ptrs2 = {p for entry in s2["live_summary"] for p in entry["ptrs"]}

        new_ptrs = ptrs2 - ptrs1
        freed_ptrs = ptrs1 - ptrs2

        new_arrays = [e for e in s2["live_summary"] if any(p in new_ptrs for p in e["ptrs"])]
        freed_arrays = [e for e in s1["live_summary"] if any(p in freed_ptrs for p in e["ptrs"])]

        new_bytes = sum(e["size"] for e in new_arrays)
        freed_bytes = sum(e["size"] for e in freed_arrays)
        net = new_bytes - freed_bytes

        print(f"\n  Arrays allocated:  {len(new_arrays)}  ({_fmt_bytes(new_bytes)})", file=out)
        for e in sorted(new_arrays, key=lambda x: x["size"], reverse=True)[:20]:
            ptrs_str = "  ".join(hex(p) for p in e["ptrs"])
            print(
                f"    + {_fmt_bytes(e['size']).rjust(10)}  shape={e['shape']}"
                f"  dtype={e['dtype']}  dev={e['devices']}  ptr={ptrs_str}",
                file=out,
            )

        print(f"\n  Arrays freed:      {len(freed_arrays)}  ({_fmt_bytes(freed_bytes)})", file=out)
        for e in sorted(freed_arrays, key=lambda x: x["size"], reverse=True)[:20]:
            ptrs_str = "  ".join(hex(p) for p in e["ptrs"])
            print(
                f"    - {_fmt_bytes(e['size']).rjust(10)}  shape={e['shape']}"
                f"  dtype={e['dtype']}  dev={e['devices']}  ptr={ptrs_str}",
                file=out,
            )

        sign = "+" if net >= 0 else ""
        print(f"\n  Net change:  {sign}{_fmt_bytes(net)}", file=out)
        print(f"{'═'*72}\n", file=out)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def find_ptr(self, ptr: int, *, file=None) -> None:
        """
        Search tracked entries for an array whose device pointer matches *ptr*.

        Useful when a profiler or CUDA tool reports a specific address and you
        want to know which variable it belongs to.

        Parameters
        ----------
        ptr:
            Integer device-memory address (can be obtained from a profiler,
            from ``array.unsafe_buffer_pointer()``, etc.).
        """
        out = file or sys.stdout
        hits = [e for e in self._entries if ptr in e.ptrs]
        if not hits:
            print(f"[MemoryTracker] ptr {hex(ptr)} not found in tracked entries.", file=out)
            return
        print(f"\n[MemoryTracker] ptr {hex(ptr)} matches:", file=out)
        for e in hits:
            print(f"  [{e.call_index}] {e.label!r}  {e.file}:{e.line}  {_fmt_bytes(e.size_bytes)}", file=out)

    def clear(self) -> None:
        """Remove all tracked entries (but keep snapshots)."""
        with self._lock:
            self._entries.clear()
            self._counter = 0

    def _filter_traceback(self, tb_str: str) -> list[str]:
        """Return only non-JAX-internal lines from a traceback string."""
        lines = tb_str.splitlines()
        return [
            ln
            for ln in lines
            if ln.strip()
            and not any(pat in ln for pat in self._jax_prefixes)
        ]

    def __repr__(self) -> str:
        return f"MemoryTracker(name={self.name!r}, entries={len(self._entries)})"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _safe_size(array: jax.Array) -> int:
    try:
        return array.on_device_size_in_bytes()
    except Exception:
        try:
            return array.nbytes
        except Exception:
            return 0


def _array_ptrs(array: jax.Array) -> list[int]:
    ptrs: list[int] = []
    for shard in array.addressable_shards:
        try:
            ptrs.append(shard.data.unsafe_buffer_pointer())
        except Exception:
            pass
    return ptrs


def _live_ptr_set() -> set[int]:
    """Return the set of device pointers for all currently live JAX arrays."""
    ptrs: set[int] = set()
    try:
        for a in jax.live_arrays():
            ptrs.update(_array_ptrs(a))
    except Exception:
        pass
    return ptrs


def _print_device_stats(file=None) -> None:
    out = file or sys.stdout
    for dev in jax.local_devices():
        try:
            stats = dev.memory_stats()
            used = stats.get("bytes_in_use", 0)
            limit = stats.get("bytes_limit", 0)
            peak = stats.get("peak_bytes_in_use", 0)
            pct = 100.0 * used / limit if limit else 0.0
            print(
                f"  {dev}:  in_use={_fmt_bytes(used)} ({pct:.1f}%)"
                f"  peak={_fmt_bytes(peak)}  limit={_fmt_bytes(limit)}",
                file=out,
            )
        except Exception:
            print(f"  {dev}:  memory_stats() unavailable", file=out)


def _infer_label(frame) -> Optional[str]:
    """
    Best-effort: read the source line at the call site and extract either
    the first argument name (``tracker.track(sinogram)``) or the LHS of an
    assignment (``x = tracker.track(arr)``).
    """
    try:
        source = linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip()
        # Case 1: assignment  →  grab LHS
        if "=" in source and not source.startswith("if") and not source.startswith("while"):
            lhs = source.split("=")[0].strip()
            if lhs.isidentifier():
                return lhs
        # Case 2: tracker.track(name, ...) or tracker(name, ...)  →  grab first arg
        import re
        m = re.search(r"\.track\(\s*(\w+)", source) or re.search(r"tracker\(\s*(\w+)", source)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

#: Global tracker instance.  Import and use directly::
#:
#:     from mbirjax.memory_tracker import tracker
#:     tracker.track(my_array, "my_array")
tracker = MemoryTracker(name="global")
