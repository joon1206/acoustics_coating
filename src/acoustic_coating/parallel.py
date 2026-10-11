"""CPU-parallel design evaluation with portable, top-level process workers."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

import numpy as np
import pandas as pd

from .design_space import evaluate_batched, evaluate_serial


def _evaluate_chunk(payload: tuple[pd.DataFrame, np.ndarray, int]) -> pd.DataFrame:
    frame, frequency, batch_size = payload
    return evaluate_batched(frame, frequency, batch_size=batch_size)


def evaluate_parallel(
    designs: pd.DataFrame,
    frequency: np.ndarray,
    workers: int | None = None,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Evaluate chunks in separate CPU processes; safe under a main guard."""
    if len(designs) == 0:
        return designs.copy()
    n_workers = workers or 1
    # Slice explicitly: np.array_split(DataFrame) currently calls deprecated
    # DataFrame.swapaxes in pandas.
    boundaries = np.linspace(0, len(designs), n_workers + 1, dtype=int)
    chunks = [designs.iloc[boundaries[i] : boundaries[i + 1]] for i in range(n_workers)]
    chunks = [chunk for chunk in chunks if len(chunk)]
    payloads = [(chunk.copy(), np.asarray(frequency), batch_size) for chunk in chunks]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        parts = list(executor.map(_evaluate_chunk, payloads))
    return pd.concat(parts, ignore_index=True).sort_values("case_id").reset_index(drop=True)


def benchmark_modes(
    designs: pd.DataFrame,
    frequency: np.ndarray,
    workers: int,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Benchmark honest wall-clock timings without synthetic extra work."""
    timings: dict[str, float] = {}
    for name, function in (
        ("serial", lambda: evaluate_serial(designs, frequency)),
        ("batched", lambda: evaluate_batched(designs, frequency, batch_size)),
        ("parallel", lambda: evaluate_parallel(designs, frequency, workers, batch_size)),
    ):
        start = perf_counter()
        function()
        timings[name] = perf_counter() - start
    serial_time = timings["serial"]
    records = []
    for mode, elapsed in timings.items():
        processors = workers if mode == "parallel" else 1
        speedup = serial_time / elapsed
        records.append(
            {
                "n_cases": len(designs),
                "mode": mode,
                "workers": processors,
                "time_seconds": elapsed,
                "speedup": speedup,
                # Efficiency is a processor-scaling metric.  Batched mode gets
                # an algorithmic speedup but has no meaningful process efficiency.
                "parallel_efficiency": (
                    speedup / processors if mode == "parallel" else (1.0 if mode == "serial" else np.nan)
                ),
            }
        )
    return pd.DataFrame(records)

