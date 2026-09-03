#!/usr/bin/env python3
"""Fast, FairTune-independent sanity check for the ROCm PyTorch environment.

Run this BEFORE touching search_mask.py/finetune_with_mask.py at all — it's
seconds, not hours, and catches two classes of problem cheaply:

1. ROCm not actually seeing the GPU (silent CPU fallback).
2. The well-known PyTorch DataLoader `num_workers > 0` deadlock on ROCm:
   this builds a tiny synthetic DataLoader and iterates it under a timeout.
   If THIS hangs, don't wait for the real training run to also hang three
   hours in - lower --workers now (0 is always safe, try 1-2 first) and
   rerun this check.
"""
import argparse
import sys
import time


def check_device():
    import torch

    print(f"torch.__version__ = {torch.__version__}")
    available = torch.cuda.is_available()
    print(f"torch.cuda.is_available() = {available}")
    if not available:
        print("FAIL: no GPU visible to torch. Check `rocm-smi`, and that librocdxg / the ROCm wheels installed cleanly.")
        sys.exit(1)

    name = torch.cuda.get_device_name(0)
    print(f"torch.cuda.get_device_name(0) = {name}")

    device = torch.device("cuda")
    t0 = time.time()
    a = torch.randn(4096, 4096, device=device, requires_grad=True)
    b = torch.randn(4096, 4096, device=device)
    c = (a @ b).sum()
    c.backward()
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"On-device matmul + backward pass OK ({dt:.3f}s) — this ran on the GPU, not a silent CPU fallback.")


def check_dataloader(workers, timeout_s):
    import multiprocessing as mp

    def _worker(workers, q):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        ds = TensorDataset(torch.randn(200, 3, 224, 224), torch.randint(0, 2, (200,)))
        dl = DataLoader(ds, batch_size=16, num_workers=workers, pin_memory=True)
        n = 0
        for _ in dl:
            n += 1
        q.put(n)

    print(f"Testing DataLoader(num_workers={workers}) on synthetic data (timeout {timeout_s}s)...")
    q = mp.Queue()
    p = mp.Process(target=_worker, args=(workers, q))
    p.start()
    p.join(timeout=timeout_s)

    if p.is_alive():
        p.terminate()
        p.join()
        print(
            f"FAIL: DataLoader(num_workers={workers}) did not finish within {timeout_s}s — "
            "no traceback, no crash, just a hang. This is the documented ROCm "
            "DataLoader deadlock signature. Rerun the real pipeline with --workers 0 "
            "(or try 1-2 first) rather than the current default."
        )
        sys.exit(1)

    print(f"OK: DataLoader(num_workers={workers}) completed {q.get()} batches with no hang.")
    print("(A crash WITH a traceback here is a different, real bug — don't just lower --workers for that.)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4, help="num_workers value to smoke-test (match what run_pipeline.sh will actually use)")
    ap.add_argument("--timeout", type=int, default=60, help="Seconds to wait before declaring a DataLoader hang")
    args = ap.parse_args()

    check_device()
    print()
    check_dataloader(args.workers, args.timeout)
    print()
    print("Environment looks good.")


if __name__ == "__main__":
    main()
