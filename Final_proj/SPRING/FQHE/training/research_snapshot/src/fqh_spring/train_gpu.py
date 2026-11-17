"""CUDA-only entry point for the unchanged energy or L-squared SPRING trainer.

Run ``python -m fqh_spring.train_gpu --check-device`` on a GPU compute node.
For training, supply ``--config`` and choose ``--objective energy`` or ``l2``.
The default is the latest L-squared experiment. An alpha-zero L-squared config
remains its unpenalized control; this launcher never changes alpha or the model.

The launcher verifies a synchronized float64 CUDA calculation before delegating
to an existing trainer. It does not distribute one run across multiple GPUs.
Existing trainer sources, numerical kernels and resume signatures are preserved.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys


def _load_trainer(objective):
    """Import only the selected stdlib-only trainer before configuring JAX."""
    name = "train" if objective == "energy" else "train_l2_penalty"
    return importlib.import_module(f"{__package__}.{name}")


def prepare_runtime(trainer, config):
    """Apply established CPU/thread settings, then select NVIDIA CUDA only."""
    trainer._set_runtime(config)
    # Use the concrete backend rather than the generic gpu alias, which can
    # expand to both CUDA and ROCm during JAX backend initialization.
    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def gpu_preflight():
    """Return device/probe metadata, or fail without falling back to the CPU."""
    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    try:
        import jax
        # The explicit config setting also handles callers that imported JAX
        # before this module, provided they have not initialized a backend.
        jax.config.update("jax_platforms", "cuda")
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        devices = jax.devices("gpu")
        backend = jax.default_backend()
        if not devices or backend not in {"gpu", "cuda"}:
            raise RuntimeError(f"Expected a CUDA GPU backend, found {backend!r}")
        if any(device.platform not in {"gpu", "cuda"} for device in devices):
            raise RuntimeError("A requested GPU device is not backed by CUDA")

        selected_device = devices[0]
        with jax.default_device(selected_device):
            matrix = jnp.arange(16, dtype=jnp.float64).reshape(4, 4)
            result = jax.jit(lambda value: value @ value.T)(matrix)
            result.block_until_ready()
        probe_devices = result.devices()
        if (not probe_devices or any(device.platform not in {"gpu", "cuda"}
                                     for device in probe_devices)):
            raise RuntimeError("The float64 probe ran on a non-GPU device")
        if probe_devices != {selected_device}:
            raise RuntimeError("The float64 probe did not use the selected GPU")
        if str(result.dtype) != "float64":
            raise RuntimeError(f"Expected float64, found probe dtype {result.dtype}")
        probe_value = float(result[0, 0])
        if probe_value != 14.0:
            raise RuntimeError(f"Unexpected GPU matrix-product result: {probe_value}")
        return {
            "requested_platform": "cuda",
            "backend": backend,
            "jax_version": jax.__version__,
            "devices": [str(device) for device in devices],
            "selected_device": str(selected_device),
            "device_kinds": [getattr(device, "device_kind", str(device))
                             for device in devices],
            "visible_device_count": len(devices),
            "probe_devices": [str(device) for device in probe_devices],
            "probe_dtype": str(result.dtype),
            "probe_value": probe_value,
            "probe_synchronized": True,
            "training_mode": "single_device",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_fallback_allowed": False,
        }
    except Exception as error:
        raise RuntimeError(
            "GPU preflight failed. Use an allocated NVIDIA GPU node with "
            "compatible JAX CUDA packages and driver. CPU fallback is disabled. "
            f"Details: {error}"
        ) from error


def run_on_gpu(trainer, config, *, resume=False):
    """Run one unchanged trainer on the first scheduler-visible CUDA GPU."""
    import jax
    selected_device = jax.devices("gpu")[0]
    with jax.default_device(selected_device):
        return trainer.run(config, resume=resume)


def _positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        help="Existing energy or L-squared trainer JSON config")
    parser.add_argument("--objective", choices=("energy", "l2"), default="l2",
                        help="Trainer to use; alpha remains the config's l2_penalty")
    parser.add_argument("--check-device", action="store_true",
                        help="Verify CUDA/float64 only; do not create or train a run")
    parser.add_argument("--steps", type=_positive_int,
                        help="Total optimizer-update target, not extra updates")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cpus", type=_positive_int,
                        help="Limit to N CPU IDs already allowed by the allocation")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.config is None and not args.check_device:
        parser.error("--config is required unless --check-device is selected")

    trainer = _load_trainer(args.objective)
    if args.config is not None:
        overrides = {name: getattr(args, name) for name in ("steps", "seed", "cpus")}
        overrides["backend"] = "gpu"
        if args.output is not None:
            overrides["output"] = str(args.output.resolve())
        try:
            config = trainer.load_config(args.config, overrides)
        except (OSError, ValueError, TypeError) as error:
            parser.error(str(error))
    else:
        # The established runtime helper needs only these administrative keys.
        # No model/config/output directory is constructed for a device-only check.
        config = {"backend": "gpu", "native_threads": 1, "cpus": args.cpus}

    prepare_runtime(trainer, config)
    try:
        metadata = gpu_preflight()
    except RuntimeError as error:
        print(str(error), file=sys.stderr, flush=True)
        return 1
    print(json.dumps({"gpu_preflight": metadata}, sort_keys=True), flush=True)
    if args.check_device:
        return 0

    # run() reapplies the historical runtime helper. JAX's explicit
    # jax_platforms='cuda' setting above remains authoritative after its import;
    # a subsequent environment assignment does not switch the initialized backend.
    run_on_gpu(trainer, config, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
