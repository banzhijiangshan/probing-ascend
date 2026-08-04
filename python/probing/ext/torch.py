import logging
import os
from typing import Optional

import probing

hooks = {}


def _torch_profiling_spec() -> Optional[str]:
    """Resolve torch profiling spec from config, falling back to the env var.

    ``sync_env_settings()`` applies ``PROBING_TORCH_PROFILING`` asynchronously;
    the first ``optimizer.step()`` can run before that finishes. Reading the env
    here avoids creating a tracer without ``backward=on`` (and other flags).
    """
    spec = probing.config.get_str("probing.torch.profiling")
    if spec is not None and str(spec).strip():
        return str(spec).strip()
    env = os.environ.get("PROBING_TORCH_PROFILING", "").strip()
    if env:
        probing.config.set("probing.torch.profiling", env)
    return env or None


def is_true(value):
    if value in ["TRUE", "True", "true", "1", "YES", "Yes", "yes", "ON", "On", "on"]:
        return True
    return False


def optimizer_step_post_hook(optimizer, *args, **kwargs):
    """Post-step hook: create tracer once, then pick up SET/config hot changes.

    Ascend/NPU: Rust ``SET probing.torch.profiling=…`` only persists the spec
    (Tokio must not call ``configure()`` / import torch). Live rate changes must
    be applied on the training main thread here — this is the Pillar-C C0 bridge.
    """
    global hooks
    from probing.tracing.hooks import maybe_auto_attach

    maybe_auto_attach(optimizer)

    from probing.profiling.torch_probe import (
        TorchProbe,
        TorchProbeConfig,
        _sync_live_tracers,
    )

    spec = _torch_profiling_spec()
    config = TorchProbeConfig.parse(spec)
    log = logging.getLogger(__name__)

    # First step for this optimizer: install tracer (or record disabled).
    if optimizer not in hooks:
        if not config.enabled:
            log.info(
                "Torch profiling disabled (torch.profiling=%s)",
                spec or "",
            )
            hooks[optimizer] = None
            return

        from probing.profiling.torch import install_hooks
        from probing.profiling.torch.module_utils import get_toplevel_module

        tracer = TorchProbe(config=config)
        log.info(
            "Torch profiling enabled: mode=%s rate=%s shadow=%s:%s backward=%s tracepy=%s sync=%s exprs=%s",
            config.mode,
            config.rate,
            config.shadow_normal,
            config.shadow_baseline if config.shadow_enabled else 0,
            config.backward,
            config.tracepy,
            config.sync,
            config.exprs or "",
        )

        models = get_toplevel_module()
        for model in models:
            install_hooks(model, tracer=tracer, backward=config.backward)
        install_hooks(opt=optimizer, tracer=tracer, backward=config.backward)
        hooks[optimizer] = tracer
        hooks["_last_spec"] = spec or ""
        return

    # Subsequent steps: if SQL/CLI SET changed the stored spec, push onto live
    # tracers on the main thread (safe on Ascend). Enabling after a prior
    # disabled state requires a process restart — we only hot-sync rate/flags.
    last = hooks.get("_last_spec", None)
    cur = spec or ""
    if last == cur:
        return
    hooks["_last_spec"] = cur
    tracer = hooks.get(optimizer)
    if tracer is None:
        log.info(
            "Torch profiling spec changed to %r but no live tracer "
            "(was disabled at first step); restart train to enable",
            cur,
        )
        return
    if not config.enabled:
        # Soft-disable: stop sampling without tearing down hooks mid-train.
        if tracer is not None:
            tracer.rate = 0.0
            tracer.layer_rate = config.layer_rate
            tracer.config.rate = 0.0
            tracer.config.layer_rate = config.layer_rate
            tracer._planned_cycle = None
        _sync_live_tracers(TorchProbeConfig(enabled=False, rate=0.0))
        log.info("Torch profiling hot-disabled via config (%r)", cur)
        return
    if tracer is not None:
        # Prefer the live optimizer tracer (Ascend-safe); gc sweep in
        # ``_sync_live_tracers`` can miss workers on NPU.
        tracer.config.backward = config.backward
        tracer.config.trace_spans = config.trace_spans
        tracer.rate = config.rate
        tracer.layer_rate = config.layer_rate
        tracer.config.rate = config.rate
        tracer.config.layer_rate = config.layer_rate
        tracer._planned_cycle = None
    _sync_live_tracers(config)
    log.info(
        "Torch profiling hot-updated: rate=%s layer_rate=%s backward=%s (spec=%r)",
        config.rate,
        config.layer_rate,
        config.backward,
        cur,
    )


def collective_hook():
    """Autostart low-overhead collective tracing for distributed torch jobs."""
    from probing.profiling.collective import maybe_start_collective_tracing

    maybe_start_collective_tracing()


def megatron_hook():
    """Autostart Megatron role/step sync when Megatron loads before torch hooks."""
    try:
        from probing.ext.megatron import maybe_autostart

        maybe_autostart()
    except Exception:
        pass


_hook_registered = False


def init():
    global _hook_registered
    if _hook_registered:
        return
    _hook_registered = True

    from torch.optim.optimizer import register_optimizer_step_post_hook

    register_optimizer_step_post_hook(optimizer_step_post_hook)

    collective_hook()
    megatron_hook()
    try:
        from probing.crash import install

        install()
    except Exception:
        pass


def deinit():
    from probing.profiling.torch import uninstall_hooks

    uninstall_hooks()
