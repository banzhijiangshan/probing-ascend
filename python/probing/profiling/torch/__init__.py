"""
Torch Profiling

Spec
----
This module implements profiling hooks for PyTorch training loops.

Responsibilities:
1.  Install/Uninstall hooks on PyTorch Modules and Optimizers.
2.  Track forward passes and optimizer steps (backward hooks optional; disabled
    by default because they can alter autograd behaviour).

The training step coordinate is owned by Rust (``probing.tracing.step_snapshot``);
there is no separate Python step counter here.

Public Interfaces:
- ``install_module_hooks`` / ``uninstall_module_hooks``: model forward hooks only.
- ``install_optimizer_hooks``: optimizer step hooks (cheap; kept for every step).
- ``install_hooks``: legacy all-in-one install.
- ``module_hooks_installed``: whether module hooks are currently attached.
"""

import torch

from ..types import BaseTracer
from .module_utils import module_analysis, module_name

__all__ = [
    "install_hooks",
    "install_module_hooks",
    "install_optimizer_hooks",
    "uninstall_hooks",
    "uninstall_module_hooks",
    "module_hooks_installed",
]

MODULE_HOOK_CACHE = {}
OPTIMIZER_HOOK_CACHE = {}
# Legacy alias used by ``uninstall_hooks``.
HOOK_CACHE = MODULE_HOOK_CACHE


def install_module_hooks(
    m: torch.nn.Module,
    tracer: BaseTracer = None,
    backward: bool = False,
):
    """Attach forward hooks on ``m`` and descendants (expensive; use lazily)."""
    if tracer is None or m is None:
        return

    global MODULE_HOOK_CACHE
    if id(m) in MODULE_HOOK_CACHE:
        return
    module_analysis(m)
    h1 = m.register_forward_pre_hook(tracer.pre_forward_hook)
    h2 = m.register_forward_hook(tracer.post_forward_hook)
    MODULE_HOOK_CACHE[id(m)] = (h1, h2)
    for child in m.children():
        install_module_hooks(child, tracer=tracer, backward=backward)


def install_optimizer_hooks(
    opt: torch.optim.Optimizer,
    tracer: BaseTracer = None,
    backward: bool = False,
):
    """Attach optimizer step hooks (one call per step; always cheap)."""
    if tracer is None or opt is None:
        return

    global OPTIMIZER_HOOK_CACHE
    if opt in OPTIMIZER_HOOK_CACHE:
        return
    module_name(opt, opt.__class__.__name__)
    h1 = opt.register_step_pre_hook(tracer.pre_step_hook)
    h2 = opt.register_step_post_hook(tracer.post_step_hook)
    OPTIMIZER_HOOK_CACHE[opt] = (h1, h2)


def install_hooks(
    m: torch.nn.Module = None,
    opt: torch.optim.Optimizer = None,
    tracer: BaseTracer = None,
    backward: bool = False,
):
    """Attach profiler hooks. ``backward`` is off by default for autograd safety.

    When ``backward`` is enabled on :class:`~probing.profiling.torch_probe.TorchProbe`,
    backward timing uses output-tensor ``register_hook`` in ``post_forward_hook``
    (not module backward hooks, which break with inplace activations).
    """
    if m is not None:
        install_module_hooks(m, tracer=tracer, backward=backward)
    if opt is not None:
        install_optimizer_hooks(opt, tracer=tracer, backward=backward)


def uninstall_module_hooks() -> None:
    global MODULE_HOOK_CACHE
    for handles in MODULE_HOOK_CACHE.values():
        if isinstance(handles, tuple):
            for handle in handles:
                handle.remove()
    MODULE_HOOK_CACHE = {}


def module_hooks_installed() -> bool:
    return bool(MODULE_HOOK_CACHE)


def uninstall_hooks(m=None):
    uninstall_module_hooks()
    global OPTIMIZER_HOOK_CACHE
    for handles in OPTIMIZER_HOOK_CACHE.values():
        if isinstance(handles, tuple):
            for handle in handles:
                handle.remove()
    OPTIMIZER_HOOK_CACHE = {}
