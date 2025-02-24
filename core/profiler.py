import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.profiler import tensorboard_trace_handler

torch.set_printoptions(linewidth=1000)


def check_profile(model: nn.Module, *inp: torch.Tensor, **kwargs):
    device = "cuda:0"
    outdir = os.path.join(os.path.dirname(__file__), "log")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    else:
        shutil.rmtree(outdir)
        os.makedirs(outdir)

    model = model.to(device=device)
    model.eval()

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=tensorboard_trace_handler(outdir),
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
        with_stack=True,
        record_shapes=True,
    ) as prof:
        for _ in range(1 + 1 + 3):
            prof.step()
            with torch.no_grad():
                _ = model(*[x.to(device=device) for x in inp])
                # prof.step()

    # print(prof.key_averages().table(sort_by="cuda_memory_usage"))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    # prof.export_chrome_trace("trace.json")
