import lightning
import torch


class CUDAProfiler(lightning.Callback):
    def __init__(self, warmup: int = 2, active: int | None = None):
        self.warmup = warmup
        self.active = active
        self.enabled = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not self.enabled and trainer.global_step > self.warmup:
            torch.cuda.cudart().cudaProfilerStart()
            self.enabled = True
            print(f"enabling cuda profiler {trainer.global_step=}")

        if self.active is not None:
            if self.global_step > self.warmup + self.active:
                torch.cuda.cudart().cudaProfilerStop()
                print(f"disabling cuda profiler {trainer.global_step=}")
