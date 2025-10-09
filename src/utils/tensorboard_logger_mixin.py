from torch.utils.tensorboard import SummaryWriter
import os

class TensorboardLoggerMixin:
    def init_writer(self, log_dir="runs", name="experiment"):
        full_log_dir = os.path.join(log_dir, name)
        self.writer = SummaryWriter(log_dir=full_log_dir)

    def log_scalars(self, metrics: dict, step: int):
        for key, val in metrics.items():
            self.writer.add_scalar(key, val, step)

    def log_model_weights(self, model, step: int):
        for name, param in model.named_parameters():
            self.writer.add_histogram(f"Weights/{name}", param, step)

    def close_writer(self):
        if hasattr(self, 'writer'):
            self.writer.close()