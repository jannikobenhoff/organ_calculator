import torch
import os

class SaveBestModel:
    def __init__(self, output_dir="checkpoints", best_metric=float('inf'), lower_better=True):
        self.best_metric = best_metric
        self.output_dir = output_dir
        self.lower_better = lower_better
        os.makedirs(output_dir, exist_ok=True)

    def __call__(self, current_metric, epoch, generator, discriminator, opt_g, opt_d):
        if (current_metric < self.best_metric and self.lower_better) or \
           (current_metric > self.best_metric and not self.lower_better):
            print(f"Saving better model in epoch {epoch}", flush=True)

            self.best_metric = current_metric
            self.save_checkpoint(
                epoch, generator, discriminator, opt_g, opt_d,
                f"best_model.pth", "Best"
            )

    def save_checkpoint(self, epoch, generator, discriminator, opt_g, opt_d, filename, prefix=""):
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_g_state_dict': opt_g.state_dict(),
            'optimizer_d_state_dict': opt_d.state_dict(),
            'metric': self.best_metric
        }, os.path.join(self.output_dir, filename))
        print(f"{prefix} model saved at epoch {epoch+1}")
