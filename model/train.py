from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMScheduler
import os
from dataclasses import dataclass
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F


@dataclass
class DiffusionConfig:
    batch_size: int = 128
    num_epochs: int = 100
    save_image_epochs: int = 1
    save_model_epochs: int = 1
    learning_rate: float = 1e-4
    num_warmup_steps: int = 500
    push_to_hub: bool = False
    output_dir: str = "output/"
    num_train_timesteps: int = 1000


config = DiffusionConfig()


# still in progress, doesn't call diffusionstep correctly
def train_diffusion(model, config: DiffusionConfig, dataloader):

    noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=1,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean = batch["molecule"]

            noise = torch.randn(clean.shape, device=clean.device)
            bs = clean.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=clean.device,
                dtype=torch.int64,
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noised = noise_scheduler.add_noise(clean, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noised, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

                # After each epoch you optionally sample some demo images with evaluate() and save the model
                if accelerator.is_main_process:
                    if (
                        (epoch + 1) % config.save_model_epochs == 0
                        or epoch == config.num_epochs - 1
                    ):
                        model.save(os.path.join(config.output_dir, "model.pt"))
