from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMScheduler

import os
from dataclasses import dataclass
from tqdm.auto import tqdm

import torch
from torch.nn import functional as F

from model.utils import kabsch_torch_batched


# move accelerator out
def train_diffusion(
    model,
    config,
    train_dataloader,
    noise_scheduler=None,
    optimizer=None,
    lr_scheduler=None,
):

    if noise_scheduler is None:
        noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps)

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=0.01
        )

    if lr_scheduler is None:
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.num_warmup_steps,
            num_training_steps=(len(train_dataloader) * config.num_epochs),
        )

    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=1,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs/"),
    )

    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("pernodediffusion")

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

            counts = batch.batch.unique(return_counts=True)[1]
            duped = batch.y.repeat_interleave(counts, dim=0)

            clean = torch.cat([batch.pos, batch.x, duped], dim=-1)

            noise = torch.randn(
                (len(counts), clean.shape[-1]), device=clean.device
            ).repeat_interleave(counts, dim=0)
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (1,),
                device=clean.device,
                dtype=torch.int64,
            )

            timesteps = timesteps.expand(clean.shape[0])

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noised = noise_scheduler.add_noise(clean, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual

                propogation = config.num_train_timesteps - timesteps[0]
                noise_pred = model(
                    noised,
                    batch.z.to(noised.device),
                    batch.edge_index.to(noised.device),
                    gnn_time_step=config.num_train_timesteps - timesteps[0],
                    diffusion_time=timesteps,
                )

                loss = 0

                partial_loss = 0
                # transform between learned and real
                for y, x in zip(
                    batch.pos.split(tuple(counts)), noise_pred.split(tuple(counts))
                ):

                    with torch.no_grad():
                        R, t = kabsch_torch_batched(y[None, :, :], x[None, :, :3])
                        # aligning noise... not sure if this is correct. could be a big issue.
                        # maybe reframe to predict final image, not noise directly, then subtract for noise estimate for diffusion loss?

                    warped = y - y.mean(dim=0, keepdims=True)
                    warped = warped @ R.squeeze().T

                    x = x - x.mean(dim=0, keepdims=True)

                    partial_loss = partial_loss + F.mse_loss(
                        warped[:, :3], x[:, :3]
                    ) * (1 / len(counts))

                loss = (
                    F.mse_loss(noise_pred[:, 3:], noise[:, 3:]) * 30 / 33
                    + partial_loss * 3 / 33
                )
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
                        torch.save(
                            model.state_dict(),
                            os.path.join(config.output_dir, "model.pt"),
                        )
