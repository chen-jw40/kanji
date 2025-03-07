import torch
from click import prompt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from accelerate import Accelerator
from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn
from types import SimpleNamespace
import wandb


def test_model(config, prompt):
    model_id = "CompVis/stable-diffusion-v1-4"

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(config.device)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(config.device)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler").to(config.device)

    # load checkpoint:
    checkpoint = torch.load(config.checkpoint_path, map_location=config.device)
    unet.load_state_dict(checkpoint["model_state_dict"])

    text_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True
    )
    text_input_ids = text_inputs.input_ids.to(config.device)
    with torch.no_grad():
        encoder_hidden_states = text_encoder(text_input_ids)[0]

    # Initialize random latent noise
    # Adjust latent shape as needed (for 512x512 images and a latent factor of 8, it's [1, 4, 64, 64])
    latent_shape = (1, unet.in_channels, 64, 64)
    latents = torch.randn(latent_shape, device=config.device)
    latents = latents * config.image_factor # Scaling factor used during training

    # Prepare the scheduler for inference (if necessary)
    # Create a list of timesteps for inference, usually in reverse order.
    scheduler_timesteps = noise_scheduler.timesteps if hasattr(noise_scheduler, 'timesteps') else list(range(config.num_inference_steps))[::-1]

    # Diffusion sampling loop
    for t in scheduler_timesteps:
        t_tensor = torch.tensor([t], device=config.device).repeat(latents.shape[0])
        with torch.no_grad():
            noise_pred = unet(latents, t_tensor, encoder_hidden_states).sample
        # Update latents using the scheduler step
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    # Decode the latents into an image
    with torch.no_grad():
        decoded = vae.decode(latents / config.image_factor).sample

    # Post-process the image: scale to [0, 1] then convert to [0, 255]
    image = (decoded / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).round().astype("uint8")
    image = Image.fromarray(image)
    return image


if __name__ == '__main__':
    conf = SimpleNamespace(
        image_folder = "kanji_dataset",
        json_file = "kanji_dataset.json",
        checkpoint_path = f'experience/run01/checkpoint_epoch_4.pth',
        batch_size = 4,
        num_epochs = 5,
        learning_rate = 5e-6,

        # Define the image transforms (ensure images are resized to the expected size, e.g. 512x512)
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]),
        device = "cuda" if torch.cuda.is_available() else "cpu",
        num_inference_steps = 1000,
        image_factor = 0.18215
    )

    checkpoint_path = "checkpoint_epoch_X.pth"

    # Generate a test image from a given prompt
    prompt = "a beautiful landscape with mountains"
    generated_image = test_model(conf, prompt)

    # Save or display the image
    generated_image.save("generated_test_image.png")
    generated_image.show()
    test_model(conf, prompt)



