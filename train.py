
import torch
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
import datetime

class CustomImageDataset(Dataset):
    def __init__(self, image_folder, json_file='kanji_dataset.json', transform=None):
        self.image_folder = image_folder
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.samples = self.match()  # list of (file_name, label) tuples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name, label = self.samples[idx]
        path = os.path.join(self.image_folder, file_name)
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def match(self):
        samples = []
        # Iterate over each image file in the folder.
        for file_name in os.listdir(self.image_folder):
            # Assume the file name (without extension) represents the kanji.
            kanji_name, _ = os.path.splitext(file_name)
            # Find the matching entry in the JSON data.
            matching_item = next((item for item in self.data if item.get("kanji") == kanji_name), None)
            if matching_item:
                # Assume the label is stored under the key 'text'
                label = matching_item.get("text", "")
                samples.append((file_name, label))
            else:
                print(f"No matching entry for image file: {file_name}")
        return samples


def train(config):
    wandb.init(project="kanji", config=config)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    # Create the dataset and dataloader
    dataset = CustomImageDataset(config.image_folder, config.json_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)


    # Load pretrained components from a Stable Diffusion model.
    # (Here we use "CompVis/stable-diffusion-v1-4" as an example.)
    model_id = "CompVis/stable-diffusion-v1-4"

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Set device
    vae.to(config.device)
    text_encoder.to(config.device)
    unet.to(config.device)

    # Freeze VAE and text encoder (we only fine-tune the UNet)
    for param in vae.parameters():
        param.requires_grad = False
    for param in text_encoder.parameters():
        param.requires_grad = False

    # Create optimizer (only for the UNet parameters)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate)

    # Use Accelerator for mixed precision and multi-GPU training if available
    accelerator = Accelerator()
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    # Begin training loop
    global_step = 0
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        sum_loss = 0
        for images, texts in progress_bar:
            # Move images to device
            images = images.to(config.device)

            # Encode images into latent space using the VAE (and scale by a factor as in training)
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() * config.image_factor

            # Sample random noise and a random timestep for each image
            noise = torch.randn_like(latents)
            bs = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=config.device).long()

            # Add noise to the latents according to the noise scheduler
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Tokenize and encode the text prompts
            text_inputs = tokenizer(texts, padding="max_length", max_length=tokenizer.model_max_length,
                                    truncation=True, return_tensors="pt")
            text_input_ids = text_inputs.input_ids.to(config.device)
            with torch.no_grad():
                encoder_hidden_states = text_encoder(text_input_ids)[0]

            # Predict the noise residual with the UNet
            model_output = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Compute mean-squared error loss between the predicted and actual noise
            criterion = nn.MSELoss()  # Instantiate the loss function
            loss = criterion(model_output, noise)  # Compute the loss by passing the tensors

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            sum_loss += loss
            wandb.log({'loss': loss})

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": unet.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": sum_loss.item(),
        }
        torch.save(checkpoint, f"{config.checkpoint_path}/checkpoint_epoch_{epoch}.pth")

    print("Training complete!")


if __name__ == "__main__":
    wandb.login(key="6970e5acc2ae4e1ce4da59616d7440953ae8fe73")

    conf = SimpleNamespace(
        image_folder = "kanji_dataset",
        json_file = "kanji_dataset.json",
        checkpoint_path = f'experience/run01/',
        batch_size = 4,
        num_epochs = 5,
        learning_rate = 5e-6,

        # Define the image transforms (ensure images are resized to the expected size, e.g. 512x512)
        device = "cuda" if torch.cuda.is_available() else "cpu",
        image_factor = 0.18215
    )
    conf_dict = vars(conf)

    os.makedirs(conf.checkpoint_path, exist_ok=True)
    # Write the dictionary to a JSON file
    with open("config.json", "w") as file:
        json.dump(conf_dict, file, indent=4)
    train(conf)
