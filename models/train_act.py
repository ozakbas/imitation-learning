import sys
import os
# This ensures the script can find the config and models directories
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..'
))
sys.path.append(PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

from config import *
from models.ACT import CVAE_encoder, PolicyEncoder, PolicyDecoder


class ACTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cvae_encoder = CVAE_encoder()
        self.policy_encoder = PolicyEncoder()
        self.policy_decoder = PolicyDecoder()
        
        self.image_proj = nn.Linear(RESNET_FEATURE_DIM, HIDDEN_DIM)
        self.qpos_proj = nn.Linear(ACTION_DIM, HIDDEN_DIM)
        self.latent_proj = nn.Linear(LATENT_DIM, HIDDEN_DIM)

    def forward(self, image_features, qpos, actions):

        # 1. Get latent distribution from CVAE encoder
        mu, log_var = self.cvae_encoder(actions, qpos)
        
        # 2. Sample z using the reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # 3. Prepare inputs and run through the policy
        memory = self.run_policy(image_features, qpos, z)
        pred_actions = self.policy_decoder(memory)
        
        return pred_actions, mu, log_var
        
    def run_policy(self, image_features, qpos, z):
        """ Helper to run the policy encoder part """
        image_embed = self.image_proj(image_features)
        qpos_embed = self.qpos_proj(qpos).unsqueeze(1)
        latent_embed = self.latent_proj(z).unsqueeze(1)
        
        # Concatenate all inputs to form the source sequence for the Transformer
        src = torch.cat([image_embed, qpos_embed, latent_embed], dim=1)
        memory = self.policy_encoder(src)
        return memory

    @torch.no_grad()
    def act(self, image_features, qpos):
        """ The inference method (deterministic, for deployment) """
        self.eval()
        # Use a zero vector for the latent variable for deterministic output
        z = torch.zeros(qpos.shape[0], LATENT_DIM, device=qpos.device)
        
        memory = self.run_policy(image_features, qpos, z)
        actions = self.policy_decoder(memory)
        self.train()
        return actions

class DummyDataset(Dataset):
    def __init__(self):
        # Simulate flattened ResNet features for 4 cameras (4 * 300 = 1200 tokens)
        self.image_features_data = torch.randn(NUM_SAMPLES, NUM_CAMERAS * IMG_TOKEN_COUNT, RESNET_FEATURE_DIM)
        self.qpos_data = torch.randn(NUM_SAMPLES, ACTION_DIM)
        self.action_data = torch.randn(NUM_SAMPLES, CHUNK_SIZE, ACTION_DIM)

    def __len__(self):
        return NUM_SAMPLES

    def __getitem__(self, idx):
        return self.image_features_data[idx], self.qpos_data[idx], self.action_data[idx]

if __name__ == "__main__":
    
    print(f"Using device: {DEVICE}")
    os.makedirs('results', exist_ok=True)

    model = ACTModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    dataset = DummyDataset()
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Data split: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test samples.")

    # --- 4. Training and Validation Loop ---
    print("Starting training...")
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        # -- Training --
        model.train()
        total_train_loss = 0
        for image_features, qpos, actions in train_loader:
            image_features, qpos, actions = image_features.to(DEVICE), qpos.to(DEVICE), actions.to(DEVICE)
            
            pred_actions, mu, log_var = model(image_features, qpos, actions)
            recon_loss = nn.functional.l1_loss(pred_actions, actions)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + BETA * kl_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # -- Validation --
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for image_features, qpos, actions in val_loader:
                image_features, qpos, actions = image_features.to(DEVICE), qpos.to(DEVICE), actions.to(DEVICE)
                
                pred_actions, mu, log_var = model(image_features, qpos, actions)
                recon_loss = nn.functional.l1_loss(pred_actions, actions)
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + BETA * kl_loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # -- Save Best Model --
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'results/best_model_weights.pth')
            print(f"   -> New best model saved with validation loss: {best_val_loss:.4f}")

    print("Training finished.")
    
    # --- 5. Final Testing ---
    print("\n--- Running Final Test ---")
    model.load_state_dict(torch.load('results/best_model_weights.pth'))
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for image_features, qpos, actions in test_loader:
            image_features, qpos, actions = image_features.to(DEVICE), qpos.to(DEVICE), actions.to(DEVICE)
            
            pred_actions, mu, log_var = model(image_features, qpos, actions)
            recon_loss = nn.functional.l1_loss(pred_actions, actions)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + BETA * kl_loss
            total_test_loss += loss.item()
            
    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Final Test Loss on the best model: {avg_test_loss:.4f}")

    # --- 6. Plot and Save Loss Curves ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/loss_plot.png')
    plt.close()
    print("Saved training and validation loss plot to results/loss_plot.png")