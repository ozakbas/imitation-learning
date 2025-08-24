import sys
import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import cv2
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..'
))
sys.path.append(PROJECT_ROOT)

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

        # --- STABILITY FIX: Clamp log_var to prevent exp() from exploding ---
        log_var = torch.clamp(log_var, max=10.0)
        
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
        image_embed = self.image_proj(image_features).unsqueeze(1)
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

class ACTDataset(Dataset):
    def __init__(self, recordings_folder, chunk_size=CHUNK_SIZE):
        """
        Args:
            recordings_folder (str): Path to the folder containing recordings.
            chunk_size (int): The size of the action sequence chunk.
        """
        self.chunk_size = chunk_size
        self.device = DEVICE

        # Find all csv files and create a list of trajectory paths
        csv_files = sorted(glob.glob(os.path.join(recordings_folder, '*.csv')))
        self.trajectories = []
        for csv_path in csv_files:
            base_name = os.path.basename(csv_path).split('.')[0]
            video_path = os.path.join(recordings_folder, base_name + '.mp4')
            if os.path.exists(video_path):
                self.trajectories.append({'csv': csv_path, 'video': video_path})
        
        
        print(f"Found {len(self.trajectories)} trajectory pairs.")

        # --- Pre-trained ResNet for feature extraction ---
        resnet = models.resnet18(pretrained=True)
        # Remove the final classification layer to get features
        self.feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()  # Set to evaluation mode

        # --- Image transformations to match ResNet's expected input ---
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # --- Pre-calculate indices for __getitem__ ---
        self.indices = []
        for traj_idx, traj in enumerate(self.trajectories):
            # lines/frames per file
            num_frames = int(RECORDING_DURATION / SAMPLING_INTERVAL)
            # We can create a sample starting from any frame that has enough future frames
            # to form a complete action chunk.
            for start_frame in range(num_frames - self.chunk_size):
                self.indices.append((traj_idx, start_frame))
    

    def _normalize(self, values):
        """ Normalizes servo positions from [0, 4095] to [-1, 1]. """
        min_val = 0
        max_val = 4095
        return 2 * (values - min_val) / (max_val - min_val) - 1

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, start_frame = self.indices[idx]
        
        csv_path = self.trajectories[traj_idx]['csv']
        positions_df = pd.read_csv(csv_path)
        
        qpos_row = positions_df.iloc[start_frame]
        qpos = torch.tensor(qpos_row.values[1:], dtype=torch.float32) # Skip 'Time' column
        
        # actions are the next `chunk_size` states
        actions_rows = positions_df.iloc[start_frame + 1 : start_frame + 1 + self.chunk_size]
        actions = torch.tensor(actions_rows.values[:, 1:], dtype=torch.float32)

        qpos = self._normalize(qpos)
        actions = self._normalize(actions)

        video_path = self.trajectories[traj_idx]['video']
        cap = cv2.VideoCapture(video_path)
        
        image_features = None
        try:
            # Set the video capture to the specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret, frame = cap.read()
            if ret:
                # Convert frame to PIL Image (OpenCV uses BGR, PIL/torchvision use RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Apply transformations and add batch dimension
                image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    # Extract features and flatten
                    features = self.feature_extractor(image_tensor)
                    # Flatten the features to (1, RESNET_FEATURE_DIM)
                    image_features = torch.flatten(features, 1)
                                        
            else:
                # If frame read fails, create a zero tensor as a fallback
                print(f"Warning: Failed to read frame {start_frame} from {video_path}")
                image_features = torch.zeros(1, RESNET_FEATURE_DIM, device=self.device)

        finally:
            cap.release()

        # The model expects image features without the batch dimension within the dataset
        return image_features.squeeze(0), qpos, actions

if __name__ == "__main__":
    
    print(f"Using device: {DEVICE}")
    os.makedirs('results', exist_ok=True)

    model = ACTModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    dataset = ACTDataset(RECORDINGS_FOLDER)
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Data split: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test samples.")

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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