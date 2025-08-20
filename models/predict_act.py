import sys
import os
import torch

# This ensures the script can find the config and models directories
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..'
))
sys.path.append(PROJECT_ROOT)

from config import *
from models.train_act import ACTModel

def main():
    """
    Main function to load the model, run a prediction, and print the output.
    """
    # --- 1. Setup paths and device ---
    print(f"Using device: {DEVICE}")
    weights_path = 'results/best_model_weights.pth'

    # Check if the trained weights exist
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at {weights_path}")
        print("Please run train_act.py to train and save the model first.")
        return

    # --- 2. Load the trained model ---
    print("Loading model and weights...")
    # Instantiate the model with the same architecture as during training
    model = ACTModel().to(DEVICE)
    # Load the saved state dictionary
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    # Set the model to evaluation mode
    model.eval()
    print("Model loaded successfully.")

    # --- 3. Prepare a dummy input observation ---
    # This simulates a single observation from the robot's sensors.
    # The batch size is 1.
    dummy_image_features = torch.randn(1, NUM_CAMERAS * IMG_TOKEN_COUNT, RESNET_FEATURE_DIM).to(DEVICE)
    dummy_qpos = torch.randn(1, ACTION_DIM).to(DEVICE)

    print("\nRunning prediction with a sample input...")
    
    # --- 4. Run Inference ---
    # The .act() method is used for inference (it's wrapped in torch.no_grad())
    predicted_actions = model.act(dummy_image_features, dummy_qpos)

    import pdb; pdb.set_trace()

    # --- 5. Display the result ---
    print("Prediction successful!")
    print(f"Output action chunk shape: {list(predicted_actions.shape)}")
    print("(This means: [batch_size, chunk_size, action_dimension])")
    
    # Print the first predicted action vector in the chunk
    first_action_in_chunk = predicted_actions[0, 0, :]
    print(f"\nFirst predicted action in the chunk:\n{first_action_in_chunk.cpu().numpy()}")

if __name__ == "__main__":
    main()