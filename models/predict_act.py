import sys
import os
import torch
import time
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from config import *
from models.train_act import ACTModel
from data.servo_handler import ServoHandler

def _normalize(values):
    """ Normalizes servo positions from [0, 4095] to [-1, 1]. """
    min_val = 0
    max_val = 4095
    return 2 * (values - min_val) / (max_val - min_val) - 1

def _denormalize(values):
    """ De-normalizes actions from [-1, 1] back to [0, 4095]. """
    min_val = 0
    max_val = 4095
    return ((values + 1) * (max_val - min_val) / 2 + min_val).round().to(torch.int)

def main():
    """
    Main function to load the model and run the inference loop on the robot.
    """
    print(f"Using device: {DEVICE}")
    weights_path = 'results/best_model_weights.pth'
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at {weights_path}")
        return

    print("Loading ACT model and weights...")
    model = ACTModel().to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    print("ACT Model loaded successfully.")

    print("Loading ResNet feature extractor...")
    resnet = models.resnet18(pretrained=True)
    feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1]).to(DEVICE)
    feature_extractor.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    servo_handler = ServoHandler()
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    try:
        print("Connecting to servos...")
        servo_handler.connect()
        for servo_id in SERVO_IDS:
            servo_handler.set_torque(servo_id, True)
        
        input("\nPress Enter to start the inference loop...")
        print("Starting inference loop. Press Ctrl+C to stop.")
        
        while True:
            loop_start_time = time.time()

            # Get current joint positions (qpos)
            current_positions = [servo_handler.read_position(sid) for sid in SERVO_IDS]
            if any(p is None or p == -1 for p in current_positions):
                print("Warning: Failed to read one or more servo positions. Skipping step.")
                time.sleep(SAMPLING_INTERVAL)
                continue
            
            qpos_raw = torch.tensor(current_positions, dtype=torch.float32)
            
            # Get current camera image
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to capture frame. Skipping step.")
                time.sleep(SAMPLING_INTERVAL)
                continue

            qpos_normalized = _normalize(qpos_raw).to(DEVICE)
            
            # Process image to get features
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            image_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                image_features = torch.flatten(feature_extractor(image_tensor), 1)

            # Add batch dimension (1) to inputs
            qpos_input = qpos_normalized.unsqueeze(0)
            
            predicted_actions_normalized = model.act(image_features, qpos_input)


            # Get the first action from the predicted chunk
            first_action_normalized = predicted_actions_normalized[0, 0, :]
            
            # De-normalize the action to servo values
            action_to_execute = _denormalize(first_action_normalized)
            
            print(f"Current Pos: {qpos_raw.numpy()} | Predicted Action: {action_to_execute.cpu().numpy()}")
            
            # Send the command to the servos
            for sid, pos in zip(SERVO_IDS, action_to_execute):
                servo_handler.move_servo(sid, pos.item())

            # Maintain a consistent loop frequency matching the recording interval
            elapsed_time = time.time() - loop_start_time
            sleep_time = SAMPLING_INTERVAL - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nInference loop stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("Cleaning up: disabling torque and disconnecting...")
        if servo_handler.is_port_open:
            for servo_id in SERVO_IDS:
                servo_handler.set_torque(servo_id, False)
            servo_handler.disconnect()
        cap.release()
        cv2.destroyAllWindows()
        print("Cleanup complete.")

if __name__ == "__main__":
    main()