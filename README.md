# Robot Arm Control with Action Chunking Transformers 

This project is a PyTorch implementation of the **Action Chunking with Transformers (ACT)** algorithm for robot imitation learning. It provides a complete framework for recording demonstrations, training a policy, and deploying it on a real robot arm.

This implementation was successfully used to train an SO-ARM100 robot arm to pick up a small sponge after recording just 30 ten-second demonstrations.


---

## Hardware Requirements 

* **Robot Arm:** This code was tested with an **SO-ARM100** 6-DOF arm using Feetech servos.
* **Camera:** Any standard UVC-compatible webcam. The project uses a custom 3D-printed camera holder, but any stable mount will work.

---

## Setup & Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/ozakbas/imitation-learning.git
    cd imitation-learning
    ```

2.  **Create a Virtual Environment and Install Dependencies**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Configure Hardware**
    Open the `config.py` file and update the following variables to match your setup:
    * `SERIAL_PORT`: The serial port for your servo controller (e.g., `/dev/ttyUSB0` on Linux, `/dev/cu.usbmodem...` on macOS).
    * `DEVICE`: The compute device for running the model. Set to `"cpu"` for CPU, `"cuda"` for NVIDIA GPUs, or `"mps"` for Apple Silicon. 

---

## Workflow:

Follow these steps to train your own robot policy.

### 1. Record Demonstrations 

Use your hands to guide the robot to perform a task. The script will record the servo positions and video footage simultaneously.  
Unlike a traditional leader–follower setup, here you directly move one robotic arm to collect demonstration samples. When guiding the robot, make sure your hand is not visible in the camera while moving the gripper : )

Run the recording script. It will prompt you to begin.
```bash
python3 data/record.py
```
This will save a `.csv` and `.mp4` file in the `data/recordings/` directory. Repeat this process for all your demonstrations (e.g., 30 times for the sponge task).

The first time you run this script, it will capture the robot's current position and save it as the default "rest position" in a file called data/recordings/rest_positions.json. The robot will automatically return to this position before and after every recording and replay. You can define a new rest position by simply deleting this JSON file and manually moving the robot to your desired starting pose before running the recording script again.

You can optionally replay a recording to check for any problems before running the model:
```bash
python3 data/play.py
```

Now, train the model on your recorded and preprocessed data.

```bash
python3 models/train_act.py
```
The script will:
Load all processed demonstrations.
Train the model for the number of epochs specified in config.py.
Save the best model weights to results/best_model_weights.pth.
Generate a training/validation loss plot at results/loss_plot.png.

Run Inference on the Robot
Deploy the trained policy on your robot. The script loads the best weights and performs real-time.

```bash
python3 models/predict_act.py
```
Place the robot in its starting position. The robot will attempt to perform the task based on what it learned from your demonstrations.

# Notes

- The hyperparameters used in this project are based on the tuning tips from the authors. You can review their document here: [Google Docs Link](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing)


- This project focuses on the ACT architecture to learn the fundamentals of imitation learning. It does not yet implement advanced techniques like temporal ensembling. To see what's possible with low-cost hardware, check out the full ALOHA paper. [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/pdf/2304.13705)