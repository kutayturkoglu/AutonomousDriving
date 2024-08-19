# Autonomous Driving Model Project

This project implements an autonomous driving system using deep learning techniques. The project is structured into two main components: a model training pipeline and a server that receives telemetry data and controls a simulated vehicle.

## Project Structure

- **`main.ipynb`**: A Jupyter notebook for training and validating the deep learning model.
- **`drive.py`**: The server-side script that handles telemetry data from a driving simulator, processes images, and sends control commands to the simulator.
- **`test_model_first.pth`**: A pre-trained model file.
- **`test_model_second.pth`**: A pre-trained model file that is used by default in `drive.py`.

## Requirements

- Python 3.x
- Flask
- SocketIO
- Eventlet
- PyTorch
- torchvision
- OpenCV
- PIL
- tqdm

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository.git
   cd your-repository
   ```
2. Install the required Python packages
    ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have the required pre-trained models (test_model_first.pth, test_model_second.pth) in the project directory.

## Usage 
1. Training Model
The training pipeline is defined in main.ipynb. You can use this notebook to train and validate a model on a custom dataset. The model architecture used is a modified ResNet18 or the Nvidia architecture.

Steps:
* Prepare your dataset with images and corresponding steering angles.
* Load the dataset in the notebook.
* Train the model using the provided pipeline.
* Save the trained model.

2. Running the Autonomous Driving Server
The drive.py script sets up a server that interfaces with a driving simulator (e.g., Udacity's self-driving car simulator). The server processes incoming images, predicts steering angles, and sends control commands to the simulator.

Steps:

* Ensure you have the simulator running and connected to the server.
* Run the server:
```bash
   python drive.py
   ```

* The server will listen on port 4567 by default and control the vehicle based on the loaded model (test_model_second.pth).

3. Testing Different Models

You can switch between different pre-trained models by modifying the drive.py script to load a different .pth file.
4. Modifying the Model

The current model in use is a modified ResNet18, which has been adapted for regression (predicting steering angles). You can experiment with different architectures (e.g., Nvidia's model) by changing the model definition in the notebook or drive.py.

## Notes
* The speed_limit variable in drive.py controls the maximum speed of the vehicle. Adjust this as needed.
* The current image preprocessing pipeline resizes images to 224x224 for ResNet18. Ensure your model's input size matches this if you modify the architecture.

## Troubleshooting

    Model Loading Issues: Ensure the .pth files are compatible with the model architecture defined in the code.
    Telemetry Errors: Check that the simulator is sending valid data and that the drive.py script is properly decoding and processing it.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
