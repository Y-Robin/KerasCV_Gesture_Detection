# Gesture Detection with KerasCV

This project leverages KerasCV for gesture detection in video feeds, allowing for the training of custom gestures. It draws inspiration from a variety of resources including a [Keras blog post on YOLOv8](https://keras.io/examples/vision/yolov8/) and a comprehensive [YouTube tutorial](https://www.youtube.com/watch?v=yqkISICHH-U&t=20s).

## Setup Instructions

### System Requirements

Due to the requirement of the latest versions of TensorFlow and Keras, Windows users are advised to use WSL2 for a compatible development environment.

### Installation Steps

1. **Install WSL2:** Follow the official guide on Microsoft's website to [install WSL2](https://learn.microsoft.com/de-de/windows/wsl/install).

2. **Setup Visual Studio Code:** The easiest way to work with WSL2 is through Visual Studio Code. For setup instructions, refer to this [Visual Studio Code blog post](https://code.visualstudio.com/blogs/2019/09/03/wsl2).

3. **Install CUDA and cuDNN on Ubuntu:** There's no need to change your graphics card drivers for this step. Instructions for installing CUDA and cuDNN can be found in this [Medium article](https://medium.com/@gokul.a.krishnan/how-to-install-cuda-cudnn-and-tensorflow-on-ubuntu-22-04-2023-20fdfdb96907).

4. **Create a New Python Environment:** Use Visual Studio Code to [create a new Python environment](https://code.visualstudio.com/docs/python/environments) for this project.

5. **Install Keras CV:** Within the new environment, you can install Keras CV by following the instructions on the [Keras website](https://keras.io/keras_cv/).

6. **Enable Camera Access in WSL2:** To use your camera for this project, follow the tutorial to rebuild the kernel as shown in this [YouTube video](https://www.youtube.com/watch?v=t_YnACEPmrM).

## Using the Project

1. **Capture Gestures:** Start by capturing images of yourself performing different gestures using the `createImg.py` script. Press 1 for thumbs up, 2 for thumbs down, 3 for fist, and 4 for index finger. Always confirm wth y and try again with n. Make sure to capture at least 10 images for each gesture, selecting the area of interest carefully from the upper left to the lower right.

2. **Train the Model:** Select one of the Jupyter notebooks provided in the project to train a model. The notebooks vary by the model used as the backbone of the network. Feel free to modify the scripts to choose a backbone that suits your preference.

3. **Test the Model:** Use `liveTest.py` to test the trained model. Make sure to specify the model you wish to use for prediction within this script. Note that the prediction might be slightly delayed (around 90ms) but should work effectively, albeit with some lag.

## Feedback and Contributions

Your feedback and contributions are welcome! If you have suggestions for optimization or improvements, please feel free to reach out or submit a pull request.

