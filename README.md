# Predict Object Angle Value Based on Dataset of Ultrasound Images with ResNet-18
This project utilizes ResNet-18 for a regression task that predicts the angles of carotid images in ultrasound recordings. The dataset is formatted as a CSV file containing pairs of ultrasound images and their corresponding angle values recorded with an IMT sensor.

## Linear Regression
Linear regression is a statistical method used to define the linear relationship between a dependent variable and one or more independent variables. In this project, we successfully recorded a sequence of ultrasound video, storing them as small frames. Each frame is accompanied by the angle produced by the IMT sensor, corresponding to the current position of the probe. Each video is stored as a sequence of frames in a single folder. The video is recorded when the carotid artery is displayed horizontally in full view. Subsequent frames capture the continuous rotation of the probe while maintaining visibility of the carotid artery. To create the dataset, we assume that the first frame of each folder has an angle of 0 degrees since the carotid artery lies along the horizontal axis. The angles of the other frames are then adjusted accordingly. Below are some carotid images taken from different angles:
![cca1](first_frame.png)
![cca2](cca2.png)
![cca3](cca3.png)
![cca4](cca4.png)

## ResNet-18
This project employs ResNet-18 as the pre-trained model. While ResNet-18 is primarily used for image classification, it can also be adapted for regression tasks by modifying the final layer to output a single continuous value instead of class labels. The reason for choosing ResNet-18 for this regression task is its strong performance in both computational efficiency and accuracy.

## Prepare Dataset
The dataset should be compiled into a CSV file consisting of two columns: `filename` and `angle`. Each file path is stored under the `filename` column, while the corresponding angle for each image is recorded in the `angle` column. This structure allows us to access all images easily through their paths once the CSV file is specified.

## Project Setup
To run the training and testing code, ensure you have downloaded all of the necessary libraries through the commands below:
- `pip install pandas`
- `pip install torch`
- `pip install pillow`
- `pip install torchvision
- `pip install matplotlib`
- `pip install numpy`

## Running the code
Execute `linear_regression.py` to initiate the training process. For each epoch, the program will compare the accuracy of the current model with that of the previous one. Once the training is complete or halted, the best model will be saved as a `.pth` file.

## Explanation of model parameter
Training a deep learning model involves selecting the right parameters to ensure optimal performance. Below is an explanation of the parameters used to train our model:
- Criterion: Huber Loss (less sensitive to outliers compared to squared error loss)
- Optimizer: Adam (provides adaptive learning rates and efficient optimization
- Scheduler: ReduceLROnPlateau (reduces the learning rate when the model stops improving)
- Early Stopping: A regularization technique that prevents overfitting by halting training before the model memorizes irrelevant patterns and noise. Early stopping also helps identify optimal epochs and avoids unnecessary training.


