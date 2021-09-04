# Face-Mask-and-Social-Distance-Detection
#### In this project, we bulid a Deep Learning Model based on MobileNetV2 and YOLOv3 to detect live Face Masks and Social Distancing respectively. MobileNetV2 and YOLO are used here in order to minimize the run time and maximize the Frames Per Second(FPS) when live inferencing, while not significantly sacrificing the accuracy. 

<br>

## Setting up the environment and installing the dependencies
* ***Open the Terminal**
* ***Change directory to where you want to downloaded this code***
* ***Run `git clone https://github.com/sagarmandiya/Face-Mask-and-Social-Distance-Detection.git`**
* **Creating a new conda environment after installing anaconda**  `  conda create -n ObjectDetection ` ***to create a virtual environment named ObjectDetection.***
* **Activate the ObjectDetection environment by**   `  conda activate ObjectDetection  python=>3.7` 
* **Run**   `  pip install -r requirements.txt  ` 
***to install the python dependencies related to this project like TensorFlow, opencv, numpy, scipy etc.***

<br>

## [Optional] Training and Saving the MobileNetV2 model for Face Mask Detecion for Future Inference
* **This Step can be skipped, as I have already provided my trained model as mask_detector.model which can be directly used for inference purpose**
* **To Train and Save the model, simply run `python train_mask_detector.py` .** 
* **Running the above line in terminal will Train the model on the Dataset already provided in the Dataset directory. The dataset can be changed as required just keep the labels same as the directory name in the dataset.**

<br>

## To run the inference on local machine:
* ***Open the terminal**
* **To Run the detection on a locally stored video named pedestrians.mp4 run** `python live_inference.py --input pedestrians.mp4 --output output.avi --display 1`
* **Whereas to run the detection live using the Webcam simply run** ` python live_inference.py `

#### **After you run the above line of command, a window will pop up depending upon your choice of live or locally stored video, and after execution of the file an `output.avi` file will be made in your directory if you chose the locally stored video option, otherwise webcam will open for live inference.**
<br>

## Inference on GPU:
**The same above code can be run with the GPU, but will require OpenCV dnn module to be built on top of CUDA, which can be done as instructed** [here](https://learnopencv.com/opencv-dnn-with-gpu-support/)**. Additionally the code can also be run on google colab to use the GPU for faster training of the MobileNetV2 model for Face Mask Detection.**
<br>

## Contact Me:
* **Website: [Sagar Mandiya](https://www.sagarmandiya.me)**
* **Blog: [Sagar Mandiya Blog](https://www.sagarmandiya.me/blog)**
* **LinkedIn: [Sagar M](https://www.linkedin.com/in/sagar-m-647a2b183)**
* **Instagram: [Sagar Mandiya](https://www.instagram.com/sagar_mandiya/)**
