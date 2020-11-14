![boat](boat.png)


# This is a repository to hold the data for the AI Tracks at sea challenge
## This is a short walkthrough of how to train a neural network to identify the boat

This walkthrough was tested on my machine which is running on Windows 10 and will be using Tensorflow 1.14.


## Steps
### Step 1. Installing Dependencies (Anaconda, CUDA and cuDNN)
Firstly we will be using Anaconda to run our environment since it will allow things to run more smoothly. You can download [Anaconda here](https://www.anaconda.com/products/individual). Anaconda will recommend to use Python 3.8 as default for your machine, you can either choose to allow this or unselect it as we will be using a different version of python later on. Make sure that conda is in your path as it will be necessary for us to be able to use.
Secondly, we will be using CUDA v10.0 as it is what I tested and seemed to work best with these settings. You can download [CUDA here](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork).
Lastly, the cuDNN that we will be using to accompany the CUDA version will be cuDNN v7.4 for CUDA v10 which can be [downloaded here](https://developer.nvidia.com/rdp/cudnn-archive). ** Note that you will have to register for a developer account in order to download this.

### Step 2. Downloading files and preparing environment

#### Step 2a. Setup up Tensorflow directories
Create a folder on your machine named 'tensorflow' and place it into your C:\ drive. * We will need to make sure that everything in our directory is set up how it is in tensorflows repo * . You can then navigate to [Tensorflows github repo](https://github.com/tensorflow/models) and download their models repo and either clone or download a zip file and place it into the tensorflow folder. If you downloaded a zip file go ahead and extract it inside your tensorflow folder and change the name from 'models-master' to just 'models'.


#### Step 2b. Download the training model and config files
The training model I decided to use the ssd_mobilenet_v1_coco model which can be found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md). If you go to the page you can see that there are different models that have higher accuracy but consequently take longer so I believe this was a good option for speed and efficiency. You will also need the .config file for this which you can get [here](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v1_pets.config) which is the same thing but using the configuration for pets rather than ordinary objects

Unzip the contents of the model file and place it inside the C:/tensorflow1/models/research/object_detection directory and the .config file inside of the C:\tensorflow1\models\research\object_detection\training folder.

#### Step 2c. Setting up conda environment
Open up a commmand window and enter this command
```bash
conda create -n tensorflow pip python=3.6
```
What we are doing is creating a new environment with the name of tensorflow and making sure it can use pip and setting the python version within the enironment as 3.6.

Next, to use our environment you we can activate it by using this command in our window:
```bash
conda activate tensorflow
```

We will then install tensorflow. From my tests, I used tensorflow-gpu 1.14.
```bash
pip install tensorflow-gpu==1.14
```

Some of the libraries that we will need to install are:
```bash
pip install pillow
pip install lxml
pip install pandas
pip install jupyter
pip install cython
pip install opencv-python
pip install contextlib2
pip install matplotlib
conda install -c anaconda protobuf
```

**Note that if you do not have a GPU then you can simply use tensorflow instead of tensorflow-gpu in the command above**

#### Step 2d. Set up PATH variable
A PYTHONPATH variable must be created and this can be done from any directory
First enter the command:
```bash
set PYTHONPATH=C:\tensorflow\models;C:\tensorflow\models\research;C:\tensorflow\models\research\slim
```
Then enter this command:
```bash
set PATH=%PATH%;PYTHONPATH
```
** Note that everytime you close the window or deactivate the environment you will have to enter the above command **

#### Step 2f. Create protobufs and run setup.pyWith the images labeled, it’s time to generate the TFRecords that serve as input data to the TensorFlow training model.
Next, compile the Protobuf files, which are used by TensorFlow to configure model and training parameters.
In the anaconda environment, switch directories to the research directory:
```bash
cd C:\tensorflow1\models\research
```
Then, copy and paster the following:
```bash
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto .\object_detection\protos\flexible_grid_anchor_generator.proto
```
This creates a name_pb2.py file from every name.proto file in the \object_detection\protos folder.
Finally, run the following commands from the C:\tensorflow1\models\research directory:
```bash
python setup.py build
python setup.py install
```
### Step 3. Classifying images from videos
The challenge's google drive contains a python script that lets us extract the images from a corresponding video. It is called AITAS_utils.py. Run this with a video that you can also download from the same google drive. I chose to use the video 6.mp4 as there are coordinates tht correspond with this video and it will be useful for the future. 

#### Step 3a. Using LablImg
![LabelImg](https://raw.githubusercontent.com/tzutalin/labelImg/master/demo/demo3.jpg)
We will be using LabelImg to classify our images that we extracted from the video. There are many videos on youtube that you can find on how to use this program.
You can follow the instructions on their [github page](https://github.com/tzutalin/labelImg) to find out how to install it. I had to download the github repo into a separate folder on my machine and run the labelImg.py script in order to use it. When you are drawing the boxes around the images and saving them, the program creates .xml files that will later be used in generating tfrecord files. Once you have labeled and saved each image, there will be one .xml file for each image.
**I have went ahead and clasified a little over 1000 images from the video to speed this process up as it can be meticulous but if you want to use your own images then you can go ahead and use this program to do so**

After you are done classifying your images, place 80% of the images inside the \object_detection\images\train directory and the other 20% inside the \object_detection\images\test directory. 

#### Step 4. Generate Training data
With the images labeled, we can then generate the TFRecords that serve as input data to the TensorFlow training model. This walkthrough  uses the xml_to_csv.py and generate_tfrecord.py scripts from Dat Tran’s Raccoon Detector dataset, with some modifications to work with our directory structure. First, the image .xml data will be used to create .csv files containing all the data for the train and test images. From the \object_detection folder, issue the following command in the Anaconda command prompt:
```bash
python xml_to_csv.py
```
Next, open the generate_tfrecord.py file in a text editor. Replace the label map starting at line 31 with your own label map, where each object is assigned an ID number. This will be labeled as boat for our purposes and will only had one ID since that is all we are looking for.
```python
# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'boat':
        return 1
    else:
        None
```
Then, generate the TFRecord files by issuing these commands from the \object_detection folder:
```bash
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
```
These generate a train.record and a test.record file in \object_detection. These will be used to train the new object detection classifier.

### Step 5. Create label map and configure training

#### Step 5a.
The label map tells the trainer what each object is by defining a mapping of class names to class ID numbers. Use a text editor to create a new file and save it as labelmap.pbtxt in the C:\tensorflow1\models\research\object_detection\training folder. (Make sure the file type is .pbtxt, not .txt)
```bash
item {
  id: 1
  name: 'nine'
}
```
**Note The label map ID numbers should be the same as what is defined in the generate_tfrecord.py file**

#### Step 5b.
Finally, the object detection training pipeline must be configured. It defines which model and what parameters will be used for training.

Navigate to C:\tensorflow1\models\research\object_detection\samples\configs and copy the ssd_mobilenet_v1_pets.config file into the \object_detection\training directory.

Make the following changes to the ssd_mobilenet_v1_pets.config file. Note: The paths must be entered with single forward slashes (NOT backslashes), or TensorFlow will give a file path error when trying to train the model! Also, the paths must be in double quotation marks ( " ), not single quotation marks ( ' ).

- Line 9: Change the num_classes to the number of different objects that you want the classifier to detect. In this case since we are only looking for the boat, our value here will be 1.

- Line 156. Change fine_tune_checkpoint to:
  - fine_tune_checkpoint : "C:/tensorflow1/models/research/object_detection/ssd_mobilenet_v1_coco_2018_01_28/model.ckpt"
  - Lines 176 and 178. In the train_input_reader section, change input_path and label_map_path to:
  - input_path : "C:/tensorflow1/models/research/object_detection/train.record"
  - label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"

- Line 183. Change num_examples to the number of images you have in the \images\test directory.

- Lines 186 and 190. In the eval_input_reader section, change input_path and label_map_path to:
  - input_path : "C:/tensorflow1/models/research/object_detection/test.record"
  - label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"

Save the file after the changes have been made. Then we will be ready to begin training.

### 6. Run the Training

**Note you may have issues with running the python script so before we run it, do a pip install pycocotools-windows just make sure that we are set to continue**

From the \object_detection directory, issue the following command to begin training:
```bash
python model_main.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config
```
If everything has been set up correctly, TensorFlow will initialize the training. The initialization can take up to 30 seconds before the actual training begins. When training begins, it will somewhat similar to this:

<p align="center">
  <img src="https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/blob/master/doc/training.jpg">
</p>

There may be some slight differences in appearances due to the fact that the original train.py is deprecated and we are using model_main.py and it runs in steps of 100.

Each step of training reports the loss. It will start high and get lower and lower as training progresses. I recommend allowing your model to train until the loss consistently drops below 2. This can take several ours(depending on how powerful your CPU and GPU are) and I left mine running over night for about 7 hours and it got through 28,000 steps.

### Step 7. Export Inference Graph
Now that training is complete, the last step is to generate the frozen inference graph (.pb file). From the \object_detection folder, issue the following command, where “XXXX” in “model.ckpt-XXXX” should be replaced with the **highest-numbered** .ckpt file in the training folder:
```bash
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```
This creates a frozen_inference_graph.pb file in the \object_detection\inference_graph folder. The .pb file contains the object detection classifier.

  
### Step 8. Use Your Newly Trained Object Detection Classifier
To run any of the scripts, type “idle” in the Anaconda Command Prompt (with the “tensorflow” virtual environment activated) and press ENTER. This will open IDLE, and from there, you can open any of the scripts and run them.

For this we will be using the Object_detection_video.py script but first we will have to edit this so that we can tell it what video to look for.
- Line 35: Change this to the name of the video, in this case I used 6.mp4. **Make sure that the video is inside the inference_graph folder so that the script can find it**. 

- Line 51: Change NUM_CLASSES to equal 1 since that is the number of classes that we will be looking for.

Once youve made these changes you can then save and run the script:
```bash
python Object_detection_video.py
```
This will start the video and you should begin to see the classifier working on detecting out boat!

## TO-DO's:
What I have done so far is find the distnace between the center point of the boat and the bottom center of the entire screen from an image that was ripped from the video. Using this distance and knowing the coordinates of the camera, I calculated an estimated coordinate for where the boat can be and was accurate up to 40 feet of the boat. This will need to be improved by finding out different methods to calculate the coordinates of the boat in real-time.
- Find a different formula to find the boats coordinates
- Possibly find the focal length of the lens the camera used to help with aiding finding the distance of the boat
- Train the model not only with the boat but also when the coordinates of the boat at the specific location in time in relation to the image it is classifying to build a map of coordinates that we can later use.
- Any other ideas...


**This walkthrough was inspired by [EdjeElectronics](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10) github tutorial.
