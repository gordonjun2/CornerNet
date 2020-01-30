# Extra Instructions!

Follow the original instruction below if you are not using OpenCV in any way.

## OpenCV Installation Notes

If you are using OpenCV, ensure that the PyTorch version (0.4.1) is installed in this environment.

Steps to get OpenCV working:

**1.** Once your "CenterNet" environment is created as shown below, install OpenCV using:

  ```
	conda install -c anaconda opencv
  ```

  If error (inconsistent environment) is shown, ignore and wait until you can update your packages. Type "y" to continue.
  Note that this step will cause PyTorch to downgrade to 0.4.0.

**2.** Update PyTorch 0.4.0 to 0.4.1 using:

  ```
	conda install pytorch==0.4.1 -c pytorch
  ```

**3.** When using video_demo.py for video inference, install seaborn and imageio using:

  ```
	conda install -c anaconda seaborn
	conda install -c anaconda imageio
  ```

  If you face an OpenCV Error (cv2.imshow not implemented), install:
	pip install opencv-contrib-python

**4.** Done and enjoy CenterNet!

## Real-Time Video Objection Detection Using CornerNet

**1.** Activate CornerNet Environment

**2.** To do real-time inference on a video, use:

  ```
  python video_demo.py --model <Select your model> --testiter <Enter '500000' if using pretrained model> --file <./path/to/video.mp4> --score <Remove bboxes based on this score> <--save>
  ```

  Example:

  ```
  python video_demo.py --model CornerNet --testiter 500000 --file road.mp4 --score 0.5 --save
  ```    

  ***Tips:***

  -> Available model is: 'CornerNet'

  -> Use '500000' in --testiter if you are using the pretrained model. Enter another value if your pretrained model was trained under that no. of iterations

  -> Bboxes are kept or removed based on the score indicated in --score. The value should be 0 <= score >= 1. For example, if '--score 0.5' is used, then bboxes with confidence scores less than 0.5 will not be shown at the output.

  -> Use '--save' if you want to save each recorded frame into .jpg images. The images will be saved under CenterNet/Video_Frames/To_Convert/.

**3.** If you typed '--save' to save the recorded frames, you may also want to convert the frames into a video (no lag but it's not real-time). Do not rename the images. With the images already saved in the CornerNet/Video_Frames/To_Convert/ folder, enter CenterNet/Video_Frames and type the command:

  ```
  python frames_to_video.py
  ```

**4.** The converted video will be generated in the same directory.

**5.** Done and enjoy CornerNet!


# CornerNet: Training and Evaluation Code
Update (4/18/2019): please check out [CornerNet-Lite](https://github.com/princeton-vl/CornerNet-Lite), more efficient variants of CornerNet

Code for reproducing the results in the following paper:

[**CornerNet: Detecting Objects as Paired Keypoints**](https://arxiv.org/abs/1808.01244)  
Hei Law, Jia Deng  
*European Conference on Computer Vision (ECCV), 2018*

## Getting Started
Please first install [Anaconda](https://anaconda.org) and create an Anaconda environment using the provided package list.
```
conda create --name CornerNet --file conda_packagelist.txt
```

After you create the environment, activate it.
```
source activate CornerNet
```

Our current implementation only supports GPU so you need a GPU and need to have CUDA installed on your machine.

### Compiling Corner Pooling Layers
You need to compile the C++ implementation of corner pooling layers. 
```
cd <CornerNet dir>/models/py_utils/_cpools/
python setup.py install --user
```

### Compiling NMS
You also need to compile the NMS code (originally from [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/cpu_nms.pyx) and [Soft-NMS](https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx)).
```
cd <CornerNet dir>/external
make
```

### Installing MS COCO APIs
You also need to install the MS COCO APIs.
```
cd <CornerNet dir>/data
git clone git@github.com:cocodataset/cocoapi.git coco
cd <CornerNet dir>/data/coco/PythonAPI
make
```

### Downloading MS COCO Data
- Download the training/validation split we use in our paper from [here](https://drive.google.com/file/d/1dop4188xo5lXDkGtOZUzy2SHOD_COXz4/view?usp=sharing) (originally from [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn/tree/master/data))
- Unzip the file and place `annotations` under `<CornerNet dir>/data/coco`
- Download the images (2014 Train, 2014 Val, 2017 Test) from [here](http://cocodataset.org/#download)
- Create 3 directories, `trainval2014`, `minival2014` and `testdev2017`, under `<CornerNet dir>/data/coco/images/`
- Copy the training/validation/testing images to the corresponding directories according to the annotation files

## Training and Evaluation
To train and evaluate a network, you will need to create a configuration file, which defines the hyperparameters, and a model file, which defines the network architecture. The configuration file should be in JSON format and placed in `config/`. Each configuration file should have a corresponding model file in `models/`. i.e. If there is a `<model>.json` in `config/`, there should be a `<model>.py` in `models/`. There is only one exception which we will mention later.

To train a model:
```
python train.py <model>
```

We provide the configuration file (`CornerNet.json`) and the model file (`CornerNet.py`) for CornerNet in this repo. 

To train CornerNet:
```
python train.py CornerNet
```
We also provide a trained model for `CornerNet`, which is trained for 500k iterations using 10 Titan X (PASCAL) GPUs. You can download it from [here](https://drive.google.com/open?id=16bbMAyykdZr2_7afiMZrvvn4xkYa-LYk) and put it under `<CornerNet dir>/cache/nnet/CornerNet` (You may need to create this directory by yourself if it does not exist). If you want to train you own CornerNet, please adjust the batch size in `CornerNet.json` to accommodate the number of GPUs that are available to you.

To use the trained model:
```
python test.py CornerNet --testiter 500000 --split <split>
```

If you want to test different hyperparameters in testing and do not want to overwrite the original configuration file, you can do so by creating a configuration file with a suffix (`<model>-<suffix>.json`). You **DO NOT** need to create `<model>-<suffix>.py` in `models/`.

To use the new configuration file:
```
python test.py <model> --testiter <iter> --split <split> --suffix <suffix>
```

We also include a configuration file for multi-scale evaluation, which is `CornerNet-multi_scale.json`, in this repo. 

To use the multi-scale configuration file:
```
python test.py CornerNet --testiter <iter> --split <split> --suffix multi_scale
```
