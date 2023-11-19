# Pot_Detection

 - **Paper**: [Stainless steel cylindrical pot outer surface defect detection method based on cascade neural network](***)
 - **Dataset**: [Pot Datasets](https://drive.google.com/file/d/1e35vI2heuz3JZW03aDpnIzq6mybACJW3/view?usp=drive_link)

    Aiming at the problem that it is difficult to detect fine defects on the outer surface of stainless steel cylindrical pot, a detection method based on cascade neural network was proposed

### Download Code and Dataset

1. Clone the TubeContourDetection repository
    ```Shell
    git clone https://github.com/SunCihan/Pot_detection.git
    ```

2. Download [Pot Datasets](https://drive.google.com/file/d/1e35vI2heuz3JZW03aDpnIzq6mybACJW3/view?usp=drive_link)
    ```Shell
    $POT_ROOT/datasets
   ./data_box_seg
   ./yolo_data5
   ./data_seg
   ./test_original
    ```
   All the pot data used in the experiments are stored in the datasets folder, where:

   ./data_box_seg contains all raw images and label files;

   ./yolo_data5 for training the YOLOv5 model;

   ./data_seg is used to train the FCN model;

   ./test_original is used for the final detection of the cascade network

[//]: # (    The METCD contains multi-exposure &#40;ME&#41; images of 72 different scenes constructed with tubes, 30 of them are used for FCN training &#40;train set&#41;, 10 of them are used for evaluation &#40;validation set&#41;, and the rest are used for additional testing &#40;test set&#41;.)

[//]: # (    )
[//]: # (    Each sample of this dataset contains 9 images collected at different exposure times, the corresponding HDR image and tube contour labels with different widths.)

[//]: # (    )
[//]: # (    ![image]&#40;https://github.com/chexqi/Tube_Contour_Detection/blob/master/A_sequence_of_tube_ME_images.jpg&#41;)

[//]: # (    )
[//]: # (    ![image]&#40;https://github.com/chexqi/Tube_Contour_Detection/blob/master/HDR_image_and_labels.jpg&#41;)
    
[//]: # (3. Pre-trained model can alse be [downloaded]&#40;https://drive.google.com/file/d/1YGyoxAHBpFO6YnNNlwvqitJu_NDmrzHi/view?usp=sharing&#41; directly for validation or testing.)

### Install Environment
Clone repo and install requirements.txt in a Python>=3.7.0 environment, including PyTorch>=1.7.
   ```bash
   git clone https://github.com/SunCihan/Pot_detection.git  # clone
   pip install -r requirements.txt  # install
   ```

[//]: # (    python              3.6.7)

[//]: # (    opencv-python       3.4.3.18   )

[//]: # (    torch               1.4.0                 )

[//]: # (    torchsummary        1.5.1                 )

[//]: # (    torchvision         0.5.0                 )

[//]: # (    Some other libraries &#40;find what you miss when running the code.&#41;)
    
### Preparation for Training, Evaluation and Testing
<details>
<summary>Training</summary>

1. Training  a YOLO model by running python train.py on the./yolo_data5 dataset
    ```Shell
   python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5s.yaml  --batch-size 128
    ```
2. 2023.06.16Unet_seg is the FCN model code, run _01TrainMain.py under the./data_seg dataset to train an FCN model
    ```Shell
   python _01TrainMain.py
    ```

2. Validation
    ```Shell
    $TCD_ROOT python _20ValiMain.py
    ```
    Evaluation with `TCD_ROOT/METCD/Val`. We employ three evaluation metrics: 
    
    (1) Mean average precision (mAP), the higher the better.
     
    (2) Maximum F-measure at optimal dataset scale (MF-ODS), the higher the better.
     
    (3) Dilate inaccuracy at optimal dataset scale (DIA-ODS), the lower the better.
    
<details>
<summary>Testing</summary>

1.  Results of the object detection model
    ```Shell
    python detect.py --save-txt --save-conf
    ```
    Evaluation with `POT_ROOT/yolo_data5/images/test`. The following is a partial presentation of the pots object detection results.
    
![image](https://github.com/SunCihan/Pot_detection/blob/main/Object%20Detection.jpg)

2.  Results of the FCN segmentation model
    ```Shell
    python _40TestMain.py
    ```
    Evaluation with `POT_ROOT/data_seg/test`. The following is a partial presentation of the pots segmentation results.

![image](https://github.com/SunCihan/Pot_detection/blob/main/Segmentation.jpg)

3.  Detection results of the cascade structure
    ```Shell
    python _01CXQ_main.py
    ```
    Evaluation with `POT_ROOT/test_original`. The following is a partial presentation of the pots Cascade prediction.

![image](https://github.com/SunCihan/Pot_detection/blob/main/Cascade%20prediction.jpg)

### License

This code and METCD is released under the MIT License (refer to the LICENSE file for details).


### Citing

If you find this code or METCD useful in your research, please consider citing:

    @article{TubeContourDetection_METCD,
        Author = {Xiaoqi Cheng, Junhua Sun, Fuqiang Zhou},
        Title = {A fully convolutional network for tube contour detection via multi-exposure images},
        Journal = {Submitted to Expert Systems with Applications},
        Year = {2020.**}
    }

