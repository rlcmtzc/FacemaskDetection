# FacemaskDetection
Facemask Detection in Google Colab with a YOLO network (Darknet) and a faster R-CNN Network (PyTorch). 3 Classes get detected: correctly worn mask, incorrectly worn mask and no worn mask.


## Setup and execution YOLO:
1) Create a Folder in your Google Drive named yolov3
2) Download a Dataset ([Kaggle](https://www.kaggle.com/andrewmvd/face-mask-detection?select=annotations), [Moxa3K](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7382322/))
3) Execute the jupyter Notebook 
4) For Testing execute `detection_utils.py` or for a detection on a video execute `detect_video.py`
You can use `video_converter.py` to create a Dataset from the Video, a video to images or images to a video.

## Setup and execution PyTorch [Source](https://github.com/adoskk/KaggleFaceMaskDetection):
1) Create a Folder in your Google Drive named rcnn and upload the dataset as a zip.
2) Download a Dataset ([Kaggle](https://www.kaggle.com/andrewmvd/face-mask-detection?select=annotations), [Moxa3K](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7382322/))
3) Execute the jupyter Notebook 

## Pretrained Weights:
There are already pretrained weights of the [YOLO](https://drive.google.com/file/d/1-9qmYOfizzTBGJA74X7Lk-xVBFHhGDgD/view?usp=sharing) (and [config file](https://drive.google.com/file/d/1XQjaqDd8TasnUdSJiGcxRzGrl0FyVZqx/view?usp=sharing)) network and the [faster-RCNN network](https://drive.google.com/file/d/1-sLAz8Nql7adqtw37BXlSxpcFgMc4MHF/view?usp=sharing) trained on the Kaggle Dataset (Methode same as described in the Report)

## Results:
All results and problems are stated in the [Report](report.pdf)
### Quantitative Results:
![MAP](images/map.png?raw=true "MAP of the trained Networks")  

![MAP](images/tpfppng.PNG?raw=true "True and False Positive Rate")  

### Qualitative Results:
![MAP](images/mask1.png?raw=true)  

![MAP](images/mask2.png?raw=true)  

![MAP](images/mask3.png?raw=true)  




### Important Scripts and usage
* **utilities/create_annotations.py**:
    * transforms xml annotations to YOLO format annotations  
    Usage of main function `xml_to_txt`:
    ```
        xml_to_txt(<folder to xml annotations>, <folder to images>, <verbos output True/False>)
    ```
* **utilities/YOLO_evaluation/detection_utils.py:**
    * Detects images with the trained YOLO Model
    Usage of main function 'run_detection':  
    ```
    run_detection(<dnn model>,
                  <classes>,
                  <path to images>,
                  <output path>, 
                  plot_gt=<True/False>, #Plots GT ontop of prediction
                  create_result_txt=<True/False> #create result txt files for later map Calculation
                 )
    ```
* **utilities/YOLO_evaluation/detect_video.py:**
     * Detects video with the trained YOLO Model   
     Usage of main function 'detect_video':
     ```
     detect_video("<path to input video>, <output path>, frame_rate=<int, framerate of result video>)  
     ``` 
     
     
* **scripts/RCNN_Tensorflow_Scripts/tf_detection_utils.py:**  
     * Detects images with the trained Tensorflow Model  (**No pretrained weights available and no train notebook**)
        Usage of main function 'run_detection':  
        ```
        run_detection(<path to images>,
                      <output path>, 
                      plot_gt=<True/False>, #Plots GT ontop of prediction
                      create_result_txt=<True/False> #create result txt files for later map Calculation
                     )
        ```
        
# Sources
* AlexeyAB. Yolo v4, v3 and v2 for windows and linux. https://github.com/AlexeyAB/
darknet. Accessed: 27.08.2020.

* Aston Zhang and Zachary C. Lipton and Mu Li and Alexander J. Smola.
Dive into deep learning. https://d2l.ai/chapter_computer-vision/anchor.html#
intersection-over-union, 2020.

* Ross B. Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik. Rich feature hierarchies
for accurate object detection and semantic segmentation. CoRR, abs/1311.2524, 2013.

* Google. Colab. https://colab.research.google.com/notebooks/intro.ipynb. Accessed: 27.08.2020.

* Google. Tensorflow. https://www.tensorflow.org/. Accessed: 14.01.2021.

* Matthias De Lange, Rahaf Aljundi, Marc Masana, Sarah Parisot, Xu Jia, Ales Leonardis,
Gregory Slabaugh, and Tinne Tuytelaars. Continual Learning: A Comparative Study on
How to Defy Forgetting in Classification Tasks. arXiv CoRR, abs/1909.08383, 2019.

* Larxel. Face mask detection. https://www.kaggle.com/andrewmvd/
face-mask-detection?select=annotations. Accessed: 01.12.2020.

* Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan,
Piotr Doll´ar, and C. Lawrence Zitnick. Microsoft COCO: Common Objects in Context. In
Proceedings of the IEEE European Conference on Computer Vision (ECCV), Z¨urich, 2014.

* PyTorch. Pytorch. https://pytorch.org/. Accessed: 14.01.2021.

* Joseph Redmon, Santosh Kumar Divvala, Ross B. Girshick, and Ali Farhadi. You only look
once: Unified, real-time object detection. CoRR, abs/1506.02640, 2015.

* Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time
object detection with region proposal networks, 2016.

* Rohith Gandhi. R-CNN, Fast R-CNN, Faster R-CNN, YOLO Object Detection Algorithms.
https://bit.ly/38es4Jb, 2018. Accessed: 01.12.2020.

* Biparnak Roy, Subhadip Nandy, Debojit Ghosh, Debarghya Dutta, Pritam Biswas, and
Tamodip Das. Moxa: A deep learning based unmanned approach for real-time monitoring of people wearing medical masks. https://www.ncbi.nlm.nih.gov/pmc/articles/
PMC7382322/. Accessed: 14.01.2021.

* Tensorflow. Tensorflow Detection Model Zoo. https://github.com/tensorflow/models/
blob/master/research/object_detection/g3doc/detection_model_zoo.md. Accessed:
12.01.2021.

* Zhongyuan Wang, Guangcheng Wang, Baojin Huang, Zhangyang Xiong, Qi Hong, Hao Wu,
Peng Yi, Kui Jiang, Nanxi Wang, Yingjiao Pei, Heling Chen, Yu Miao, Zhibing Huang, and
Jinbi Liang. Masked face recognition dataset and application, 2020.

* Mengliu Zhao. Face Mask Detection using Faster RCNN. https://github.com/adoskk/
KaggleFaceMaskDetection. Accessed: 14.01.2021.

