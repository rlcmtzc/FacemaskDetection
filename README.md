# FacemaskDetection
Facemask Detection in Google Colab with a YOLO network (Darknet) and a faster R-CNN Network (PyTorch)


## Setup and execution YOLO:
1) Create a Folder in your Ondrive named yolov3
2) Download a Dataset ([Kaggle](https://www.kaggle.com/andrewmvd/face-mask-detection?select=annotations), [Moxa3K](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7382322/))
3) Execute the jupyter Notebook 
4) For Testing execute `detection_utils.py` or for a detection on a video execute `detect_video.py`
You can use `video_converter.py` to create a Dataset from the Video, a video to images or images to a video.

## Setup and execution PyTorch [Source](https://github.com/adoskk/KaggleFaceMaskDetection):
1) Create a Folder in your Onedrive named rcnn and upload the dataset as a zip.
2) Download a Dataset ([Kaggle](https://www.kaggle.com/andrewmvd/face-mask-detection?select=annotations), [Moxa3K](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7382322/))
3) Execute the jupyter Notebook 

## Pretrained Weights:
There are already pretrained weights of the YOLO network and the faster-RCNN network trained on the Kaggle Dataset (Methode same as described in the Report)

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
