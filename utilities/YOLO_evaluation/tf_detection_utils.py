import cv2
import numpy as np
import os
import signal
import glob
import random
from pathlib import Path
from rich.progress import track

    
def run_detection(file_path, out_path, plot_gt=False, create_result_txt=False):
    for filename in track(os.listdir(file_path), description= f"Detecting {len(os.listdir(file_path))/2} Images..."):
        if os.path.splitext(filename)[-1] == ".png":
            test_image_path = os.path.normpath(os.path.join(file_path, filename))
            
            ground_truth = []
            with open(os.path.splitext(test_image_path)[0] + ".txt", "r") as ground_truth_file:
                ground_truth_lines = ground_truth_file.read().split("\n")
                for ground_truth_line in ground_truth_lines:
                    ground_truth_line = [float(ground_truth_element) for ground_truth_element in ground_truth_line.split()]
                    ground_truth.append(ground_truth_line)

            #print(ground_truth)
            if plot_gt:
                detect_images(test_image_path, out_path, 
                              ground_truth=ground_truth,
                              create_result_txt=create_result_txt)
            else:
                detect_images(test_image_path, out_path)

            #os.kill (os.getpid (), signal.SIGTERM)


def detect_images(image_path, out_path, ground_truth=None, create_result_txt=False):

    try:
        img = cv2.imread(image_path)#cap.read()
        height, width, _ = img.shape
    except:
        print(image_path)
        os.kill (os.getpid (), signal.SIGTERM)

    pred_res_path = os.path.join(out_path, "pred_res")
    if create_result_txt and not os.path.exists(pred_res_path):
        os.mkdir(pred_res_path)

    gt_res_path = os.path.join(out_path, "gt_res")
    if create_result_txt and not os.path.exists(gt_res_path):
        os.mkdir(gt_res_path)
    head, tail = os.path.split(image_path)
    
    #cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
 
    networkOutput = net.forward()
    
    # Loop on the outputs
    boxes = []
    confidences = []
    class_ids = []

    for detection in networkOutput[0,0]:
        
        score = float(detection[2])
        if score >= 0.5:
            
            x = int(detection[3] * width)
            y = int(detection[4] * height)

            right = int(detection[5] * width)
            bottom = int(detection[6] * height)

            w = int(right - x)
            h = int(bottom - y)

            boxes.append([x, y, w, h])
            confidences.append(score)
            class_ids.append(int(detection[1]))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    #indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.0, 0.0)


    if create_result_txt:
        gt_res_file = open(os.path.join(gt_res_path, os.path.splitext(tail)[0] + ".txt"), "w")
    if ground_truth is not None:
        for ground_truth_box in ground_truth:
            if ground_truth_box:
                label_id, x, y, w, h = ground_truth_box
                label = str(classes[int(label_id)])
                color = (0,255,255)
                x = x - w/2
                y = y - h/2
                x = int(x * width)
                y = int(y * height)
                w = int(w * width)
                h = int(h * height)
                if create_result_txt:
                    gt_res_file.write(f"{'_'.join(label.split())} {x} {y} {x+w} {y+h}\n")
                #print(label, x,y,w,h)
                if int(label_id) == 0:
                    color = (0,0,255)
                elif int(label_id) == 1:
                    color = (0,255,0)
                else:
                    color = (255,255,255)
                #color = (255,0,0)
                text = "GT: " + label
                textsize = cv2.getTextSize(text, font, 0.5, 1)[0]
                text_x = abs(x + w//2 - textsize[0]//2)
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 1)
                #cv2.putText(img, text, (text_x, y+h+10), font, 0.5, color, 1, cv2.LINE_AA)
    if create_result_txt:
        gt_res_file.close()
        pred_res_file = open(os.path.join(pred_res_path, os.path.splitext(tail)[0] + ".txt"), "w")

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            if class_ids[i] == 0:
                color = (0,0,255)
            elif class_ids[i] == 1:
                color = (0,255,0)
            else:
                color = (255,255,255)
            
            if create_result_txt:
                pred_res_file.write(f"{'_'.join(label.split())} {confidence} {x} {y} {x+w} {y+h}\n")

            cv2.rectangle(img, (x,y), (x+w, y+h), color, 1)
            text = label + " " + confidence
            textsize = cv2.getTextSize(text, font, 0.5, 1)[0]
            text_x = abs(x + w//2 - textsize[0]//2)
            #cv2.putText(img, text , (text_x, y-2), font, 0.5,color, 1, cv2.LINE_AA)
    if create_result_txt:
        pred_res_file.close()
        

    
    
    cv2.imwrite(os.path.join(out_path, os.path.splitext(tail)[0] + "_out.png"), img)


net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'output.pbtxt')


classes = []
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()
#cap = cv2.VideoCapture('test1.mp4')
font = cv2.FONT_HERSHEY_PLAIN

run_detection("../data/test_small/", "../data/result_images_small/", plot_gt=False, create_result_txt=True)
os.kill (os.getpid (), signal.SIGTERM) # End the dnn Thread... There is for sure a nicer way...