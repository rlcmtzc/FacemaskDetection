import cv2
font = cv2.FONT_HERSHEY_PLAIN

import numpy as np
import os
import signal
import glob
import random
from pathlib import Path
from rich.progress import track

def detect_images(net, classes, image_path, out_path, ground_truth=None, create_result_txt=False, files = False):
    if files:
        img = image_path
        height, width, _ = img.shape
    else:
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
    if not files:
        head, tail = os.path.split(image_path)

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        
        for detection in output:

            scores = detection[5:]

            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence >= 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

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
            #textsize = cv2.getTextSize(text, font, 3, 2)[0]
            text_x = abs(x + w//2 - textsize[0]//2)
            #cv2.putText(img, text , (text_x, y-2), font, 0.5,color, 1, cv2.LINE_AA)
            #cv2.putText(img, text , (text_x, y-2), font, 3,color, 2, cv2.LINE_AA)
    if create_result_txt:
        pred_res_file.close()
        

    
    if files:
        return img
    else:
        cv2.imwrite(os.path.join(out_path, os.path.splitext(tail)[0] + "_out.png"), img)
    

    
def run_detection(net, classes, file_path, out_path, plot_gt=False, create_result_txt=False):
    for filename in track(os.listdir(file_path), description= f"Detecting {len(os.listdir(file_path))/2} Images..."):
        if os.path.splitext(filename)[-1] in [".png", ".jpg"]:
            test_image_path = os.path.normpath(os.path.join(file_path, filename))
            
            ground_truth = []
            if plot_gt:
                with open(os.path.splitext(test_image_path)[0] + ".txt", "r") as ground_truth_file:
                    ground_truth_lines = ground_truth_file.read().split("\n")
                    for ground_truth_line in ground_truth_lines:
                        ground_truth_line = [float(ground_truth_element) for ground_truth_element in ground_truth_line.split()]
                        ground_truth.append(ground_truth_line)

            #print(ground_truth)
            if plot_gt:
                detect_images(net, classes, test_image_path, out_path, 
                              ground_truth=ground_truth,
                              create_result_txt=create_result_txt)
            else:
                detect_images(net, classes, test_image_path, out_path)

            #os.kill (os.getpid (), signal.SIGTERM)



def move_train_test(file_path, test_out_path, test_data_paths, train_out_path, train_data_paths):
    test_data_paths = [os.path.normpath(path) for path in test_data_paths]
    train_data_paths = [os.path.normpath(path) for path in train_data_paths]

    #FIRST CREATE THE DIRECTORIES
    for filename in os.listdir(file_path):
        norm_path = os.path.normpath(os.path.join(file_path, filename))
        if norm_path in test_data_paths:
            new_image_path = os.path.normpath(os.path.join(test_out_path, filename))
            new_annotation_path = os.path.normpath(os.path.join(test_out_path, os.path.splitext(filename)[0] + ".txt"))
            old_image_path = norm_path
            old_annotation_path = os.path.splitext(norm_path)[0] + ".txt"

            os.rename(old_annotation_path, new_annotation_path)
            os.rename(old_image_path, new_image_path)
        else:
            new_image_path = os.path.normpath(os.path.join(train_out_path, filename))
            new_annotation_path = os.path.normpath(os.path.join(train_out_path, os.path.splitext(filename)[0] + ".txt"))
            old_image_path = norm_path
            old_annotation_path = os.path.splitext(norm_path)[0] + ".txt"

            os.rename(old_annotation_path, new_annotation_path)
            os.rename(old_image_path, new_image_path)

# Only needed if one want to create a new test/train set
def create_train_test(file_path, test_path="../data/test/", train_path="../data/train/"):
    images_list = glob.glob(file_path + "*.png")

    k = int(len(images_list) * 0.8)
    print(len(images_list))
    train = random.sample(images_list, k=k)
    print(len(train))
    test = list(set(images_list) - set(train))
    print( len(test))

    move_train_test(file_path, test_path, train, train_path, test)
        # with open("train.txt", "w") as f:
        #     f.write("\n".join(train))
        # with open("test.txt", "w") as f:
        #     f.write("\n".join(test))

if __name__ == "__main__":
    #create_train_test("../data/images/", "../data/test/", "../data/train/")
    net = cv2.dnn.readNet('yolov3_training_final.weights', 'yolov3_testing.cfg')
    #net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'output.pbtxt')

    classes = []
    with open("classes.txt", "r") as f:
        classes = f.read().splitlines()
    #cap = cv2.VideoCapture('test1.mp4')
    

    run_detection(net, classes, "<input images folder>", "<output images Folder>", plot_gt=False, create_result_txt=False)
    
    # for included TG Plot and GT and Predicted txt creation
    #run_detection(net, classes, "<input images folder>", "<output images Folder>", plot_gt=True, create_result_txt=True)
    
    os.kill (os.getpid (), signal.SIGTERM) # End the dnn Thread... There is for sure a nicer way...