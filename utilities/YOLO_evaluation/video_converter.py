import cv2
import numpy as np
import os
from os.path import isfile, join
import lxml.etree
import lxml.builder    

import shutil



# Empty XML needed for PyTorch Model testing
def create_empty_annotation(filename_image, out_path, height, width, depth):
    E = lxml.builder.ElementMaker()
    annotation = E.annotation
    size = E.size
    folder = E.folder
    filename = E.filename
    the_doc = annotation(
                        folder("images"),
                        filename(f"{filename_image}"),
                        size(
                            E.width(f"{width}"),
                            E.height(f"{height}"),
                            E.depth(f"{depth}"),
                            ),  
                        E.segmented("0"),
                        E.object(
                            E.name("Placeholder"),
                            E.pose("Unspecified"),
                            E.truncated("0"),
                            E.occluded("0"),
                            E.difficult("0"),
                            E.bndbox(
                                E.xmin("0"),
                                E.ymin("0"),
                                E.xmax("0"),
                                E.ymax("0"),
                            ),
                        ),
                )   

    xml = lxml.etree.tostring(the_doc, pretty_print=True)
    with open(os.path.join(out_path, filename_image.split(".")[0]) + ".xml", "wb") as xml_file:
        xml_file.write(xml)


def create_dataset_from_video(input_path, out_path):
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    images_path = os.path.join(out_path, "images")
    annotaion_path = os.path.join(out_path, "annotations")
    if os.path.isdir(images_path):
        shutil.rmtree(images_path)
    os.mkdir(images_path)

    if os.path.isdir(annotaion_path):
        shutil.rmtree(annotaion_path)
    os.mkdir(annotaion_path)


    vidcap = cv2.VideoCapture(input_path)
    count=1
    success,image = vidcap.read()
    image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE) 
    image_name = os.path.join(images_path, f"image_{count}.png")
    cv2.imwrite(image_name, image)
    height, width, depth = image.shape
    create_empty_annotation(f"image_{count}.png", annotaion_path, height, width, depth)
    while success:
        count = count + 1
        success,image = vidcap.read()
        if success:
            image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE) 
            image_name = os.path.join(images_path, f"image_{count}.png")
            cv2.imwrite(image_name, image)     # save frame as JPG file
            height, width, depth = image.shape
            create_empty_annotation(f"image_{count}.png", annotaion_path, height, width, depth)
    cv2.destroyAllWindows()
    vidcap.release()

def video_to_images(path, out_path):
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    vidcap = cv2.VideoCapture(path)
    count=1
    success,image = vidcap.read()
    image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE) 
    cv2.imwrite(f"{out_path}/image_{count}.png", image)
    while success:
        count = count + 1
        success,image = vidcap.read()
        if success:
            image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE) 
            cv2.imwrite(f"{out_path}/image_{count}.jpg", image)     # save frame as JPG file
    cv2.destroyAllWindows()
    vidcap.release()


def images_to_video(input_path, output_path, frame_rate = 30):
    #images = sorted([img for img in os.listdir(input_path) if img.endswith(".jpg")], key = lambda x: int(x.split("_")[-1].split(".")[0]))
    images = sorted([img for img in os.listdir(input_path) if img.endswith(".png")], key = lambda x: int(x.split(".")[0][6:]))
    
    frame = cv2.imread(os.path.join(input_path, images[0]))
    height, width, _ = frame.shape

    video = cv2.VideoWriter(output_path, 0, frame_rate, (width,height))
    for image in images:
        frame = cv2.imread(os.path.join(input_path, image))
        height, width, _ = frame.shape
        video.write(cv2.imread(os.path.join(input_path, image)))

    cv2.destroyAllWindows()
    video.release()


fps = 30
#create_dataset_from_video("mask_video2.mp4", "../data/video_out/")
#video_to_images("mask_video2.mp4", "../data/video_out/")
images_to_video("../data/output", "mask_video_frcnn_oversampling.mp4", frame_rate=fps)