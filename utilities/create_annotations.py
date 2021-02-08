import xml.etree.ElementTree as ET
import os


def xml_to_txt(file_path, out_file_path, verbose= False):
    for filename in os.listdir(file_path):
        if filename.endswith(".xml"): 
            file_ = os.path.join(file_path, filename)
            root = ET.parse(file_).getroot()
            if verbose:
                print(f"\n----------------------- {filename} ------------------------------")
            with open(os.path.join(out_file_path, os.path.splitext(filename)[0]) + ".txt", "w") as out_file:
                for child in root:
                    if child.tag == "size":
                        image_info = []
                        for subelement in child:
                            image_info.append(float(subelement.text))
                        image_width, image_height, _ = image_info
                        if verbose:
                            print(image_width, image_height)
                    if child.tag == "object":
                        for subelement in child:
                            if subelement.tag == "name":
                                object_class = subelement.text
                                if object_class == "without_mask":
                                    object_class = 0
                                elif object_class == 'with_mask':
                                    object_class = 1
                                elif object_class == "mask_weared_incorrect":
                                    object_class = 2
                            if subelement.tag == "bndbox":
                                bndbox_info = []
                                for bndbox in subelement:
                                    bndbox_info.append(float(bndbox.text))
                                xmin, ymin, xmax, ymax = bndbox_info
                                width = xmax - xmin
                                height = ymax - ymin
                                center_x = (xmin + (width/2))/image_width
                                center_y = (ymin + (height/2))/image_height
                                width /= image_width
                                height /= image_height
                                if verbose:
                                    print(object_class, center_x, center_y, width, height)
                                out_file.write(f"{object_class} {center_x} {center_y} {width} {height}\n")
        else:
            continue


xml_to_txt("data/annotations/", "data/images/", True)

import glob
import random
images_list = glob.glob("data/images/*.png")
k = int(len(images_list) * 0.8)

print(len(images_list))
train = random.sample(images_list, k=k)
print(len(train))
test = list(set(images_list) - set(train))
print( len(test))
