import os
from xml.dom.minidom import parse
from shutil import copyfile
import random
import re


print('total image num = ', len(os.listdir(os.path.join('/content/KaggleFaceMaskDetection/data/original_data', "images"))))
wo_num = 0
w_num = 0
wo_image_num = 0
w_image_num = 0
wi_num = 0
wi_image_num = 0
for dirname, _, filenames in os.walk('/content/KaggleFaceMaskDetection/data/original_data/annotations'):
    for filename in filenames:
        dom = parse(os.path.join('/content/KaggleFaceMaskDetection/data/original_data/annotations', filename))
        root = dom.documentElement
        objects = root.getElementsByTagName("object")
        for o in objects:
            label_type = o.getElementsByTagName("name")[0].childNodes[0].data
            if label_type == 'without_mask':
                wo_num += 1
            elif label_type == 'with_mask':
                w_num += 1
            else:
                wi_num += 1
        w_image_num += 1
        for o in objects:
            label_type = o.getElementsByTagName("name")[0].childNodes[0].data
            if label_type == 'without_mask':
                wo_image_num += 1
                break
        for o in objects:
            label_type = o.getElementsByTagName("name")[0].childNodes[0].data
            if not label_type == 'without_mask' and not label_type == 'with_mask':
                wi_image_num += 1
                break
print('total without mask object: ', wo_num)
print('total with mask object: ', w_num)
print('total inncorect worn mask object: ', wi_num)
print('total images without mask object: ', wo_image_num)
print('total images with mask object: ', w_image_num)
print('total images inncorect worn mask object: ', w_image_num)

if not os.path.exists('/content/KaggleFaceMaskDetection/data/train'):
    os.mkdir('/content/KaggleFaceMaskDetection/data/train')
    os.mkdir('/content/KaggleFaceMaskDetection/data/train/images')
    os.mkdir('/content/KaggleFaceMaskDetection/data/train/annotations')
    os.mkdir('/content/KaggleFaceMaskDetection/data/test')
    os.mkdir('/content/KaggleFaceMaskDetection/data/test/images')
    os.mkdir('/content/KaggleFaceMaskDetection/data/test/annotations')
annotation_list = os.listdir('/content/KaggleFaceMaskDetection/data/original_data/annotations')

random.seed(10)
random.shuffle(annotation_list)
train_list = annotation_list[:int(len(annotation_list)/4*3)]
test_list = annotation_list[int(len(annotation_list)/4*3):]

train_num = 0
for filename in train_list:
    img_id = int(re.findall(r'\d+', filename)[0])
    image_name = '/content/KaggleFaceMaskDetection/data/original_data/images/maksssksksss' + str(img_id)+'.png'
    dom = parse(os.path.join('/content/KaggleFaceMaskDetection/data/original_data/annotations', filename))
    root = dom.documentElement
    objects = root.getElementsByTagName("object")
    wo_mask = False
    wi_mask = False
    for o in objects:
        label_type = o.getElementsByTagName("name")[0].childNodes[0].data
        if label_type == "mask_weared_incorrect":
            wi_mask = True
        if label_type == 'without_mask':
            wo_mask = True
    if wo_mask:
        for ii in range(4):
            copyfile(image_name, '/content/KaggleFaceMaskDetection/data/train/images/maksssksksss' + str(train_num)+'.png')
            copyfile(os.path.join('/content/KaggleFaceMaskDetection/data/original_data/annotations', filename), \
                     '/content/KaggleFaceMaskDetection/data/train/annotations/maksssksksss' + str(train_num)+'.xml')
            train_num += 1
    if wi_mask:
        for ii in range(20):
            copyfile(image_name, '/content/KaggleFaceMaskDetection/data/train/images/maksssksksss' + str(train_num)+'.png')
            copyfile(os.path.join('/content/KaggleFaceMaskDetection/data/original_data/annotations', filename), \
                     '/content/KaggleFaceMaskDetection/data/train/annotations/maksssksksss' + str(train_num)+'.xml')
            train_num += 1
    if wi_mask == False and wo_mask == False:
        copyfile(image_name, '/content/KaggleFaceMaskDetection/data/train/images/maksssksksss' + str(train_num) + '.png')
        copyfile(os.path.join('/content/KaggleFaceMaskDetection/data/original_data/annotations', filename), \
                 '/content/KaggleFaceMaskDetection/data/train/annotations/maksssksksss' + str(train_num) + '.xml')
        train_num += 1
    
    copyfile(image_name, '/content/KaggleFaceMaskDetection/data/train/images/maksssksksss' + str(train_num)+'.png')
    copyfile(os.path.join('/content/KaggleFaceMaskDetection/data/original_data/annotations', filename), \
              '/content/KaggleFaceMaskDetection/data/train/annotations/maksssksksss' + str(train_num)+'.xml')
    train_num += 1
        
test_num = 0
for filename in test_list:
    img_id = int(re.findall(r'\d+', filename)[0])
    image_name = '/content/KaggleFaceMaskDetection/data/original_data/images/maksssksksss' + str(img_id)+'.png'
    dom = parse(os.path.join('/content/KaggleFaceMaskDetection/data/original_data/annotations', filename))
    root = dom.documentElement
    objects = root.getElementsByTagName("object")

    copyfile(image_name, '/content/KaggleFaceMaskDetection/data/test/images/maksssksksss' + str(test_num) + '.png')
    copyfile(os.path.join('/content/KaggleFaceMaskDetection/data/original_data/annotations', filename), \
             '/content/KaggleFaceMaskDetection/data/test/annotations/maksssksksss' + str(test_num) + '.xml')
    test_num += 1
    

print('total training num: ', train_num)
print('total testing num: ', test_num)