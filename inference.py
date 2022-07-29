import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import copy
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from craft_text_detector import (
    export_extra_results,
    load_craftnet_model,
    get_prediction,
    file_utils
)
from craft_text_detector.file_utils import rectify_poly
import os

from tqdm import tqdm

craft_net = load_craftnet_model(
    cuda=True, weight_path='/media/aimenext/disk1/tuanbm/CRAFT/checkpoint/26_35.727.pth')
debug_sample = '/media/aimenext/disk1/tuanbm/CRAFT/data/debug_Sample_data'
input_path = '/media/aimenext/disk1/tuanbm/CRAFT/data/test'
box_sample = '/media/aimenext/disk1/tuanbm/CRAFT/data/sample_data_no_rectify'

cal_cer = False



def draw_polygon(img, polygon):
    img = cv2.line(img, polygon[0], polygon[1], color=(255, 0, 0), thickness=3)
    img = cv2.line(img, polygon[1], polygon[2], color=(255, 0, 0), thickness=3)
    img = cv2.line(img, polygon[2], polygon[3], color=(255, 0, 0), thickness=3)
    img = cv2.line(img, polygon[3], polygon[0], color=(255, 0, 0), thickness=3)

    return img

def extract_text_box(image, name):
    image = copy.deepcopy(image)
    prediction_result = get_prediction(
        image=image,
        craft_net=craft_net,
        text_threshold=0.7,
        link_threshold=0.3,
        low_text=0.35,
        cuda=True,
        long_size=768            
    )

    for index, region in enumerate(prediction_result["boxes"]):
        
        box = np.array(rectify_poly(image, region))
        box1 = np.array(file_utils.crop_poly(image, region))
        # print(box1)
        # print(box)
        img_box = Image.fromarray(np.uint8(box1))
        box_name = name[:-4] + '_' + str(index) + '.jpg'
        
        img_box.save(os.path.join(box_sample, box_name))
    export_extra_results(
        file_name=name[:-4],
        image=image,
        regions=prediction_result["boxes"],
        heatmaps=prediction_result["heatmaps"],
        output_dir=debug_sample
    )



list_file = os.listdir(input_path)
list_file.sort()
for filename in tqdm(list_file):
        image = cv2.imread(os.path.join(input_path,filename), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        extract_text_box(image, filename)