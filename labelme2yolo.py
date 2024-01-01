# coding:utf-8

import os
import cv2
import json
from pathlib import Path
import matplotlib.pyplot as plt

"""
1. One row per object
2. Each row is class x_center y_center width height format.
3. Box coordinates must be in normalized xywh format (from 0 - 1). 
If your boxes are in pixels, divide x_center and width by image width, and y_center and height by image height.
4. Class numbers are zero-indexed (start from 0).
"""

# labelme 中预设的类别名和类别 id 的对应关系
label_idx_map = {
    "Text": 0, "Title": 1, "Figure": 2, "Figure caption": 3, "Table": 4,
    "Table caption": 5, "Header": 6, "Footer": 7, "Reference": 8, "Equation": 9
}

color_list = [
    [200, 0, 0], [0, 200, 0], [0, 0, 200], [200, 200, 0], [0, 200, 200],
    [200, 0, 200], [0, 0, 0], [128, 128, 0],[200, 0, 0], [70, 20, 0]
]
# color_list = [[200, 0, 0], [0, 200, 0], [0, 0, 200], [200, 200, 0], [0, 200, 200], [200, 0, 200], [0, 0, 0],
#               [128, 128, 0],[200, 0, 0], [70, 20, 0], [100, 0, 200], [200, 200, 200], [0, 20, 200], [200, 0, 200], [0, 0, 0],
#               [128, 128, 0]]


def labelme_to_yolo(img_dir, json_dir, save_dir):
    name_list = [x for x in os.listdir(json_dir) if Path(x).suffix == ".json"]
    for name in name_list:
        if name.startswith('.'):
            continue
        save_path = os.path.join(save_dir, name.replace(".json", ".txt"))
        im_path = os.path.join(img_dir, name.replace(".json", ".jpg"))
        json_path = os.path.join(json_dir, name)
        im = cv2.imread(im_path)
        with open(json_path, 'r') as f:
            print("load file: ", json_path)
            label_dict = json.loads(f.read())
        height = label_dict["imageHeight"]
        width = label_dict["imageWidth"]
        loc_info_list = label_dict["shapes"]
        label_info_list = list()
        for loc_info in loc_info_list:
            obj_name = loc_info.get("label")
            label_id = label_idx_map.get(obj_name)
            # print(label_id)
            loc = loc_info.get("points")
            x0, y0 = loc[0]  # 左上角点
            x1, y1 = loc[1]  # 右下角点
            cv2.rectangle(im, (int(x0), int(y0)), (int(x1), int(y1)), color_list[label_id], 2)
            x_center = (x0 + x1) / 2 / width
            y_center = (y0 + y1) / 2 / height
            box_w = (abs(x1 - x0)) / width  # 这里使用绝对值是因为有时候先标注的右下角点
            box_h = (abs(y1 - y0)) / height
            assert box_w > 0, print((int(x0), int(y0)), (int(x1), int(y1)))
            assert box_h > 0
            label_info_list.append([str(label_id), str(x_center), str(y_center), str(box_w), str(box_h)])

        with open(save_path, 'a') as f:
            for label_info in label_info_list:
                label_str = ' '.join(label_info)
                f.write(label_str)
                f.write('\n')

        # debug
        # plt.figure(0)
        # plt.imshow(im)
        # plt.show()
        # print("xxx")


if __name__ == "__main__":
	# 图像文件夹
    image_dir = "D:\\Code\\DATAS\\CDLA_DATASET\\train"
    # labelme 的标注结果
    json_dir = "D:\\Code\\DATAS\\CDLA_DATASET\\train"
    # yolo 使用的 txt 结果
    save_dir = "D:\\Code\\DATAS\\CDLA_DATASET\\train"

    labelme_to_yolo(image_dir, json_dir, save_dir)
