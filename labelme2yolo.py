#################################
# Date: 2024/01/02
# Author: Miles Xu
# Email: kanonxmm@163.com
# Desc.: 
#################################
# -*- coding: utf-8 -*-
import os
import math
import shutil
import click
from typing import Optional
from pathlib import Path
from collections import OrderedDict

import json
import cv2
import PIL.Image
  
from sklearn.model_selection import train_test_split
from labelme import utils


class Labelme2YOLO(object):
    
    def __init__(self, json_dir):
        self._json_dir = json_dir
        self._label_dir_path = json_dir
        self._label_id_map = self._get_label_id_map(self._json_dir)
        
    def _make_train_val_dir(self):
        self._label_dir_path = os.path.join(self._json_dir, 
                                            'YOLODataset/labels/')
        self._image_dir_path = os.path.join(self._json_dir, 
                                            'YOLODataset/images/')
        
        for yolo_path in (os.path.join(self._label_dir_path + 'train/'), 
                          os.path.join(self._label_dir_path + 'val/'),
                          os.path.join(self._image_dir_path + 'train/'), 
                          os.path.join(self._image_dir_path + 'val/')):
            if os.path.exists(yolo_path):
                shutil.rmtree(yolo_path)
            
            os.makedirs(yolo_path)    
                
    def _get_label_id_map(self, json_dir):
        label_set = set()
    
        for file_name in os.listdir(json_dir):
            if file_name.endswith('json'):
                json_path = os.path.join(json_dir, file_name)
                data = json.load(open(json_path))
                for shape in data['shapes']:
                    label_set.add(shape['label'])
        
        return OrderedDict([(label, label_id) \
                            for label_id, label in enumerate(label_set)])
    
    def _train_test_split(self, folders, json_names, val_size):
        if len(folders) > 0 and 'train' in folders and 'val' in folders:
            train_folder = os.path.join(self._json_dir, 'train/')
            train_json_names = [train_sample_name + '.json' \
                                for train_sample_name in os.listdir(train_folder) \
                                if os.path.isdir(os.path.join(train_folder, train_sample_name))]
            
            val_folder = os.path.join(self._json_dir, 'val/')
            val_json_names = [val_sample_name + '.json' \
                              for val_sample_name in os.listdir(val_folder) \
                              if os.path.isdir(os.path.join(val_folder, val_sample_name))]
            
            return train_json_names, val_json_names
        
        train_idxs, val_idxs = train_test_split(range(len(json_names)), 
                                                test_size=val_size)
        train_json_names = [json_names[train_idx] for train_idx in train_idxs]
        val_json_names = [json_names[val_idx] for val_idx in val_idxs]
        
        return train_json_names, val_json_names

    def convert(self, val_size):
        json_names = [file_name for file_name in os.listdir(self._json_dir) \
                    if os.path.isfile(os.path.join(self._json_dir, file_name)) and \
                    file_name.endswith('.json')]
        folders =  [file_name for file_name in os.listdir(self._json_dir) \
                    if os.path.isdir(os.path.join(self._json_dir, file_name))]

        if val_size is not None:
            train_json_names, val_json_names = self._train_test_split(folders, json_names, val_size)
            
            self._make_train_val_dir()
            iter_items = zip(('train/', 'val/'), (train_json_names, val_json_names))
        else:
            target_dir = str(Path(json_names[0]).parent)
            self._image_dir_path = self._json_dir
            iter_items = zip((target_dir, ), (json_names, ))


        # convert labelme object to yolo format object, and save them to files
        # also get image from labelme json file and save them under images folder
        for target_dir, json_names in iter_items:
            for json_name in json_names:
                json_path = os.path.join(self._json_dir, json_name)
                json_data = json.load(open(json_path, "r"))
                
                print('Converting %s for %s ...' % (json_name, target_dir.replace('/', '')))
                
                img_path = str(Path(json_path).with_suffix(".jpg"))
                if not Path(img_path).exists():
                    img_path = self._save_yolo_image(json_data, 
                        json_name, 
                        self._image_dir_path, 
                        target_dir
                    )
                    
                yolo_obj_list = self._get_yolo_object_list(json_data, img_path)
                self._save_yolo_label(json_name, 
                                      self._label_dir_path, 
                                      target_dir, 
                                      yolo_obj_list)
        
        # print('Generating dataset.yaml file ...')
        # self._save_dataset_yaml()
                
    def convert_one(self, json_name):
        json_path = os.path.join(self._json_dir, json_name)
        json_data = json.load(open(json_path))
        
        print('Converting %s ...' % json_name)
        
        img_path = self._save_yolo_image(json_data, json_name, 
                                         self._json_dir, '')
        
        yolo_obj_list = self._get_yolo_object_list(json_data, img_path)
        self._save_yolo_label(json_name, self._json_dir, 
                              '', yolo_obj_list)
    
    def _get_yolo_object_list(self, json_data, img_path):
        yolo_obj_list = []
        
        img_h, img_w, _ = cv2.imread(img_path).shape
        for shape in json_data['shapes']:
            # labelme circle shape is different from others
            # it only has 2 points, 1st is circle center, 2nd is drag end point
            if shape['shape_type'] == 'circle':
                yolo_obj = self._get_circle_shape_yolo_object(shape, img_h, img_w)
            elif shape['shape_type'] == 'polygon': #lll
                yolo_obj = self._get_polygon_shape_yolo_object(shape, img_h, img_w)
                yolo_obj_list.append(yolo_obj)
            elif shape['shape_type'] == 'rectangle':
                yolo_obj = self._get_other_shape_yolo_object(shape, img_h, img_w)
            
            
            #yolo_obj_list.append(yolo_obj)
            
        return yolo_obj_list
    
    def _get_circle_shape_yolo_object(self, shape, img_h, img_w):
        obj_center_x, obj_center_y = shape['points'][0]
        
        radius = math.sqrt((obj_center_x - shape['points'][1][0]) ** 2 + 
                           (obj_center_y - shape['points'][1][1]) ** 2)
        obj_w = 2 * radius
        obj_h = 2 * radius
        
        yolo_center_x= round(float(obj_center_x / img_w), 6)
        yolo_center_y = round(float(obj_center_y / img_h), 6)
        yolo_w = round(float(obj_w / img_w), 6)
        yolo_h = round(float(obj_h / img_h), 6)
            
        label_id = self._label_id_map[shape['label']]
        
        return label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h
    
    def _get_other_shape_yolo_object(self, shape, img_h, img_w): 
        def __get_object_desc(obj_port_list):
            __get_dist = lambda int_list: max(int_list) - min(int_list)
            
            x_lists = [port[0] for port in obj_port_list]        
            y_lists = [port[1] for port in obj_port_list]
            
            return min(x_lists), __get_dist(x_lists), min(y_lists), __get_dist(y_lists)
        
        obj_x_min, obj_w, obj_y_min, obj_h = __get_object_desc(shape['points'])
              
        yolo_center_x= round(float((obj_x_min + obj_w / 2.0) / img_w), 6)
        yolo_center_y = round(float((obj_y_min + obj_h / 2.0) / img_h), 6)
        yolo_w = round(float(obj_w / img_w), 6)
        yolo_h = round(float(obj_h / img_h), 6)
            
        label_id = self._label_id_map[shape['label']]
        
        return label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h
    
    # compute polygon points # add by lll
    def _get_polygon_shape_yolo_object(self, shape, img_h, img_w): 
        def __get_points_list(obj_port_list):
                  
            x_lists = [port[0] for port in obj_port_list]        
            y_lists = [port[1] for port in obj_port_list]
            
            return x_lists, y_lists
        
        label_id_polygon_points = []
        label_id = self._label_id_map[shape['label']]
        label_id_polygon_points.append(label_id)

        x_lists, y_lists = __get_points_list(shape['points'])
        for x_point,y_point in zip(x_lists,y_lists):
            yolo_x = round(float(x_point / img_w), 6)
            label_id_polygon_points.append(yolo_x)
            yolo_y = round(float(y_point / img_h), 6)
            label_id_polygon_points.append(yolo_y)
        
        
        return tuple(label_id_polygon_points)

    def _save_yolo_label(self, json_name, label_dir_path, target_dir, yolo_obj_list):
        if label_dir_path == target_dir:
            txt_path = os.path.join(label_dir_path, json_name.replace(".json", ".txt"))
        else:
            txt_path = os.path.join(
                label_dir_path, 
                target_dir, 
                json_name.replace('.json', '.txt')
            )

        with open(txt_path, 'w+') as f: #lll
            for yolo_obj_idx, yolo_obj in enumerate(yolo_obj_list):
                if len(yolo_obj) > 5: #lll
                    for point in yolo_obj:
                        point_line = '%s ' % point
                        f.write(point_line)
                    f.write('\n')
                else:
                    yolo_obj_line = '%s %s %s %s %s\n' % yolo_obj \
                        if yolo_obj_idx + 1 != len(yolo_obj_list) else \
                        '%s %s %s %s %s' % yolo_obj
                    f.write(yolo_obj_line)
                
                
    def _save_yolo_image(self, json_data, json_name, image_dir_path, target_dir):
        img_name = json_name.replace('.json', '.png')
        img_path = os.path.join(image_dir_path, target_dir,img_name)
        
        if not os.path.exists(img_path):
            img = utils.img_b64_to_arr(json_data['imageData'])
            PIL.Image.fromarray(img).save(img_path)
        
        return img_path
    
    def _save_dataset_yaml(self):
        yaml_path = os.path.join(self._json_dir, 'YOLODataset/', 'dataset.yaml')
        
        with open(yaml_path, 'w+') as yaml_file:
            yaml_file.write('train: %s\n' % \
                            os.path.join(self._image_dir_path, 'train/'))
            yaml_file.write('val: %s\n\n' % \
                            os.path.join(self._image_dir_path, 'val/'))
            yaml_file.write('nc: %i\n\n' % len(self._label_id_map))
            
            names_str = ''
            for label, _ in self._label_id_map.items():
                names_str += "'%s', " % label
            names_str = names_str.rstrip(', ')
            yaml_file.write('names: [%s]' % names_str)
    

@click.command
@click.option('--json_dir',type=click.types.STRING, help='Please input the path of the labelme json files.')
@click.option('--val_size',type=click.types.FLOAT, default=None, help='Please input the validation dataset size, for example 0.1')
@click.option('--json_name',type=click.types.STRING, default=None, help='If you put json name, it would convert only one json file to YOLO.')
def run(json_dir: str, val_size: Optional[float], json_name: str):
    convertor = Labelme2YOLO(json_dir)
    if json_name is None:
        convertor.convert(val_size=val_size)
    else:
        convertor.convert_one(json_name)


if __name__ == '__main__':
    run()
