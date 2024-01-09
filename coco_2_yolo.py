import tqdm
import click
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO


@click.command
@click.option("--annotation-path", type=click.types.STRING, help="coco annotation file path")
@click.option("--output-dir", type=click.types.STRING, help="output dir for yolo label files")
def convert(annotation_path: str, output_dir: str):
    annotation_path = annotation_path
    save_base_path = output_dir

    data_source = COCO(annotation_file=annotation_path)
    catIds = data_source.getCatIds()
    categories = data_source.loadCats(catIds)
    categories.sort(key=lambda x: x['id'])
    classes = {}
    coco_labels = {}
    coco_labels_inverse = {}
    for c in categories:
        coco_labels[len(classes)] = c['id']
        coco_labels_inverse[c['id']] = len(classes)
        classes[c['name']] = len(classes)

    img_ids = data_source.getImgIds()
    for index, img_id in tqdm.tqdm(enumerate(img_ids), desc='change .json file to .txt file'):
        img_info = data_source.loadImgs(img_id)[0]
        file_name = img_info['file_name'].split('.')[0]
        height = img_info['height']
        width = img_info['width']

        save_path = Path(save_base_path) / (file_name + '.txt')
        if not Path(save_path).parent.exists():
            Path(save_path).parent.mkdir(parents=True)

        with open(str(save_path), mode='w') as fp:
            annotation_id = data_source.getAnnIds(img_id)
            boxes = np.zeros((0, 5))
            if len(annotation_id) == 0:
                fp.write('')
                continue
            annotations = data_source.loadAnns(annotation_id)
            lines = ''
            for annotation in annotations:
                box = annotation['bbox']
                # some annotations have basically no width / height, skip them
                if box[2] < 1 or box[3] < 1:
                    continue
                #top_x,top_y,width,height---->cen_x,cen_y,width,height
                box[0] = round((box[0] + box[2] / 2) / width, 6)
                box[1] = round((box[1] + box[3] / 2) / height, 6)
                box[2] = round(box[2] / width, 6)
                box[3] = round(box[3] / height, 6)
                label = coco_labels_inverse[annotation['category_id'] - 1]
                lines = lines + str(label)
                for i in box:
                    lines += ' ' + str(i)
                lines += '\n'
            fp.writelines(lines)
    print('finish')


if __name__ == "__main__":
    convert()
