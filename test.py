import numpy as np
import cv2
import os
import torch
from tqdm import tqdm
import config
from model import model
import pandas as pd
from BBox import BBox
import time

def _putText(_img, text, coordinate, color):
    cv2.putText(_img,
                text,
                coordinate,
                cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                2, lineType=cv2.LINE_AA)
    return _img

def _get_condition(marked_box, predict_box, orig_image):
    iou = predict_box.get_IoU(marked_box)
    ioo = predict_box.get_IoO(marked_box)
    # condition = iou > 0.5
    check_ioo = lambda a, b: (iou > a) and (ioo > b)

    temp_img = orig_image.copy()
    marked_box.paint(temp_img, color['orange'])
    predict_box.paint(temp_img, color['yellow'])
    _putText(temp_img, f'iou = {iou}', (30, 30), color['red'])
    _putText(temp_img, f'ioo = {ioo}', (30, 70), color['red'])

    # cv2.imshow('ImageWindow', temp_img)
    # cv2.waitKey()
    # cv2.imwrite(f"{TEST_SAVE}/{test_images[i]}", orig_image, )
    return (iou > 0.5) or check_ioo(0.25, 0.6) \
                or check_ioo(0.15, 0.8) or check_ioo(0.05, 0.9) or \
                check_ioo(0.03, 0.99)

# set the computation device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# load the model and the trained weights


annotations_df_path = 'checkpoints/400 photos/test_annotations.csv'
# annotations_df_path = 'checkpoints/400 photos/test_annotations > 200.csv'


# # model_path = 'checkpoints/400 photos/fasterrcnn_resnet50_fpn_400_photos.pth'
# model_path = 'checkpoints/400 photos/fasterrcnn_resnet50_fpn_400_3ep_new.pth'
#
# DIR_TEST = 'checkpoints/400 photos/test'
# TEST_SAVE = 'checkpoints/400 photos/test_predictions'
# # test_results_save = 'checkpoints/400 photos/test_results.csv'
# test_results_save = 'checkpoints/400 photos/test_results > 200.csv'


# model_path = 'checkpoints/4000 photos/fasterrcnn_resnet50_fpn_4000.pth'
# model_path = 'checkpoints/4000 photos/fasterrcnn_resnet50_fpn_4000_new.pth'
# DIR_TEST = 'checkpoints/4000 photos/test'
# TEST_SAVE = 'checkpoints/4000 photos/test_predictions'
# test_results_save = 'checkpoints/4000 photos/test_results.csv'
# test_results_save = 'checkpoints/4000 photos/test_results > 200.csv'

# model_path = 'checkpoints/FLIR filtered/fasterrcnn_resnet50_fpn.pth'
# model_path = 'checkpoints/FLIR filtered/fasterrcnn_resnet50_fpn_filtered.pth'
model_path = 'checkpoints/FLIR filtered/fasterrcnn_resnet50_fpn_filt_4ep_07_03.pth'
DIR_TEST = 'checkpoints/FLIR filtered/test'
TEST_SAVE = 'checkpoints/FLIR filtered/test_predictions'
# test_results_save = 'checkpoints/FLIR filtered/test_results.csv'
test_results_save = 'checkpoints/FLIR filtered/test_results_4ep_0703_rpn_00.csv'

# test_results_save = 'checkpoints/FLIR filtered/test_results > 200.csv'


color = {
    "yellow": (32, 232, 232),
    "orange": (65, 105, 225),
    "red": (0, 0, 255),
    "green": (0, 255, 1),
    "blue": (255, 0, 0),
    "pink": (242, 0, 255)
}

model = model().to(device)
model.load_state_dict(torch.load(model_path))

# DIR_TEST = config.TEST_PATH
test_images = os.listdir(DIR_TEST)
print(f"Validation instances: {len(test_images)}")
detection_threshold = config.PREDICTION_THRES
model.eval()
with torch.no_grad():
    data = []
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    annotations_df = pd.read_csv(annotations_df_path, sep=",", encoding="cp1251")

    for i, image in tqdm(enumerate(test_images), total=len(test_images)):
        if (test_images[i] == '.DS_Store'):
            continue
        # if (test_images[i] != 'FLIR_08394.jpeg'):
        #     continue
        time_start = time.time()
        orig_image = cv2.imread(f"{DIR_TEST}/{test_images[i]}", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        image_time = time.time()
        image /= 255.0
        image = np.transpose(image, (2, 0, 1)).astype(np.float)
        image = torch.tensor(image, dtype=torch.float)  # .cuda()
        image = torch.unsqueeze(image, 0)
        image_time_end = time.time()
        cpu_device = torch.device("cpu")
        start = time.time()
        outputs = model(image)
        predict_time = time.time()
        print(f"\nimage transpose time = {image_time_end - image_time}")
        print(f"Total time = {predict_time - start}")
        time_end = time_start - time.time()
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        if len(outputs[0]['boxes']) != 0:
            # for counter in range(len(outputs[0]['boxes'])):
            # boxes

            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            for index, score in enumerate(scores):
                data.append(
                    {
                        'image_id': f'{test_images[i]}',
                        'x': boxes[index][0],
                        'y': boxes[index][1],
                        'w': boxes[index][2] - boxes[index][0],
                        'h': boxes[index][3] - boxes[index][1],
                        'area': (boxes[index][2] - boxes[index][0]) * (boxes[index][3] - boxes[index][1]),
                        'score': score,
                        'time': time_end
                    })
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            draw_boxes = boxes.copy()
            # FPS 400 = 1.14
            test_box = []
            markup = annotations_df[annotations_df.image_id == test_images[i]]
            # print(markup)
            # markup_all = annotations_all_df[annotations_all_df.image_id == test_images[i]]

            marked_boxes = [
                BBox(row[1].get('x'), row[1].get('y'), row[1].get('x2'), row[1].get('y2'))
                for row in markup.iterrows()
            ]
            # marked_boxes_all = [
            #     BBox(
            #         float(row[1].get('x')),
            #         float(row[1].get('y')),
            #         float(row[1].get('x2')),
            #         float(row[1].get('y2'))
            #     )
            #     for row in markup_all.iterrows()
            # ]
            predict_boxes = [
                BBox(box[0], box[1], box[2], box[3])
                for box in draw_boxes
            ]
            for marked_box in marked_boxes:
                for predict_box in predict_boxes:
                    if _get_condition(marked_box, predict_box, orig_image):
                        predict_box.flag = True
                        marked_box.flag = True

            # # Тут учитываем, что ранее фильтровали малые объекты. Если сетка их определила - не считаем это ошибкой
            # for marked_box_all in marked_boxes_all:
            #     for predict_box in predict_boxes:
            #         if _get_condition(marked_box_all, predict_box, orig_image):
            #             predict_box.flag = True

            temp_tp = len([box for box in marked_boxes if box.flag])
            temp_fn = len(marked_boxes) - temp_tp
            temp_fp = len([predict_box for predict_box in predict_boxes if not predict_box.flag])
            # temp_tn = len(predict_boxes) - temp_fp

            tp += temp_tp
            fp += temp_fp
            fn += temp_fn
            # tn += temp_tn
            orig_image = _putText(orig_image, f'tp = {temp_tp}', (35, 20), color['blue'])
            orig_image = _putText(orig_image, f'fn = {temp_fn}', (35, 50), color['red'])
            orig_image = _putText(orig_image, f'fp = {temp_fp}', (35, 80), color['yellow'])
            # orig_image = _putText(orig_image, f'tn = {temp_tn}', (35, 110), color['green'])

            for marked_box in marked_boxes:
                orig_image = marked_box.paint(orig_image, color['green'] if marked_box.flag else color['red'])

            for predict_box in predict_boxes:
                orig_image = predict_box.paint(orig_image, color['blue'] if predict_box.flag else color['yellow'])


            # cv2.imshow('ImageWindow', orig_image)
            # cv2.waitKey()
            cv2.imwrite(f"{TEST_SAVE}/{test_images[i]}", orig_image, )

    print(f'tp = {tp}')
    print(f'fp = {fp}')
    print(f'fn = {fn}')

    print(f'tn = {tn}')
    print(f'tpr = {tp / (tn + fn + tp + fp)}')
    print(f'fpr = {fp / (tn + fn + tp + fp)}')
    print(f'fnr = {fn / (tn + fn + tp + fp)}')
    print(f'tnr = {tn / (tn + fn + tp + fp)}')

    pd.DataFrame(data).to_csv(test_results_save, sep=',', encoding='utf-8')

    print('TEST PREDICTIONS COMPLETE')
