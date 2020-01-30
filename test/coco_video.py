import os
import cv2
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from config import system_configs
from utils import crop_image, normalize_
from external.nms import soft_nms, soft_nms_merge

import sys
sys.path.append("../../")                                       # Adds higher directory to python modules path.
from db.detection_video import db_configs                       # Import 'db' parameters
from db.coco_video import mscoco_classes                        # Import 'class_name' function

def _rescale_dets(detections, ratios, borders, sizes):
    xs, ys = detections[..., 0:4:2], detections[..., 1:4:2]
    xs    /= ratios[:, 1][:, None, None]
    ys    /= ratios[:, 0][:, None, None]
    xs    -= borders[:, 2][:, None, None]
    ys    -= borders[:, 0][:, None, None]
    np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
    np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)

def save_image(data, fn):
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(data)
    plt.savefig(fn, dpi = height)
    plt.close()

def kp_decode(nnet, images, K, ae_threshold=0.5, kernel=3):
    detections = nnet.test([images], ae_threshold=ae_threshold, K=K, kernel=kernel)
    detections = detections.data.cpu().numpy()
    return detections

def kp_detection(frame, nnet, score_min, debug=False, decode_func=kp_decode):
    K             = db_configs.top_k
    ae_threshold  = db_configs.ae_threshold
    nms_kernel    = db_configs.nms_kernel
    
    scales        = db_configs.test_scales
    weight_exp    = db_configs.weight_exp
    merge_bbox    = db_configs.merge_bbox
    categories    = db_configs.categories
    nms_threshold = db_configs.nms_threshold
    max_per_image = db_configs.max_per_image
    nms_algorithm = {
        "nms": 0,
        "linear_soft_nms": 1, 
        "exp_soft_nms": 2
    }[db_configs.nms_algorithm]

    top_bboxes = {}
#for ind in tqdm(range(0, num_images), ncols=80, desc="locating kps"):
#    db_ind = db_inds[ind]
#    print(db_ind)
#    image_id   = db.image_ids(db_ind)
#    image_file = db.image_file(db_ind)
    #image_file = os.path.join(system_configs.data_dir, "coco", "images", "testdev2017", "{}").format("00000000000" + str(db_ind + 1) + ".jpg")
    #if db_ind < 9:
    #    image_id   = "00000000000" + str(db_ind + 1) + ".jpg"
    #    image_file = os.path.join(system_configs.data_dir, "coco", "images", "testdev2017", "{}").format("00000000000" + str(db_ind + 1) + ".jpg")
    #elif db_ind >= 9 and db_ind < 99:
    #    image_id   = "0000000000" + str(db_ind + 1) + ".jpg"
    #    image_file = os.path.join(system_configs.data_dir, "coco", "images", "testdev2017", "{}").format("0000000000" + str(db_ind + 1) + ".jpg")
    #print(image_id)
    #print(image_file)
    
    #image      = cv2.imread(image_file)
    image = frame
    height, width = image.shape[0:2]

    detections = []

    for scale in scales:
        new_height = int(height * scale)
        new_width  = int(width * scale)
        new_center = np.array([new_height // 2, new_width // 2])

        inp_height = new_height | 127
        inp_width  = new_width  | 127

        images  = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
        ratios  = np.zeros((1, 2), dtype=np.float32)
        borders = np.zeros((1, 4), dtype=np.float32)
        sizes   = np.zeros((1, 2), dtype=np.float32)

        out_height, out_width = (inp_height + 1) // 4, (inp_width + 1) // 4
        height_ratio = out_height / inp_height
        width_ratio  = out_width  / inp_width

        resized_image = cv2.resize(image, (new_width, new_height))
        resized_image, border, offset = crop_image(resized_image, new_center, [inp_height, inp_width])

        mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)     # From CenterNet/db/coco.py
        std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)      # From CenterNet/db/coco.py

        resized_image = resized_image / 255.
        normalize_(resized_image, mean, std)

        images[0]  = resized_image.transpose((2, 0, 1))
        borders[0] = border
        sizes[0]   = [int(height * scale), int(width * scale)]
        ratios[0]  = [height_ratio, width_ratio]

        images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images = torch.from_numpy(images)
        dets   = decode_func(nnet, images, K, ae_threshold=ae_threshold, kernel=nms_kernel)
        dets   = dets.reshape(2, -1, 8)
        dets[1, :, [0, 2]] = out_width - dets[1, :, [2, 0]]
        dets   = dets.reshape(1, -1, 8)

        _rescale_dets(dets, ratios, borders, sizes)
        dets[:, :, 0:4] /= scale
        detections.append(dets)

    detections = np.concatenate(detections, axis=1)

    classes    = detections[..., -1]
    classes    = classes[0]
    detections = detections[0]

    # reject detections with negative scores
    keep_inds  = (detections[:, 4] > -1)
    detections = detections[keep_inds]
    classes    = classes[keep_inds]

    top_bboxes = {}
    for j in range(categories):
        keep_inds = (classes == j)
        top_bboxes[j + 1] = detections[keep_inds][:, 0:7].astype(np.float32)
        if merge_bbox:
            soft_nms_merge(top_bboxes[j + 1], Nt=nms_threshold, method=nms_algorithm, weight_exp=weight_exp)
        else:
            soft_nms(top_bboxes[j + 1], Nt=nms_threshold, method=nms_algorithm)
        top_bboxes[j + 1] = top_bboxes[j + 1][:, 0:5]

    scores = np.hstack([
        top_bboxes[j][:, -1] 
        for j in range(1, categories + 1)
    ])
    if len(scores) > max_per_image:
        kth    = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, categories + 1):
            keep_inds = (top_bboxes[j][:, -1] >= thresh)
            top_bboxes[j] = top_bboxes[j][keep_inds]

    # if debug:
    #     image_file = db.image_file(db_ind)
    #     image      = cv2.imread(image_file)

    #     bboxes = {}
    #     for j in range(1, categories + 1):
    #         keep_inds = (top_bboxes[image_id][j][:, -1] > 0.5)
    #         cat_name  = db.class_name(j)
    #         cat_size  = cv2.getTextSize(cat_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    #         color     = np.random.random((3, )) * 0.6 + 0.4
    #         color     = color * 255
    #         color     = color.astype(np.int32).tolist()
    #         for bbox in top_bboxes[image_id][j][keep_inds]:
    #             bbox  = bbox[0:4].astype(np.int32)
    #             if bbox[1] - cat_size[1] - 2 < 0:
    #                 cv2.rectangle(image,
    #                     (bbox[0], bbox[1] + 2),
    #                     (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 2),
    #                     color, -1
    #                 )
    #                 cv2.putText(image, cat_name, 
    #                     (bbox[0], bbox[1] + cat_size[1] + 2), 
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1
    #                 )
    #             else:
    #                 cv2.rectangle(image,
    #                     (bbox[0], bbox[1] - cat_size[1] - 2),
    #                     (bbox[0] + cat_size[0], bbox[1] - 2),
    #                     color, -1
    #                 )
    #                 cv2.putText(image, cat_name, 
    #                     (bbox[0], bbox[1] - 2), 
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1
    #                 )
    #             cv2.rectangle(image,
    #                 (bbox[0], bbox[1]),
    #                 (bbox[2], bbox[3]),
    #                 color, 2
    #             )
    #     debug_file = os.path.join(debug_dir, "{}.jpg".format(db_ind))

    # result_json = os.path.join(result_dir, "results.json")
    # detections  = db.convert_to_coco(top_bboxes)
    # with open(result_json, "w") as f:
    #     json.dump(detections, f)

    # cls_ids   = list(range(1, categories + 1))
    # image_ids = [db.image_ids(ind) for ind in db_inds]
    # db.evaluate(result_json, cls_ids, image_ids)

    detections = mscoco_classes.convert_to_coco(top_bboxes, score_min)

    return detections

def testing(frame, nnet, score_min, debug=False):
    return globals()[system_configs.sampling_function](frame, nnet, score_min, debug=debug)
