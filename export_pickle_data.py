import pickle
import numpy as np
import cv2
import os
import json
import argparse
TRESHOLD = 0.8
OUPUT_DIR = "/home/appuser/output/dense_pose_outputs/"
args = argparse.ArgumentParser()
args.add_argument(
    "--path_to_pickle", required=True, help="Path to pickle"
)
params = args.parse_args()
input_file = params.path_to_pickle
output_dir = os.path.join("/home/appuser/output/dense_pose_outputs/", os.path.basename(input_file))
print("OUTPUT_DIR", output_dir)
def save_cut(filename, bbox, idx):
    image = cv2.imread(filename)
    x_l, y_l, x_r, y_r = tuple(map(int, bbox))
    cut = image[y_l:y_r, x_l:x_r]
    cv2.imwrite(os.path.join(output_dir, "cuts", f'{os.path.basename(filename)}_{idx}.jpg'), cut)
def save_bbox_info(filename, bbox, score, idx):
    with open(os.path.join(output_dir, "bboxes", f'{os.path.basename(filename)}_{idx}.json'), "w") as f:
        json.dump({"bbox": bbox, "score": score}, f)
def save_mash(filename, mash, idx):
    with open(os.path.join(output_dir, "mashes", f'{os.path.basename(filename)}_{idx}.pkl'), "wb") as f:
        pickle.dump(mash, f)
with open(input_file, 'rb') as f:
    data = pickle.load(f)
if not os.path.exists(os.path.join(output_dir, "cuts")):
    os.makedirs(os.path.join(output_dir, "cuts"))
if not os.path.exists(os.path.join(output_dir, "mashes")):
    os.makedirs(os.path.join(output_dir, "mashes"))
if not os.path.exists(os.path.join(output_dir, "bboxes")):
    os.makedirs(os.path.join(output_dir, "bboxes"))
for image_info in data:
    filename = image_info["file_name"]
    th_indecies = np.where(image_info["scores"] > TRESHOLD)[0]
    bboxes = image_info["pred_boxes_XYXY"]
    for idx in th_indecies:
        bbox = bboxes[idx].cpu().detach().numpy().tolist()
        score = float(image_info["scores"][idx].cpu().detach().numpy())
        mash = image_info["pred_densepose"][idx]
        save_cut(filename, bbox, idx)
        save_bbox_info(filename, bbox, score, idx)
        save_mash(filename, mash, idx)
