import sys
sys.path.append("/home/trnkat/Documents/BigBrotherICU/detectron2_repo")
sys.path.append(".")

from slowfast.visualization.predictor import DefaultPredictor
from slowfast.visualization.async_predictor import AsyncDemo, AsyncVis
from slowfast.visualization.ava_demo_precomputed_boxes import (
    AVAVisualizerWithPrecomputedBox,
)
from slowfast.visualization.predictor import ActionPredictor
import numpy as np
import time
import torch
import tqdm
import cv2

from slowfast.utils import logging
from slowfast.visualization.predictor import DefaultPredictor, Predictor
from slowfast.visualization.async_predictor import AsyncDemo, AsyncVis, draw_predictions
from slowfast.visualization.ava_demo_precomputed_boxes import (
    AVAVisualizerWithPrecomputedBox,
)
from slowfast.visualization.video_visualizer import VideoVisualizer

from slowfast.utils import logging
from slowfast.utils.parser import load_config, parse_args
from slowfast.visualization.utils import TaskInfo
from slowfast.datasets import cv2_transform

logger = logging.get_logger(__name__)

args = parse_args()
cfg = load_config(args)

np.random.seed(cfg.RNG_SEED)
torch.manual_seed(cfg.RNG_SEED)
# Setup logging format.
logging.setup_logging(cfg.OUTPUT_DIR)
# Print config.
logger.info("Run demo with config:")
logger.info(cfg)
common_classes = (
    cfg.DEMO.COMMON_CLASS_NAMES
    if len(cfg.DEMO.LABEL_FILE_PATH) != 0
    else None
)

video_vis = VideoVisualizer(
        num_classes=cfg.MODEL.NUM_CLASSES,
        class_names_path=cfg.DEMO.LABEL_FILE_PATH,
        top_k=cfg.TENSORBOARD.MODEL_VIS.TOPK_PREDS,
        thres=cfg.DEMO.COMMON_CLASS_THRES,
        lower_thres=cfg.DEMO.UNCOMMON_CLASS_THRES,
        common_class_names=common_classes,
        colormap=cfg.TENSORBOARD.MODEL_VIS.COLORMAP,
        mode=cfg.DEMO.VIS_MODE,
    )

def draw_predictions(task, video_vis):
    """
    Draw prediction for the given task.
    Args:
        task (TaskInfo object): task object that contain
            the necessary information for visualization. (e.g. frames, preds)
            All attributes must lie on CPU devices.
        video_vis (VideoVisualizer object): the video visualizer object.
    """
    boxes = task.bboxes
    frames = task.frames
    preds = task.action_preds
    if boxes is not None:
        img_width = task.img_width
        img_height = task.img_height
        if boxes.device != torch.device("cpu"):
            boxes = boxes.cpu()
        boxes = cv2_transform.revert_scaled_boxes(
            task.crop_size, boxes, img_height, img_width
        )

    keyframe_idx = len(frames) // 2 - task.num_buffer_frames
    draw_range = [
        keyframe_idx - task.clip_vis_size,
        keyframe_idx + task.clip_vis_size,
    ]
    buffer = frames[: task.num_buffer_frames]
    frames = frames[task.num_buffer_frames :]
    if boxes is not None:
        if len(boxes) != 0:
            frames = video_vis.draw_clip_range(
                frames,
                preds,
                boxes,
                keyframe_idx=keyframe_idx,
                draw_range=draw_range,
            )
    else:
        frames = video_vis.draw_clip_range(
            frames, preds, keyframe_idx=keyframe_idx, draw_range=draw_range
        )
    del task

    return buffer + frames

predictor = Predictor(cfg=cfg, gpu_id=0)

def open_file(fname):
    img = cv2.imread(fname)
    return img

filename = '/data/ikem_hackathon/sedi_lezi_lezi.png'

task = TaskInfo()
task.add_frames(0, [open_file(x) for x in [filename]*15])
task.num_buffer_frames = 0
oy, ox, dim = task.frames[0].shape
task.img_width, task.img_height = 320, 240
task.crop_size = cfg.DATA.TEST_CROP_SIZE
task.clip_vis_size = cfg.DEMO.CLIP_VIS_SIZE
# task.bboxes = None #[[] for x in range(len(task.frames))]

task = predictor(task)
with open(f"{filename}_results_tt.txt", "w") as ofile:
    ofile.write(f"{task.bboxes.numpy().tolist()}\n")
    ofile.write(f"{task.action_preds.numpy().tolist()}\n")

# code how to get the real bboxes
if task.bboxes is not None:
    img_width = task.img_width
    img_height = task.img_height
    if task.bboxes.device != torch.device("cpu"):
        task.bboxes = task.bboxes.cpu()
    boxes = cv2_transform.revert_scaled_boxes(
        task.crop_size, task.bboxes, img_height, img_width
    )

o2 = draw_predictions(task, video_vis)
cv2.imwrite(f'{filename}_results_tt.png', o2[7])