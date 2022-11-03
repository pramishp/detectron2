import numpy as np

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

import sys
import os

fpath = os.path.dirname(os.path.abspath(__file__))
bpath = os.path.join(fpath, '../../')
densepose_path = os.path.join(bpath, 'projects/DensePose')
sys.path.append(densepose_path)

from densepose import add_densepose_config
from densepose.vis.extractor import (
    DensePoseOutputsExtractor,
    DensePoseResultExtractor,
)
from densepose.structures import DensePoseChartPredictorOutput, DensePoseEmbeddingPredictorOutput

from detectron2.inference.config import add_timmnets_config

from detectron2.inference import backbone
from detectron2.inference import roi_heads

import torch
from torch import nn
from torch.nn import functional as F

config_file = os.path.join(bpath, 'detectron2/model_zoo/configs/mobile_parsing_rcnn_b_wc2m_s3x.yaml')
model_path = os.path.join(bpath, 'models/mobile_parsing_rcnn_b_wc2m_s3x.pth')

# setup config
cfg = get_cfg()
add_densepose_config(cfg)
add_timmnets_config(cfg)
cfg.merge_from_file(config_file)

cfg.MODEL.WEIGHTS = model_path
cfg.freeze()

predictor = DefaultPredictor(cfg)


def run(images):
    results = []
    for img in images:
        with torch.no_grad():
            outputs = predictor(img)["instances"]
        result = {}
        if outputs.has("scores"):
            result["scores"] = outputs.get("scores").cpu()
            if outputs.has("pred_boxes"):
                result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
                if outputs.has("pred_densepose"):
                    if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
                        extractor = DensePoseResultExtractor()
                    elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
                        extractor = DensePoseOutputsExtractor()
                    result["pred_densepose"] = extractor(outputs)[0]

        data = [result]
        pred_box = data[0]['pred_boxes_XYXY'].data[0].cpu().numpy()
        pred_box = np.floor(pred_box).astype(int)
        i = data[0]['pred_densepose'][0].labels.cpu().numpy()
        uv = data[0]['pred_densepose'][0].uv.cpu().numpy()
        # iuv = np.stack((i, uv[1, :, :], uv[0, :, :]))
        iuv = np.concatenate((i[:, :, None], uv[0, :, :][:, :, None], uv[1, :, :][:, :, None]), axis=2)
        # iuv = np.transpose(iuv, (0,2,1))
        fine_seg_confidence = data[0]['pred_densepose'][0].__dict__['fine_segm_confidence'].cpu().numpy()
        coarse_seg_confidence = data[0]['pred_densepose'][0].__dict__['coarse_segm_confidence'].cpu().numpy()

        results.append({'iuv': iuv, 'pred_box': pred_box,
                        'scores': result['scores'],
                        'fine_seg_confidence': fine_seg_confidence,
                        'coarse_seg_confidence': coarse_seg_confidence
                        })


if __name__ == "__main__":
    from detectron2.data.detection_utils import read_image

    input_img_path = '/Users/pramish/Desktop/Codes/mmexperiments/mm-densepose/data/rgbd/pramish/1.jpg'
    img = read_image(input_img_path, format="BGR")  # predictor expects BGR image.
    results = run([img])
    print()
