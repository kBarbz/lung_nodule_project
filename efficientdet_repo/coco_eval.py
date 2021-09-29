# Author: Zylo117

"""
COCO-Style Evaluations

put test here datasets/your_project_name/val_set_name/*.jpg
put annotations here datasets/your_project_name/annotations/instances_{val_set_name}.json
put weights here /path/to/your/weights/*.pth
change compound_coef

"""

import json
import os

import argparse
import torch
import yaml
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .backbone import EfficientDetBackbone
from .efficientdet.utils import BBoxTransform, ClipBoxes
from .utils.utils import preprocess, invert_affine, postprocess, boolean_string
from config.config import DATASET_PATH


def evaluate_coco(img_path, set_name, image_ids, coco, model, use_cuda, input_sizes, compound_coef, gpu, use_float16, params, nms_threshold, threshold=0.05):
    results = []

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        image_path = img_path + image_info['file_name']

        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[compound_coef], mean=params['mean'], std=params['std'])
        x = torch.from_numpy(framed_imgs[0])

        if use_cuda:
            x = x.cuda(gpu)
            if use_float16:
                x = x.half()
            else:
                x = x.float()
        else:
            x = x.float()

        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        features, regression, classification, anchors = model(x)

        preds = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, nms_threshold)
        
        if not preds:
            continue

        preds = invert_affine(framed_metas, preds)[0]

        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    'image_id': image_id,
                    'category_id': label + 1,
                    'score': float(score),
                    'bbox': box.tolist(),
                }

                results.append(image_result)

    if not len(results):
        raise Exception('the model does not provide any valid output, check model architecture and the data input')

    # write output
    filepath = f'results/{set_name}_bbox_results_d{compound_coef}.json'
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)


def _eval(coco_gt, image_ids, pred_json_path):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def run(weights):
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficient_det')
    ap.add_argument('-p', '--project', type=str, default='luna16', help='project file that contains parameters')
    ap.add_argument('--weights', type=str, default=weights, help='/path/to/weights')
    ap.add_argument('--nms_threshold', type=float, default=0.5,
                    help='nms threshold, don\'t change it if not for testing purposes')
    ap.add_argument('--cuda', type=boolean_string, default=True)
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--float16', type=boolean_string, default=False)
    ap.add_argument('--override', type=boolean_string, default=True,
                    help='override previous bbox results file if exists')
    ap.add_argument('-n', '--num_workers', type=int, default=12, help='num_workers of dataloader')
    ap.add_argument('--batch_size', type=int, default=12, help='The number of images per batch among all devices')
    ap.add_argument('--head_only', type=boolean_string, default=False,
                    help='whether finetunes only the regressor and the classifier, '
                         'useful in early stage convergence or small/easy datasets')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                               'suggest using \'admaw\' until the'
                                                               ' very final stage then switch to \'sgd\'')
    ap.add_argument('--num_epochs', type=int, default=500)
    ap.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    ap.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    ap.add_argument('--es_min_delta', type=float, default=0.0,
                    help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    ap.add_argument('--es_patience', type=int, default=0,
                    help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    ap.add_argument('--data_path', type=str, default=f'{DATASET_PATH}/prepared_data/',
                    help='the root folder of datasets')
    ap.add_argument('--log_path', type=str, default='logs/')
    ap.add_argument('-w', '--load_weights', type=str, default=None,
                    help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    ap.add_argument('--saved_path', type=str, default='logs/')
    ap.add_argument('--debug', type=boolean_string, default=False,
                    help='whether visualize the predicted boxes of training, '
                         'the output images will be in test/')
    ap.add_argument('--skip_pre', type=boolean_string, default=False,
                    help='Skip preprocessing')
    ap.add_argument('--skip_train', type=boolean_string, default=False,
                    help='Skip preprocessing')

    args = ap.parse_args()
    compound_coef = args.compound_coef
    nms_threshold = args.nms_threshold
    use_cuda = args.cuda
    gpu = args.device
    use_float16 = args.float16
    override_prev_results = args.override
    project_name = args.project
    weights_path = f'efficientdet_repo/weights/efficientdet-d{compound_coef}.pth' if not weights else args.weights

    print(f'running coco-style evaluation on project {project_name}, weights {weights_path}...')

    params = yaml.safe_load(open(fr'efficientdet_repo/projects/{project_name}.yml'))
    obj_list = params['obj_list']

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    SET_NAME = params['val_set']
    VAL_GT = f'{DATASET_PATH}prepared_data/annotations/instances_{SET_NAME}.json'
    VAL_IMGS = f'{DATASET_PATH}prepared_data/{SET_NAME}/images/'
    MAX_IMAGES = 10000
    coco_gt = COCO(VAL_GT)
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]
    
    if override_prev_results or not os.path.exists(f'{SET_NAME}_bbox_results.json'):
        model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                     ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.requires_grad_(False)
        model.eval()

        if use_cuda:
            model.cuda(gpu)

            if use_float16:
                model.half()

        evaluate_coco(VAL_IMGS, SET_NAME, image_ids, coco_gt, model, use_cuda, input_sizes, compound_coef, gpu, use_float16, params, nms_threshold)

    _eval(coco_gt, image_ids, f'{SET_NAME}_bbox_results.json')
