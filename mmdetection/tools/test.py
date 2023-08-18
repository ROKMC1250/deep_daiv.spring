# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
from copy import deepcopy

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmdet.apis import DetInferencer
from PIL import Image
from rich.pretty import pprint
import pickle
import mmcv
import mmengine
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS
import numpy as np
import cv2
from collections import defaultdict
import os
import natsort
import pickle

seg_label = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet',
'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble',
'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower',
'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel',
'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal',
'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net',
'paper', 'pavement', 'pillow', 'plant-other', 'plastic',
'platform', 'playingfield', 'railing', 'railroad', 'river', 'road',
'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf',
'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs',
'stone', 'straw', 'structural-other', 'table', 'tent',
'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick',
'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone',
'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
'window-blind', 'window-other', 'wood']



# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--tta', action='store_true')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def demo(img):

    model_name = 'rtmdet-ins_l_8xb32-300e_coco'

    checkpoint = 'checkpoints/rtmdet-ins_l_8xb32-300e_coco_20221124_103237-78d1d652.pth'

    device = 'cuda:0'

    inferencer = DetInferencer(model_name, checkpoint, device)

    result = inferencer(img, out_dir='./output_demo')

    return result

def demo_seg(img):
    config_file = 'configs/rtmdet/rtmdet-ins_l_8xb32-300e_coco.py'
    #    Setup a checkpoint file to load
    checkpoint_file = 'checkpoints/rtmdet-ins_l_8xb32-300e_coco_20221124_103237-78d1d652.pth'
    # register all modules in mmdet into the registries
    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'

    
    image = mmcv.imread(img,channel_order='rgb')
    result = inference_detector(model, image)

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    visualizer.add_datasample(
    'result',
    image,
    data_sample=result,
    draw_gt = None,
    wait_time=0,
    )
    # visualizer.show()
    
    return result

def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # testing speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:

        if 'tta_model' not in cfg:
            warnings.warn('Cannot find ``tta_model`` in config, '
                          'we will set it as default.')
            cfg.tta_model = dict(
                type='DetTTAModel',
                tta_cfg=dict(
                    nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
        if 'tta_pipeline' not in cfg:
            warnings.warn('Cannot find ``tta_pipeline`` in config, '
                          'we will set it as default.')
            test_data_cfg = cfg.test_dataloader.dataset
            while 'dataset' in test_data_cfg:
                test_data_cfg = test_data_cfg['dataset']
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='RandomFlip', prob=1.),
                        dict(type='RandomFlip', prob=0.)
                    ],
                    [
                        dict(
                            type='PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'flip',
                                       'flip_direction'))
                    ],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpDetResults(out_file_path=args.out))

    # start testing
    runner.test()
    
def process_image_with_pixels(image, pixel_coordinates,i_min,j_min):


    image_array = np.array(image)
    white_image = np.ones_like(image_array) * 255
    for pixel_coord in pixel_coordinates:
        
        x, y = pixel_coord
        if y-j_min >= len(image_array[0]):
            white_image[x-i_min, len(image_array[0])-2] = image_array[x-i_min, len(image_array[0])-2]
        elif x - i_min >= len(image_array):
            white_image[len(image_array)-2, y-j_min] = image_array[len(image_array)-2, y-j_min]
        else:
            white_image[x-i_min, y-j_min] = image_array[x-i_min, y-j_min]
    return white_image

def resize_image(image, target_size,threshold_width, threshold_height):

    if image.shape[1] > threshold_width or image.shape[0] > threshold_height:
        height, width = image.shape[:2]
        if height > width:
            new_height = target_size
            new_width = int(width * (target_size / height))
        else:
            new_width = target_size
            new_height = int(height * (target_size / width))
        resized_image = cv2.resize(image, (new_width, new_height))
    else:
        resized_image = image


    return resized_image

if __name__ == '__main__':
    # main()
    root_dir = 'data/coco/demo_images'
    files = os.listdir(root_dir)
    sorted_files = natsort.natsorted(files)

    file = sorted_files[-1]
        
    img = root_dir +'/'+ file
    print(img)
    image = cv2.imread(img,cv2.IMREAD_COLOR)
    image = resize_image(image,1000,700,700)
    print(len(image),len(image[0]))
    file2 = sorted_files[-1]
    img2 = root_dir + '/' + file
    print(img2)
    cv2.imwrite(img2,image)

    result = demo_seg(img2) 
    result2 = demo(img2)

    # pprint(result['predictions'][0], max_length=1000)
    # pprint(result['predictions'][0]['scores'], max_length= 1000)

    best_score = []
    
    scores = result.pred_instances.scores
    mask = result.pred_instances.masks
    bboxes = result.pred_instances.bboxes
    labels = result.pred_instances.labels

    scores.cpu().numpy()
    mask_numpy = mask.cpu().numpy()
    bboxes_numpy = bboxes.cpu().numpy()
    labels_numpy = labels.cpu().numpy()

    mask2 = result2['visualization']

    for i in range(len(scores)):
        if scores[i] >= 0.3:
            best_score.append(i)


    seg_dict = defaultdict()
    label_dict = defaultdict(list)

    for seg in best_score:
        pixels = []
        bounding_box = bboxes_numpy[seg]
        j_min, i_min,j_max, i_max = bounding_box

        for i in range(len(mask_numpy[seg])):
            for j in range(len(mask_numpy[seg][i])):
                if mask_numpy[seg][i][j] == True: 
                    if i-int(i_min) <= 0:
                        x = int(i_min) + 1
                        y = j
                    elif int(i_max) - i <=0:
                        x = int(i_max) -1
                        y = j
                    elif j - int(j_min) <=0:
                        x = i
                        y = int(j_min) + 1
                    elif int(j_max) - j <=0:
                        x = i
                        y = int(j_max) -1
                    else:
                        x = i
                        y = j
                    pixels.append([x,y])
        if seg_label[labels_numpy[seg]] in label_dict.keys():
            label_dict[seg_label[labels_numpy[seg]]].append(str(seg))
            continue
        else:
            label_dict[seg_label[labels_numpy[seg]]] = [str(seg)]
        
        bounding_box_image = image[int(i_min):int(i_max),int(j_min):int(j_max)]
        # print(pixels)
        # print(len(bounding_box_image),len(bounding_box_image[0]))

        result_image = process_image_with_pixels(bounding_box_image, pixels,int(i_min),int(j_min))
        cv2.imwrite('results_demo/'+ file.split('.')[0] + '_'+seg_label[labels_numpy[seg]] +'_'+  str(seg) + '.jpg',result_image)

    with open('pkl_output/' + file.split('.')[0] + '.pkl','wb') as f:
        pickle.dump(label_dict,f)
