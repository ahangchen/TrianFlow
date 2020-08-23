import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.evaluation import eval_depth
from core.visualize import Visualizer_debug
from core.networks import Model_depth_pose, Model_flow, Model_flowposenet
from core.evaluation import load_gt_flow_kitti, load_gt_mask
import torch
from tqdm import tqdm
import pdb
import cv2
import numpy as np
import yaml


def disp2depth(disp, min_depth=0.001, max_depth=80.0):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def resize_depths(gt_depth_list, pred_disp_list):
    gt_disp_list = []
    pred_depth_list = []
    pred_disp_resized = []
    for i in range(len(pred_disp_list)):
        h, w = gt_depth_list[i].shape
        pred_disp = cv2.resize(pred_disp_list[i], (w, h))
        pred_depth = 1.0 / (pred_disp + 1e-4)
        pred_depth_list.append(pred_depth)
        pred_disp_resized.append(pred_disp)

    return pred_depth_list, pred_disp_resized


def resize_disp(pred_disp_list, gt_depths):
    pred_depths = []
    h, w = gt_depths[0].shape[0], gt_depths[0].shape[1]
    for i in range(len(pred_disp_list)):
        disp = pred_disp_list[i]
        resize_disp = cv2.resize(disp, (w, h))
        depth = 1.0 / resize_disp
        pred_depths.append(depth)

    return pred_depths


import h5py
import scipy.io as sio


def test_single_image(img_path, model, training_hw, save_dir='./'):
    img = cv2.imread(img_path)
    h, w = img.shape[0:2]
    img_resized = cv2.resize(img, (training_hw[1], training_hw[0]))
    img_t = torch.from_numpy(np.transpose(img_resized, [2, 0, 1])).float().cuda().unsqueeze(0) / 255.0
    disp = model.infer_depth(img_t)
    disp = np.transpose(disp[0].cpu().detach().numpy(), [1, 2, 0])
    disp_resized = cv2.resize(disp, (w, h))
    depth = 1.0 / (1e-6 + disp_resized)

    visualizer = Visualizer_debug(dump_dir=save_dir)
    visualizer.save_disp_color_img(disp_resized, name='demo')
    print('Depth prediction saved in ' + save_dir)


if __name__ == '__main__':
    import argparse

    arg_parser = argparse.ArgumentParser(
        description="TrianFlow testing."
    )
    arg_parser.add_argument('-c', '--config_file', default=None, help='config file.')
    arg_parser.add_argument('-g', '--gpu', type=str, default=0, help='gpu id.')
    arg_parser.add_argument('--mode', type=str, default='depth', help='mode for testing.')
    arg_parser.add_argument('--task', type=str, default='kitti_depth',
                            help='To test on which task, kitti_depth or kitti_flow or nyuv2 or demo')
    arg_parser.add_argument('--image_path', type=str, default=None,
                            help='Set this only when task==demo. Depth demo for single image.')
    arg_parser.add_argument('--pretrained_model', type=str, default=None,
                            help='directory for loading flow pretrained models')
    arg_parser.add_argument('--result_dir', type=str, default=None, help='directory for saving predictions')

    args = arg_parser.parse_args()
    if not os.path.exists(args.config_file):
        raise ValueError('config file not found.')
    with open(args.config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['img_hw'] = (cfg['img_hw'][0], cfg['img_hw'][1])
    # cfg['log_dump_dir'] = os.path.join(args.model_dir, 'log.pkl')
    cfg['model_dir'] = args.result_dir

    # copy attr into cfg
    for attr in dir(args):
        if attr[:2] != '__':
            cfg[attr] = getattr(args, attr)


    class pObject(object):
        def __init__(self):
            pass


    cfg_new = pObject()
    for attr in list(cfg.keys()):
        setattr(cfg_new, attr, cfg[attr])

    if args.task == 'demo':
        model = Model_depth_pose(cfg_new)

    model.cuda()
    weights = torch.load(args.pretrained_model)
    model.load_state_dict(weights['model_state_dict'])
    model.eval()
    print('Model Loaded.')

    if args.task == 'demo':
        test_single_image(args.image_path, model, training_hw=cfg['img_hw'], save_dir=args.result_dir)

