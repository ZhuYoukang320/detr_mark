# -*-coding:utf-8-*-
import argparse
from models import build_model
import cv2
import torch
import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--weight', default='pretrained_model/detr-r50-e632da11.pth', type=str)

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


if __name__ == '__main__':
    arg_parser = get_args_parser()
    args = arg_parser.parse_args()
    args.aux_loss = False
    model, _, _ = build_model(args)
    weight_file = args.weight
    ckpts = torch.load(weight_file)
    model.load_state_dict(ckpts['model'])
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape((1, 1, -1))
    std = torch.tensor([0.229, 0.224, 0.225]).reshape((1, 1, -1))
    device = torch.device('cuda')
    model = model.to(device)
    cap = cv2.VideoCapture('samples/sample.avi')
    nc = 91
    if cap.isOpened():
        while True:
            ret, image = cap.read()
            h, w, c = image.shape
            if not ret:
                cap.release()
                break
            inp = np.ascontiguousarray(image[..., ::-1])  # ->RGB
            inp = torch.from_numpy(inp)
            inp = ((inp / 255. - mean) / std).permute(2, 0, 1).unsqueeze(0).to(device)
            out = model(inp)
            logits = torch.softmax(out['pred_logits'], dim=-1).squeeze()
            boxes = out['pred_boxes'].squeeze()
            confs, cids = torch.max(logits, dim=1)
            mask = (confs > 0.5) & (cids != 91)
            confs = confs[mask]
            cids = cids[mask]
            boxes = boxes[mask]
            if len(confs):
                boxes[:, [0, 2]] *= w
                boxes[:, [1, 3]] *= h
                for conf, cid, box in zip(confs, cids, boxes):
                    text = f'{cid},{conf.item():.2f}'
                    box = box.detach().cpu().numpy()
                    box[:2] -= box[2:] * 0.5
                    box[2:] += box[:2]
                    pt1 = (int(box[0]), int(box[1]))
                    pt2 = (int(box[2]), int(box[3]))
                    cv2.rectangle(image, pt1, pt2, (0, 0, 255), thickness=3)
                    cv2.putText(image, text, pt1, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))

            cv2.imshow('', image)
            if cv2.waitKey(5) == ord('q'):
                cap.release()
                break
