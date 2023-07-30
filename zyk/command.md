# 训练，评估
训练
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --coco_path /home/zyk/datasets/coco

评估
python main.py --batch_size 2 --no_aux_loss --eval --resume pretrained_model/detr-r50-e632da11.pth --coco_path /home/zyk/datasets/coco
