from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3_cv_multi.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/0601.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, default="/home/detection/projects/PyTorch-YOLOv3/weights/yolov3_cv_multi_0601_cv.pth",help="if specified starts from checkpoint model")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints_YOLOv3", help="path to save weight file")

    parser.add_argument("--n_gpu", type=int, default=2, help="number of gpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")                                  
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=True, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("output", exist_ok=True)
    os.makedirs(opt.checkpoints_dir, exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            # 加载所有层的权重
            model.load_state_dict(torch.load(opt.pretrained_weights))

            # # 加载分类层之前的所有层权重
            # pretrained_dict = torch.load(opt.pretrained_weights)
            # model_dict=model.state_dict()
            # # 当新旧键名不一样时，可以用这句更新
            # # pretrained_dict={k:v for k,v in pretrained_dict.items() if k in model_dict}
            # del pretrained_dict['module_list.81.conv_81.weight']
            # del pretrained_dict['module_list.81.conv_81.bias']
            # del pretrained_dict['module_list.93.conv_93.weight']
            # del pretrained_dict['module_list.93.conv_93.bias']
            # del pretrained_dict['module_list.105.conv_105.weight']
            # del pretrained_dict['module_list.105.conv_105.bias']
            # pretrained_dict.update(model_dict)   # 有相同键时，以pretrained_dict为准
            # # for key, param in pretrained_dict.items():
            # #     print(key)
            # model.load_state_dict(pretrained_dict)
        else:
            model.load_darknet_weights(opt.pretrained_weights)
            print(model)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_gpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    # 按照config中的参数配置
    optimizer = torch.optim.Adam(model.parameters(), lr=float(model.hyperparams['learning_rate']), weight_decay=float(model.hyperparams['decay']))

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------
            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))
            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]
            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)
            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"
            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"
            print(log_str)

            model.seen += imgs.size(0)


        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"{opt.checkpoints_dir}/yolov3_ckpt_%d.pth" % epoch)
            # torch.save(model, f"{opt.checkpoints_dir}/yolov3_ckpt_%d.pth" % epoch)

            
        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.1,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            print(precision)
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            # print(ap_class, class_names)
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")



