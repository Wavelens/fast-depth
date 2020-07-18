import os
import time
import csv
import cv2
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True

import models
from metrics import AverageMeter, Result
import utils

args = utils.parse_command()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu # Set the GPU.

fieldnames = ['rmse', 'mae', 'delta1', 'absrel',
            'lg10', 'mse', 'delta2', 'delta3', 'data_time', 'gpu_time']
best_fieldnames = ['best_epoch'] + fieldnames
best_result = Result()
best_result.set_to_worst()

def main():
    global args, best_result, output_directory, train_csv, test_csv

    # Data loading code
    print("=> creating data loaders...")
    valdir = os.path.join('..', 'data', args.data, 'val')

    val_dataset = torch.Tensor(cv2.divide(cv2.resize(cv2.imread("data/nyudepthv2/val/0000000140.jpg"), (128, 128)), 255.))
    val_dataset = val_dataset.unsqueeze(0)

    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    print("=> data loaders created.")

    # evaluation mode
    model = models.MobileNetSkipAdd((640, 480))
    # model = model.load_state_dict(torch.load('model_best.pth.tar')['state_dict'])
    args.start_epoch = 0
    validate(val_loader, model, args.start_epoch, write_to_file=False)


def validate(val_loader, model, epoch, write_to_file=True):
    average_meter = AverageMeter()
    model.eval()
    end = time.time()
    i = 0
    cam = cv2.VideoCapture("http://192.168.178.195:4747/mjpegfeed?640x480")
    while True:
        input = cv2.resize(cam.read()[1], (640, 480))
        cv2.imshow("IN", input)
        print(input)
        input = torch.Tensor(np.reshape(input, [3, 480, 640])).unsqueeze(0).float()
        # torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            pred = model(input)
        # torch.cuda.synchronize()
        gpu_time = time.time() - end

        pred = np.array(pred.squeeze(0))*10.
        print(pred)

        cv2.imshow("PRED", cv2.resize(np.reshape(pred, [480, 640, 1]), (640, 480)))
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
