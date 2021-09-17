import sys
from pathlib import Path
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from flask import Flask, render_template, Response

from config import ROOT_PATH

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path



sys.path.insert(0, ROOT_PATH + '/yolov5')
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, is_ascii, non_max_suppression, scale_coords, set_logging
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync


app = Flask(__name__)
camera = cv2.VideoCapture(0)

def generate_frame():
    view_img=False
    source='0'  # file/dir/URL/glob, 0 for webcam
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))


    # Initialize
    set_logging()
    device = select_device('')
    half = False
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    pt = '.pt'
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        # weights = 'yolov5s.pt'
        weights = ROOT_PATH + '/yolov5/yolov5s.pt'
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16

    imgsz = 640
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    ascii = is_ascii(names)  # names are ascii (use PIL for UTF-8)

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(camera, source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)


    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    for path, img, im0s, vid_cap in dataset:

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        t1 = time_sync()
        if pt:
            pred = model(img, augment=False, visualize=False)[0]


        # NMS
        max_det = 1000
        iou_thres = 0.45
        conf_thres = 0.25
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)
        t2 = time_sync()


        # Process predictions
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % img.shape[2:]  # print string
            line_thickness = 3
            annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    if view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = f'{names[c]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))


            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            result_img = annotator.result()
            
            ret, buffer = cv2.imencode('.jpg', result_img)
            frame = buffer.tobytes()

            yield(b'--frame\r\n'
                            b'Content-type: image/jpeg\r\n\r\n' + frame + b'\r\n')




@app.route('/')
def index():
    return render_template('index_video.html')

@app.route('/video')
def video():
    return Response(generate_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    app.run()