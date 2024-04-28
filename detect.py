# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"
                    # print(c, label, det)
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

            # Stream results
            im0 = annotator.result()

            # Building type detection
            im0 = CheckBuildingType(pred, gn, im0)
            
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")


    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

def ClosestObject(xyxy, gn, det, im0):
    import math
    import shapely.geometry as sg

    # Define the height of the image
    img_height, img_width = im0.shape[:2]

    # Calculate the center of the bounding box
    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
    center_x = xywh[0]
    center_y = xywh[1]
    center_y_top = xywh[1] - xywh[3] / 2

    # Convert normalized coordinates to pixel coordinates
    center_x_px = int(center_x * img_width)
    center_y_px = int(center_y * img_height)
    center_y_top_px = int(center_y_top * img_height)

    # Draw the lines on the image
    # cv2.line(im0, (0, center_y_px), (img_width, center_y_px), (0, 255, 0), 2)    # Horizontal line
    # cv2.line(im0, (0, center_y_top_px), (img_width, center_y_top_px), (0, 255, 0), 2)    # Horizontal line

    # Define a line from the center of object 2 going straight up
    line = sg.LineString([(center_x_px, center_y_px), (center_x_px, 0)])
    # line_cross = sg.LineString([(0, center_y_px), (img_width, center_y_px)])
    # line_cross_windows = sg.LineString([(0, center_y_top_px), (img_width, center_y_top_px)])

    # Check for collisions with objects of class 0
    last_point_x = center_x_px
    last_point_y = center_y_px
    
    found_intersections = []
    found_intersections_xyxy = []

    for *xyxy_0, _, cls_0 in det:

        if int(cls_0) == 0:
            if (xyxy_0 != xyxy):
                xywh_0 = (xyxy2xywh(torch.tensor(xyxy_0).view(1, 4)) / gn).view(-1).tolist()
                # Create a polygon representing the bounding box of object 0
                polygon_0 = sg.box((xywh_0[0] - xywh_0[2]/2) * img_width, (xywh_0[1] - xywh_0[3]/2) * img_height, (xywh_0[0] + xywh_0[2]/2) * img_width, (xywh_0[1] + xywh_0[3]/2) * img_height)
                
                # For debugging the polygons
                # cv2.rectangle(im0, (int((xywh_0[0] - xywh_0[2]/2) * img_width), int((xywh_0[1] - xywh_0[3]/2) * img_height)), 
                # (int((xywh_0[0] + xywh_0[2]/2) * img_width), int((xywh_0[1] + xywh_0[3]/2) * img_height)), 
                # (255, 0, 255), 2)

                # Check if the line intersects with the bounding box of object 0
                if line.intersects(polygon_0):
                    found_intersections.append(xywh_0)
                    found_intersections_xyxy.append(xyxy_0)

                    temp_x = int(xywh_0[0] * img_width)
                    temp_y = int(xywh_0[1] * img_height)
                    # cv2.line(im0, (last_point_x, last_point_y), (last_point_x, 0), (0, 255, 0), 2)  # Vertical line
                    # cv2.line(im0, (last_point_x, last_point_y), (temp_x, temp_y), (0, 255, 255), 2)  # Vertical line

                    # last_point_x = temp_x
                    # last_point_y = temp_y

    if not found_intersections:
        return None

    closest_intersection = None
    min_distance = float('inf')
    temp_counter = -1

    for intersection in found_intersections:
        x = intersection[0] * img_width
        y = intersection[1] * img_height
        distance = math.sqrt((last_point_x - x) ** 2 + (last_point_y - y) ** 2)

        if distance < min_distance:
            min_distance = distance
            closest_intersection = (x, y)
            temp_counter += 1

    cv2.line(im0, (last_point_x, last_point_y), (int(closest_intersection[0]), int(closest_intersection[1])), (0, 255, 255), 2)  # Vertical line
    return found_intersections_xyxy[temp_counter]

def WindowCollisions(xyxy, gn, det, im0, direction = True):
    import shapely.geometry as sg
    img_height, img_width = im0.shape[:2]
    collisions = 0
    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
    center_x = xywh[0]
    center_y = xywh[1]
    center_y_top = xywh[1] - xywh[3] / 2
    center_y_top_px = int(center_y_top * img_height)

    if (direction):
        # Windows to the right
        cv2.line(im0, (int(center_x * img_width), center_y_top_px), (img_width, center_y_top_px), (0, 255, 0), 2)    # Horizontal line
        line_cross_windows = sg.LineString([(int(center_x * img_width), center_y_top_px), (img_width, center_y_top_px)])
    else:
        # Windows to the left
        cv2.line(im0, (int(center_x * img_width), center_y_top_px), (0, center_y_top_px), (255, 255, 0), 2)    # Horizontal line
        line_cross_windows = sg.LineString([(int(center_x * img_width), center_y_top_px), (0, center_y_top_px)])

    # Check for collisions with objects of class 0
    for *xyxy_0, _, cls_0 in reversed(det):
        xywh_0 = (xyxy2xywh(torch.tensor(xyxy_0).view(1, 4)) / gn).view(-1).tolist()
        # Create a polygon representing the bounding box of object 0
        polygon_0 = sg.box((xywh_0[0] - xywh_0[2]/2) * img_width, (xywh_0[1] - xywh_0[3]/2) * img_height, (xywh_0[0] + xywh_0[2]/2) * img_width, (xywh_0[1] + xywh_0[3]/2) * img_height)

        if int(cls_0) == 0:
            # Check if the line intersects with the bounding box of object 0
            if line_cross_windows.intersects(polygon_0):
                collisions += 1
    
    return collisions

def calculate_accuracy(top, right, left):
    # Define building types with their respective values
    # Building type : Windows going up, windows going right, windows going left
    building_types = [
        {"type": "Type_1_317", "expected": [6, 6, 6]},
        {"type": "Type_1_464", "expected": [5, 6, 6]},
        {"type": "Type_1_432", "expected": [4, 4, 2]}
    ]
    
    # Define weights
    weight_top = 0.5
    weight_right = 0.25
    weight_left = 0.25
    
    # Initialize a dictionary to store accuracy for each building type
    accuracy_dict = {}
    
    # Iterate over building types
    for building in building_types:
        expected_top, expected_right, expected_left = building["expected"]
        
        # Calculate weighted deviations from expected values
        deviation_top = abs(top - expected_top)
        deviation_right = abs(right - expected_right)
        deviation_left = abs(left - expected_left)
        
        # Calculate weighted average deviation
        weighted_average_deviation = (deviation_top * weight_top + deviation_right * weight_right + deviation_left * weight_left)
        
        # Calculate maximum possible deviation
        max_deviation = (expected_top + expected_right + expected_left) / 3
        
        # Calculate accuracy
        accuracy = 1 - (weighted_average_deviation / max_deviation)
        
        # Ensure accuracy is between 0 and 1
        accuracy = max(0, accuracy)
        
        # Store accuracy for the building type
        accuracy_dict[building["type"]] = accuracy
    
    # Sort accuracy dictionary by values (accuracy) in descending order
    sorted_accuracy = sorted(accuracy_dict.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_accuracy

def write_accuracy_on_image(image, building_accuracy):
    # Define the font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    line_thickness = 2
    
    # Define colors
    color_white = (255, 255, 255)  # White color
    color_green = (0, 255, 0)  # Green color
    color_red = (0, 0, 255)  # Red color
    
    # Define the initial y-coordinate for writing text
    y = 30
    
    # Write accuracies on the image
    for building, accuracy in building_accuracy:
        text = f"{building}: {accuracy:.2f}"
        color = color_green if accuracy == building_accuracy[0][1] else color_red
        cv2.rectangle(image, (20, y - 25), (20 + 350, y + 5), color, -1)  # Draw rectangle as background
        cv2.putText(image, text, (30, y), font, font_scale, color_white, line_thickness)
        y += 40

    return image

def CheckBuildingType(pred, gn, im0):
    # Initialize variables to store information about the object of class 2 with the highest confidence
    max_confidence = 0
    max_confidence_obj = None

    # Counter for collisions with object 0
    collisions = 0
    door_collisions = 0

    # Iterate through the detections
    for i, det in enumerate(pred):
        if len(det):
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                confidence = float(conf)

                # Check if the object is of class 2 and has higher confidence than the current maximum
                if c == 2 and confidence > max_confidence:
                    max_confidence = confidence
                    max_confidence_obj = (xyxy, confidence)  # Store the bounding box and confidence

    # If there is an object of class 2 with the highest confidence, proceed to draw the line
    if max_confidence_obj is not None:
        xyxy, max_confidence = max_confidence_obj
        
        temp = xyxy
        collisions_top = collisions_right = collisions_left = 0

        while (temp != None):
            temp = ClosestObject(temp, gn, det, im0)
            collisions_top += 1

        print(f"Number of collisions with window to the top: {collisions_top}")

        collisions_left = WindowCollisions(xyxy, gn, det, im0, False)
        print(f"Number of collisions with window to the left: {collisions_left}")

        collisions_right = WindowCollisions(xyxy, gn, det, im0, True)
        print(f"Number of collisions with window to the right: {collisions_right}")

        building_accuracy = calculate_accuracy(collisions_top, collisions_right, collisions_left)
        write_accuracy_on_image(im0, building_accuracy)
        print("Building Accuracy:")
        for building, accuracy in building_accuracy:
            print(building, ":", accuracy)
    else:
        print("No door found!")


    return im0

def parse_opt():
    """Parses command-line arguments for YOLOv5 detection, setting inference options and model configurations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes YOLOv5 model inference with given options, checking requirements before running the model."""
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
