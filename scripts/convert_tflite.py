import os
import sys
import argparse
import ast
import cv2
import time
import torch
import onnx
import torchvision
import numpy as np
import onnx_tf
import tensorflow as tf

sys.path.insert(1, os.getcwd())
from SimpleHRNet import SimpleHRNet
from misc.visualization import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict, check_video_rotation
from misc.utils import find_person_id_associations


def main(camera_id, filename, hrnet_m, hrnet_c, hrnet_j, hrnet_weights, hrnet_joints_set, image_resolution,
         single_person, yolo_version, use_tiny_yolo, disable_tracking, max_batch_size, disable_vidgear, save_video,
         video_format, video_framerate, device, enable_tensorrt):
    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    image_resolution = ast.literal_eval(image_resolution)

    fourcc = cv2.VideoWriter_fourcc(*video_format)  # video format
    # video_writer = cv2.VideoWriter('output.avi', fourcc, 1, (1800, 4000))
    
    if yolo_version == 'v3':
        if use_tiny_yolo:
            yolo_model_def = "./models_/detectors/yolo/config/yolov3-tiny.cfg"
            yolo_weights_path = "./models_/detectors/yolo/weights/yolov3-tiny.weights"
        else:
            yolo_model_def = "./models_/detectors/yolo/config/yolov3.cfg"
            yolo_weights_path = "./models_/detectors/yolo/weights/yolov3.weights"
        yolo_class_path = "./models_/detectors/yolo/data/coco.names"
    elif yolo_version == 'v5':
        # YOLOv5 comes in different sizes: n(ano), s(mall), m(edium), l(arge), x(large)
        if use_tiny_yolo:
            yolo_model_def = "yolov5n"  # this  is the nano version
        else:
            yolo_model_def = "yolov5m"  # this  is the medium version
        if enable_tensorrt:
            yolo_trt_filename = yolo_model_def + ".engine"
            if os.path.exists(yolo_trt_filename):
                yolo_model_def = yolo_trt_filename
        yolo_class_path = ""
        yolo_weights_path = ""
    else:
        raise ValueError('Unsopported YOLO version.')

    model = SimpleHRNet(
        hrnet_c,
        hrnet_j,
        hrnet_weights,
        model_name=hrnet_m,
        resolution=image_resolution,
        multiperson=not single_person,
        return_bounding_boxes=not disable_tracking,
        max_batch_size=max_batch_size,
        yolo_version=yolo_version,
        yolo_model_def=yolo_model_def,
        yolo_class_path=yolo_class_path,
        yolo_weights_path=yolo_weights_path,
        device=device,
        enable_tensorrt=False
        # enable_tensorrt=enable_tensorrt
    )
    model.model.to('cpu')
    # torch.onnx.export(model.model.module, torch.randn((1, 3, 384, 288)), 'pretrained/simplehrnet.onnx', opset_version=11)
    # torch.onnx.export('pretrained/simplehrnet2.onnx', 
    #               input_names=['input'], output_names=['output'],
    #               dynamic_axes={'input' : {0 : 'batch_size'},
    #                             'output' : {0 : 'batch_size'}})
    # Load  ONNX model
    onnx_model = onnx.load('pretrained/simplehrnet2.onnx')
    # Convert ONNX model to TensorFlow format
    tf_model = onnx_tf.backend.prepare(onnx_model)
    # Export  TensorFlow  model 
    tf_model.export_graph("pretrained/simplehrnet.tf")

    converter = tf.lite.TFLiteConverter.from_saved_model("pretrained/simplehrnet.tf")
    tflite_model = converter.convert()
    open('pretrained/simplehrnet.tflite', 'wb').write(tflite_model)


    # video_writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_id", "-d", help="open the camera with the specified id", type=int, default=0)
    parser.add_argument("--filename", "-f", help="open the specified video (overrides the --camera_id option)",
                        type=str, default=None)
    parser.add_argument("--hrnet_m", "-m", help="network model - 'HRNet' or 'PoseResNet'", type=str, default='HRNet')
    parser.add_argument("--hrnet_c", "-c", help="hrnet parameters - number of channels (if model is HRNet), "
                                                "resnet size (if model is PoseResNet)", type=int, default=48)
    parser.add_argument("--hrnet_j", "-j", help="hrnet parameters - number of joints", type=int, default=17)
    parser.add_argument("--hrnet_weights", "-w", help="hrnet parameters - path to the pretrained weights",
                        type=str, default="pretrained/pose_hrnet_w48_384x288.pth")
    parser.add_argument("--hrnet_joints_set",
                        help="use the specified set of joints ('coco' and 'mpii' are currently supported)",
                        type=str, default="coco")
    parser.add_argument("--image_resolution", "-r", help="image resolution", type=str, default='(384, 288)')
    parser.add_argument("--single_person",
                        help="disable the multiperson detection (YOLOv3 or an equivalen detector is required for"
                             "multiperson detection)",
                        action="store_true")
    parser.add_argument("--yolo_version",
                        help="Use the specified version of YOLO. Supported versions: `v3` (default), `v5`.",
                        type=str, default="v3")
    parser.add_argument("--use_tiny_yolo",
                        help="Use YOLOv3-tiny in place of YOLOv3 (faster person detection) if `yolo_version` is `v3`."
                             "Use YOLOv5n(ano) in place of YOLOv5m(edium) if `yolo_version` is `v5`."
                             "Ignored if --single_person", default=True,
                        action="store_true")
    parser.add_argument("--disable_tracking",
                        help="disable the skeleton tracking and temporal smoothing functionality",
                        action="store_true")
    parser.add_argument("--max_batch_size", help="maximum batch size used for inference", type=int, default=16)
    parser.add_argument("--disable_vidgear",
                        help="disable vidgear (which is used for slightly better realtime performance)", default=True,
                        action="store_true")  # see https://pypi.org/project/vidgear/
    parser.add_argument("--save_video", help="save output frames into a video.", action="store_true", default=True)
    parser.add_argument("--video_format", help="fourcc video format. Common formats: `MJPG`, `XVID`, `X264`."
                                               "See http://www.fourcc.org/codecs.php", type=str, default='MJPG')
    parser.add_argument("--video_framerate", help="video framerate", type=float, default=30)
    parser.add_argument("--device", help="device to be used (default: cuda, if available)."
                                         "Set to `cuda` to use all available GPUs (default); "
                                         "set to `cuda:IDS` to use one or more specific GPUs "
                                         "(e.g. `cuda:0` `cuda:1,2`); "
                                         "set to `cpu` to run on cpu.", type=str, default=None)
    parser.add_argument("--enable_tensorrt",
                        help="Enables tensorrt inference for HRnet. If enabled, a `.engine` file is expected as "
                             "weights (`--hrnet_weights`). This option should be used only after the HRNet engine "
                             "file has been generated using the script `scripts/export-tensorrt-model.py`.",
                        action='store_true')

    args = parser.parse_args()
    main(**args.__dict__)
