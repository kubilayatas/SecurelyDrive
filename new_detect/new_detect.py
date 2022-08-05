import argparse
import cv2
import torch
import random


from elements.yolo import OBJ_DETECTION
from models.experimental import attempt_load
from utils.torch_utils import select_device

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def detect():
    
    Object_detector = OBJ_DETECTION(opt.DetectorWeights, opt.device, opt.img_size)
    names = Object_detector.classes
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    if opt.source == 0:
        cap = cv2.VideoCapture(0)
    else:
        print(opt.source)
        cap = cv2.VideoCapture(opt.source, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret, frame = cap.read()
            if ret:
                # detection process
                objs = Object_detector.detect(frame)

                # plotting
                for obj in objs:
                    # print(obj)
                    label = obj['label']
                    score = obj['score']
                    [(xmin,ymin),(xmax,ymax)] = obj['bbox']
                    color = colors[names.index(label)]
                    frame = cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2) 
                    frame = cv2.putText(frame, f'{label} ({str(score)})', (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX , 0.75, color, 1, cv2.LINE_AA)

            cv2.imshow("CSI Camera", frame)
            keyCode = cv2.waitKey(30)
            if keyCode == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DetectorWeights', nargs='+', type=str, default='weights/ObjectDetectorModel.pt', help='model.pt path(s)')
    parser.add_argument('--DriverGazeWeights', nargs='+', type=str, default='DriverGazeModel.pt', help='model.pt path(s)')
    parser.add_argument('--LSTMWeights', nargs='+', type=str, default='LSTMmodel.pkl', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='csicam', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--deviceGAZE', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--deviceLSTM', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--extract-lstmdata', action='store_true', help='extract lstm data')    
    parser.add_argument('--lstm-detect', action='store_true', help='extract lstm data')    
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--drv-gaze', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    """if opt.drv_gaze:
        ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
        resnet18_config = ResNetConfig(block = BasicBlock,n_blocks = [2,2,2,2],channels = [64, 128, 256, 512])
        resnet34_config = ResNetConfig(block = BasicBlock,n_blocks = [3,4,6,3],channels = [64, 128, 256, 512])
        resnet50_config = ResNetConfig(block = Bottleneck,n_blocks = [3, 4, 6, 3],channels = [64, 128, 256, 512])
        resnet101_config = ResNetConfig(block = Bottleneck,n_blocks = [3, 4, 23, 3],channels = [64, 128, 256, 512])
        resnet152_config = ResNetConfig(block = Bottleneck,n_blocks = [3, 8, 36, 3],channels = [64, 128, 256, 512])
        OUTPUT_DIM = 2
        DriverGazeModel = ResNet(resnet50_config, OUTPUT_DIM)
        DriverGazeModel.load_state_dict(torch.load(opt.DriverGazeWeights))
        DriverGazeModel.to(select_device(opt.deviceGAZE))
        DriverGazeModel.eval()
    
    if opt.lstm_detect:
        n_hidden = 128
        n_joints = 19
        #n_categories = 6
        regression_out = 3
        n_layer = 3
        lstm_model = LSTM(n_joints,n_hidden,regression_out,n_layer)
        lstm_model.load_state_dict(torch.load(opt.LSTMWeights[0]))
        lstm_model.to(select_device(opt.deviceLSTM))
        lstm_model.eval()
    
    LABELS = [
    "000", # 0
    "001", # 1
    "010", # 2
    "011", # 3
    "100", # 4
    "110" # 5
    ]
    #opt.weights = 'best.pt'
    #opt.source = "0"
    #opt.source = "./001-AltayMirzaliyev-1.mp4"
    #opt.source = "./20200224002.mp4"
    #opt.img_size = 416
    #opt.conf = 0.4
    #opt.save_txt = True
    #opt.view_img = True
    #opt.save_conf = True
    #opt.cache = True
    #opt.drv_gaze = True"""
    if opt.source == "csicam":
        opt.source = gstreamer_pipeline(
            capture_width=3264,
            capture_height=1848,
            display_width=1280,
            display_height=720,
            framerate=28,
            flip_method=6
            )
    elif opt.source == "webcam":
        opt.source = 0
    
    with torch.no_grad():
        detect()
