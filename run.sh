#!/bin/bash
python ~/SecurelyDrive_v1.0.1/detect_drvgz.py --weights ObjectDetectorModel.pt --img 416 --conf 0.4 --drv-gaze --source 0 --name TestDetection --project Test
