#!/bin/bash
python ~/SecurelyDrive/detect_drvgz.py --weights ObjectDetectorModel.pt --img 416 --conf 0.4 --drv-gaze --source 0 --name TestDetection --project Test
sleep 30s