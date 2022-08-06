#!/bin/bash
rm ~/.cache/gstreamer-1.0/registry.aarch64.bin
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvvidconv.so
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvcompositor.so
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/tegra/libnvbuf_utils.so
source ~/.bashrc
export conda deactivate
python3.8 ~/SecurelyDrive/new_detect/new_detect.py --img 416 --conf 0.4 --drv-gaze --source csicam --lstm-detect --LSTMWeights weights/regress.pkl
