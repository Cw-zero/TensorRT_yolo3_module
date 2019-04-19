from __future__ import print_function

import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

TRT_LOGGER = trt.Logger()

def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.fp16_mode = True
            builder.strict_type_constraints = True
            
            builder.max_workspace_size = 1 << 30 # 1GB
            builder.max_batch_size = 4
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            # return engine

    if os.path.exists(engine_file_path):
        print("Please delete yolov3-608.trt firstly, otherwise you can not get a new file")
    else:
        build_engine()

def main():

    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    onnx_file_path = 'yolov3-608.onnx'
    engine_file_path = "yolov3-608.trt"
    get_engine(onnx_file_path, engine_file_path)

if __name__ == '__main__':
    main()
