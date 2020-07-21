from gluoncv.model_zoo import get_model
from gluoncv.utils import export_block
import argparse
import os
import subprocess

import torch
import imp
import numpy as np


def generate_model(args):
    save_path = os.path.join(args.save_path, args.model)
    subprocess.call(["mkdir", "-p", save_path])

    # get model from zoo
    net = get_model(args.model, pretrained=True)

    model_prefix = os.path.join(save_path, "model")

    # and export mxnet model files (*.json, *params)
    export_block(
        model_prefix, net, preprocess=False, data_shape=(3, 32, 112, 112), layout="CTHW"
    )

    # modify resulting file
    import json

    sym = model_prefix + "-symbol" + ".json"
    params = model_prefix + "-0000" + ".params"
    model_symbols = json.load(open(sym))
    c = 0
    for i in range(len(model_symbols["nodes"])):
        node = model_symbols["nodes"][i]
        node["name"] = node["name"].replace("_fwd", "")
        if node["op"] == "Activation":
            act_type = node["attrs"]["act_type"]
            node["name"] = f"{act_type}_{c:03}"
            model_symbols["nodes"][i] = node
            c += 1
        if node["op"] == "squeeze":
            node["op"] = "Flatten"
            model_symbols["nodes"][i] = node
        if node["op"] == "Reshape":
            node["op"] = "reshape"
            model_symbols["nodes"][i] = node
    # update model symbol file
    json.dump(model_symbols, open(sym, "w"))

    # step 1 : Convert mxnet model files(*.json,*.params) to IR(Intermediate Representation) withMMDNN toolkit.
    os.system(
        f"python -m mmdnn.conversion._script.convertToIR -f mxnet -n {sym} -w {params} -d {model_prefix} --inputShape 3,32,112,112"
    )
    # step 2 : Generate python code describing Pytorch model
    os.system(
        f"python -m mmdnn.conversion._script.IRToCode -f pytorch --IRModelPath {model_prefix+'.pb'} --dstModelPath {model_prefix + '_' + 'kit.py'} --IRWeightPath {model_prefix+'.npy'} -dw {model_prefix+'_kit_pytorch.npy'}"
    )
    # step 3 : Create pytorch model(*.pth) file
    os.system(
        f"python -m mmdnn.conversion.examples.pytorch.imagenet_test --dump {model_prefix+'.pth'} -n {model_prefix + '_' + 'kit.py'} -w {model_prefix+'_kit_pytorch.npy'}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="i3d_resnet50_v1_kinetics400",
        help="model architecture to generate, we supports models listed at https://gluon-cv.mxnet.io/model_zoo/action_recognition.html#id114",
    )
    parser.add_argument(
        "--save_path", type=str, default="./zoo", help="path to save directory root"
    )

    args = parser.parse_args()

    generate_model(args)
