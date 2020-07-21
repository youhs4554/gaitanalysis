# Backbone utils

Toolkit for converting state-the-of-the-arts models from gluonCV(whose backend is `MXNet`) into `Pytorch` models.

## Usage

```python
import torch
import imp
import numpy as np
MainModel = imp.load_source('MainModel', "kit_i3d_resnet50_v1.py")
x = torch.randn(5,3,16,112,112) # (B,C,T,H,W)

the_model = torch.load("i3d_resnet50_v1.pth")
prediction = the_model(x)  # (B, n_class)
```

## Steps

1. Modify \_.json

- format relu -> relu_0x (asign sequence number)
- format `moving_*` -> `running_*`
  - most simplest thing is to modify moving -> running in mxnet_parser.py

```python
# /path/to/site_packages/mmdnn/conversion/mxnet/mxnet_parser.py
self.set_weight(
    source_node.name,
    "mean",
    # self.weight_data.get(source_node.name + "_moving_mean").asnumpy(),
    self.weight_data.get(source_node.name + "_running_mean").asnumpy()
)
```

2. Slight modification on MMDNN code...(😱😱 terrible!)

- add this function to convert `mean` operation of `mxnet` -> `IR`

```python
def rename_mean(self, source_node):
  IR_node = self._convert_identity_operation(source_node)

  # mean axis
  IR_node.attr["axis"].i = int(source_node.get_attr("axis"))

  # output shape
  self.set_output_shape(source_node, IR_node)
```

- disable topological ordering(it seems 🐛, ommit some operations..)

```python
# Spot 1: /path/to/site_packages/mmdnn/conversion/mxnet/mxnet_parser.py
def gen_IR(self):
  self.IR_layer_map = dict()
  # for layer in self.mxnet_graph.topological_sort:
  for layer in self.mxnet_graph.layer_map.keys():
      current_node = self.mxnet_graph.get_node(layer)
      ...

# Spot 2 : /path/to/site-packages/mmdnn/conversion/pytorch/pytorch_emitter.py
def gen_code(self, phase):
  self.add_init(1, """
                      def __init__(self, weight_file):
                          super(KitModel, self).__init__()
                          global _weights_dict
                          _weights_dict = load_weights(weight_file)
                      """)

  self.add_body(1, "def forward(self, x):")

  # for layer in self.IR_graph.topological_sort:
  for layer in self.IR_graph.layer_map:
    current_node = self.IR_graph.get_node(layer)
    node_type = current_node.type
```

3. Convert mxnet model files(`_.json`,`_.params`) to`IR(Intermediate Representation)`with`MMDNN` toolkit.

```bash
python -m mmdnn.conversion._script.convertToIR -f mxnet -n i3d_resnet50_v1_kinetics400-symbol.json -w i3d_resnet50_v1_kinetics400-0000.params -d i3d_resnet50_v1 --inputShape 3,32,112,112
```

4. Generate python code describing Pytorch model

```bash
python -m mmdnn.conversion._script.IRToCode -f pytorch --IRModelPath i3d_resnet50_v1.pb --dstModelPath kit_i3d_resnet50_v1.py --IRWeightPath i3d_resnet50_v1.npy -dw kit_pytorch.npy
```

5. Generate a Pytorch model(`_.pth`) file containing model & weights

```bash
python -m mmdnn.conversion.examples.pytorch.imagenet_test --dump i3d_resnet50_v1.pth -n kit_i3d_resnet50_v1.py -w kit_pytorch.npy
```
