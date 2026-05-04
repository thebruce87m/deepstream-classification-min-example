# deepstream-classification-min-example


# Train the classifier


https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/vehiclemakenet

At the time of writing this had:
```
Latest Version
pruned_onnx_v1.1.0
Updated
November 27, 2024
```

# First train the on the data

We need a minimal dataset to prove the process. ChatGPT has generated some people:

![image](data/train/blue/blue1.jpg)
![image](data/train/red/red1.jpg)
![image](data/train/green/green1.jpg)

We only have a small number of identical samples but we are just trying to prove the process.

```
data/
├── classes.txt
├── test
│   ├── blue
│   │   ├── blue1.jpg
│   │   ├── blue2.jpg
│   │   └── blue3.jpg
│   ├── green
│   │   ├── green1.jpg
│   │   ├── green2.jpg
│   │   └── green3.jpg
│   └── red
│       ├── red1.jpg
│       ├── red2.jpg
│       └── red3.jpg
├── train
│   ├── blue
│   │   ├── blue1.jpg
│   │   ├── blue2.jpg
│   │   └── blue3.jpg
│   ├── green
│   │   ├── green1.jpg
│   │   ├── green2.jpg
│   │   └── green3.jpg
│   └── red
│       ├── red1.jpg
│       ├── red2.jpg
│       └── red3.jpg
└── val
    ├── blue
    │   ├── blue1.jpg
    │   ├── blue2.jpg
    │   └── blue3.jpg
    ├── green
    │   ├── green1.jpg
    │   ├── green2.jpg
    │   └── green3.jpg
    └── red
        ├── red1.jpg
        ├── red2.jpg
        └── red3.jpg
```

The training is based on instructions here: https://docs.nvidia.com/tao/tao-toolkit/latest/text/quick_start_guide/running_from_containers.html#running-from-containers

Specs file

https://github.com/NVIDIA-TAO/tao-tutorials/blob/main/notebooks/tao_launcher_starter_kit/classification_pyt/specs/train_cats_dogs.yaml

notebook

https://github.com/NVIDIA-TAO/tao-tutorials/blob/main/notebooks/tao_launcher_starter_kit/classification_pyt/classification.ipynb


## Download the example model weights

### Option 1 - FAN small

https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/pretrained_fan_classification_imagenet?version=fan_hybrid_small

```bash

# Go to the directory to store the FAN model
cd train/pretrained_fan_classification_imagenet_vfan_hybrid_small/

# Download from nvidia
ngc registry model download-version nvidia/tao/pretrained_fan_classification_imagenet:fan_hybrid_small

# Back to where we were
cd -
```

```bash
# Calculate the md5sum
md5sum train/pretrained_fan_classification_imagenet_vfan_hybrid_small/fan_hybrid_small.pth 

# Note that this was the version available at the time
bad95b5e464120d22cc631456619cc7a  train/pretrained_fan_classification_imagenet_vfan_hybrid_small/fan_hybrid_small.pth
```

# Option 2 - Resnet18 

https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/vehiclemakenet?version=unpruned_v1.0

Note: you will need to update the model details in the yaml files, e.g.

```yaml
model:
  backbone:
    type: resnet_18
    pretrained_backbone_path: null
    freeze_backbone: false
```


# Train The Model

```bash
./train/train.sh
```

100% Accuracy - who would have thought?

```
Epoch 26: 100%|██████████| 1/1 [00:00<00:00,  5.73it/s, v_num=1, train_loss_step=0.0246, lr=0.00161, val_loss=0.132, val_acc_1=1.000, train_loss_epoch=0.0245]
Epoch 27: 100%|██████████| 1/1 [00:00<00:00,  5.41it/s, v_num=1, train_loss_step=0.023, lr=0.00129, val_loss=0.118, val_acc_1=1.000, train_loss_epoch=0.0246]]
Epoch 28: 100%|██████████| 1/1 [00:00<00:00,  5.62it/s, v_num=1, train_loss_step=0.0228, lr=0.000968, val_loss=0.105, val_acc_1=1.000, train_loss_epoch=0.023]
Epoch 29: 100%|██████████| 1/1 [00:00<00:00,  5.52it/s, v_num=1, train_loss_step=0.0221, lr=0.000645, val_loss=0.0949, val_acc_1=1.000, train_loss_epoch=0.0228]
Epoch 29: 100%|██████████| 1/1 [00:00<00:00,  1.40it/s, v_num=1, train_loss_step=0.0221, lr=0.000645, val_loss=0.086, val_acc_1=1.000, train_loss_epoch=0.0221]`Trainer.fit` stopped: `max_epochs=30` reached.
Epoch 29: 100%|██████████| 1/1 [00:01<00:00,  0.99it/s, v_num=1, train_loss_step=0.0221, lr=0.000645, val_loss=0.086, val_acc_1=1.000, train_loss_epoch=0.0221]2026-04-30 15:40:41,162 - [TAO Toolkit] - INFO - Sending telemetry data. (entrypoint.py:355)
2026-04-30 15:40:41,162 - [TAO Toolkit] - INFO - Sending telemetry data. (entrypoint.py:355)
[2026-04-30 15:40:41,162 - TAO Toolkit - TAO Toolkit - INFO] Sending telemetry data.
2026-04-30 15:40:41,906 - [TAO Toolkit] - INFO - Execution status: PASS (entrypoint.py:376)
2026-04-30 15:40:41,906 - [TAO Toolkit] - INFO - Execution status: PASS (entrypoint.py:376)
[2026-04-30 15:40:41,906 - TAO Toolkit - TAO Toolkit - INFO] Execution status: PASS
```

# Evaluate the model


```bash
./train/evaluate.sh
```

```
Testing DataLoader 0: 100%|██████████| 1/1 [00:01<00:00,  0.61it/s]┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         val_acc_1         │            1.0            │
└───────────────────────────┴───────────────────────────┘
```



# Deploy the model

Now we follow [Deploying to DeepStream for Classification TF1/TF2/PyTorch](https://docs.nvidia.com/tao/tao-toolkit/latest/text/ds_tao/classification_ds.html)

It gives 3 options:

* Option 1: Integrate the .etlt model directly in the DeepStream app. The model file is generated by export.

* Option 2: Generate a device-specific optimized TensorRT engine using TAO Deploy. The generated TensorRT engine file can also be ingested by DeepStream.

* Option 3 (Deprecated for x86 devices): Generate a device-specific optimized TensorRT engine using TAO Converter.

Notes:

* `.etlt` sounds great, right? Wrong. Later in the same page you see `From TAO 5.0.0, .etlt is deprecated.`
* `TAO Deploy` sounds great, right? Wrong. The containers that you will have to use for deepstream inference here will probably have a different version of tensorrt and therefore the .engine file will be incompatible
* Option 3 sounds deprecated, right? Correct. `(Deprecated for x86 devices)`
* Best way is to get an onnx file into your deployment environment and have deepstream convert it there to avoid incompatibilites

Here is the error you get:

```
ERROR: [TRT]: IRuntime::deserializeCudaEngine: Error Code 6: API Usage Error (The engine plan file is not compatible with this version of TensorRT, expecting library version 10.14.1.48 got 
..      , please rebuild. In checkEngineVersionCompatible at /_src/runtime/api/engine.cpp:871)
ERROR: ../nvdsinfer/nvdsinfer_model_builder.cpp:1363 Deserialize engine failed from file: /code/models/classifier_model.onnx_b16_gpu0_fp16.engine
```

# Export

Export the `.onnx` file using the `classification_pyt export` inside the tao container

```bash
./train/export-onnx.sh
```



# Build the deepstream docker with python bindings

```bash
./docker/docker-build.sh
```


# Run the docker

```bash
./docker/docker-run.sh
```



# Test out a python deepstream example from the default install 

```bash

# Run inside the docker


# Go to the folder
cd /opt/nvidia/deepstream/deepstream-9.0/sources/deepstream_python_apps/apps/deepstream-test1

# Run it
python3 \
deepstream_test_1.py \
/opt/nvidia/deepstream/deepstream-9.0/samples/streams/sample_720p.h264
```



# Test out our example with one sgie

```bash

# Run inside the docker

python3 \
deepstream_test2.py \
/code/example-data/static-video.h264
```


![image](README/screenshot.jpg)
