# NeuroMesh

NeuroMesh: Dynamic Patching-Driven Neural Compression for Efficient, Scalable Video Analytics

## Overview

NEUROMESH, an efficient and scalable video analytics system built on dynamic patching-driven neural compression, to enable cooperative workload
optimization for dynamics and mobility. NEUROMESH integrates dynamics-aware masked auto-encoders with a mobility-enhanced workload optimizer for better accuracy, latency, and
throughput. Masked auto-encoders adaptively filter and compress video frames according to camera feeds, preserving critical spatio-temporal patches to support efficient multi-view inference propagation on edge cluster. Meanwhile, the workload optimizer carefully balances workloads across heterogeneous cameras based on content complexity derived by
masked auto-encoders. Comprehensive evaluations demonstrate that NEUROMESH reduces latency by about 17.4%, boosts throughput by 1.4x, and maintains over 90% accuracy, surpassing state-of-the-art analytics and compression techniques. These results confirm its effectiveness for future practical applications such as multi-user augmented reality
navigation.

## Repository Structure

```
.
├── baselines
│   ├── crucio/...                  # crucio
│   └── TileClipper/...             # TileClipper
├── dataset/...                     # dataset related
├── dnn_model/...                   # inference task models
├── vitransformer                   # mae model
│   ├── engine_for_pretraining.py   # zero_decoding
│   ├── evaluate_zero_decoding.py   # evaluate_zero_decoding
│   ├── optim_factory.py            # optim_factory
│   ├── pretrain.py                 # models
│   └── pretrain_mae_vit.py         # training
├── msc                             # training related
│   ├── engine_for_pretraining.py   # traning related
│   ├── evaluate.py                 # evaluate_zero_decoding
│   ├── loss.py                     # loss functions 
│   └── utils.py                    # utils
└── eval                            # system implementations
    ├── codec.py                    # codecs implementations
    ├── edge_server.py              # edge_server implementation
    ├── camera.py                   # camera side implementation
    ├── utils.py                    # utils
    ├── capture.py                  # video capture
    ├── calc_accuracy.py            # postprocess result
    ├── load_optimizer.py           # load balancer modules, zero decoding modules
    └── load_predictor.py           # load predictor modules
```

## Setup

Our testbed includes serveral heterogeneous smart cameras (jetson series devices) and an PC as edge cluster.

on smart cameras, run

```
cd scripts && bash mesh_jetson_setup.sh 
```
this will create a conda environment `mesh`

on edge server, run

```
cd scripts && bash mesh_setup.sh 
```

## Usage

### Prepare Video Datas

prepare video datas as images folder with format `<video_name>/%06d.png` in `/data/<username>/video_dirs/` on every devices. and add discriptions in `eval/info.json`. Please see the template file for more details.

### Modify Device Configurations

Modify devices info in `eval/infos.json` and scenarios setups in `eval/scenarios/<scenario_name>`. Please see the template file for more details.

### Run NeuroMesh

to run edge server,

```
conda acitivate mesh
cd eval && python edge_server.py --scenarios <scenario_name> --output_dir <output_dir> 
```

then, run on each camera devices

```
conda acitivate mesh
cd eval && python camera.py --scenarios <scenario_name>
```

results will save at `<output_dir>`.

postprocess results,

```
conda acitivate mesh
cd eval && python calc_accuracy.py
```





