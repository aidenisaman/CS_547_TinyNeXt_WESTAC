# Speed Benchmark

### Results

[Throughtput on NVIDIA GeForce RTX 3090 and AMD EPYC 7542 32-Core Processor.](throughput.log)

| Model | Input | Params (M) | FLOPs (M) | GPU | CPU | Top-1 (%) | Top-5 (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ShuffleNetv2-1.0 | 224 | 2.3 | 146 | 12085 | 104 | 69.4 | 88.9 |
| PVTv2-b0 | 224 | 3.7 | 572 | 2586 | 26 | 70.5 | - |
| MobileNetv1 | 224 | 4.2 | 579 | 5310 | 45 | 70.6 | - |
| EfficientViT-M2 | 224 | 4.2 | 201 | 16688 | 141 | 70.8 | 90.2 |
| MobileOne-S0 | 224 | 2.1 | 275 | 8410 | 79 | 71.4 | - |
| **TinyNeXt-T** | 224 | **1.0** | 259 | 7257 | 85 | **71.5** | **90.3** |
| MobileNetv2-1.0 | 224 | 3.5 | 300 | 4572 | 38 | 72.0 | 91.0 |
| MobileViTv2-0.5 | 256 | 1.4 | 466 | 4435 | 22 | 70.2 | - |
| MobileViT-XXS | 256 | 1.3 | 410 | 3374 | 23 | 69.0 | - |
| EdgeNeXt-XXS | 256 | 1.3 | 261 | 4679 | 64 | 71.2 | - |
| EMO-1M | 224 | 1.3 | 261 | 3949 | 40 | 71.5 | 90.4 |
| **TinyNeXt-S** | 224 | **1.3** | 304 | **6555** | **75** | **72.7** | **91.0** |
| MobileNetv2-1.4 | 224 | 6.1 | 585 | 3091 | 19 | 74.7 | - |
| EdgeViT-XXS | 224 | 4.1 | 557 | 3326 | 36 | 74.4 | - |
| MobileViT-XS | 256 | 2.3 | 986 | 2028 | 8 | 74.8 | - |
| EdgeNeXt-XS | 256 | 2.3 | 538 | 2906 | 32 | 75.0 | - |
| EMO-2M | 224 | 2.3 | 439 | 2779 | 29 | 75.1 | 92.2 |
| **TinyNeXt-M** | 224 | **2.3** | 475 | **4889** | **51** | **75.3** | **92.3** |

Latency on [Nvidia Jetson Nano](nano_latency.log) and [Arm Cortex-A57](arm_latency.log).
| Model | Input | Params (M) | Top-1 (%) | Nano (ms) | Cortex-A57 (ms) |
|:--:|:--:|:--:|:--:|:--:|:--:|
| MobileViT-XXS | 256<sup>2</sup> | 1.3 | 69.0 | 30.1 | 467.9 |
| EdgeNeXt-XXS | 256<sup>2</sup> | 1.3 | 71.2 | 33.5 | 242.8 |
| EMO-1M | 224<sup>2</sup> | 1.3 | 71.5 | 26.2 | 526.4 |
| **TinyNeXt-S** | 224<sup>2</sup> | 1.3 | **72.7** | **14.3** | **227.7** |
| MobileViT-XS | 256<sup>2</sup> | 2.3 | 74.7 | 50.8 | 859.1 |
| EdgeNeXt-XS | 256<sup>2</sup> | 2.3 | 75.0 | 50.8 | 442.4 |
| EMO-2M | 224<sup>2</sup> | 2.3 | 75.1 | 35.7 | 759.2 |
| **TinyNeXt-M** | 224<sup>2</sup> | 2.3 | **75.3** | **19.4** | **311.6** |


### Test on GPU/CPU

```
python benchmark_gpu_cpu.py
```

### Test on Nvidia Jetson Nano

```
python benchmark_jetson.py
```

