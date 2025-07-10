import os, torch, time
from fvcore.nn import FlopCountAnalysis
from timm.models import create_model
from models import *
from tabulate import tabulate

torch.autograd.set_grad_enabled(False)
T0, T1 = 10, 60

def compute_throughput(model, device, batch_size, resolution=224):
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    start = time.time()
    with torch.cuda.amp.autocast() if device == 'cuda:0' else torch.no_grad():
        while time.time() - start < T0: model(inputs)
    timing = []
    while sum(timing) < T1:
        start = time.time()
        model(inputs)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        timing.append(time.time() - start)
    throughput = batch_size / torch.as_tensor(timing, dtype=torch.float32).mean().item()
    return throughput

def benchmark_throughput(test_list, device):
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("No CUDA available")
        return []
    if device == 'cpu':
        os.system('echo -n "nb processors "; '
                  'cat /proc/cpuinfo | grep ^processor | wc -l; '
                  'cat /proc/cpuinfo | grep ^"model name" | tail -1')
        print('Using 1 CPU thread'); torch.set_num_threads(1)
    else:
        print(torch.cuda.get_device_name(torch.cuda.current_device()))

    results = []
    for n, batch_size0, resolution in test_list:
        try:
            batch_size = 16 if device == 'cpu' else batch_size0
            inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
            model = create_model(n, num_classes=1000).to(device).eval()
            model = torch.jit.trace(model, inputs)
            throughput = compute_throughput(model, device, batch_size, resolution)
            results.append({'model': n, 'batch_size': batch_size, 'resolution': resolution, 'throughput': throughput})
        except Exception as e:
            print(f"Error processing {n} on {device}: {e}")
    return results

def print_throughput_results(results, device):
    if not results: return
    table_data = [[r['model'], r['batch_size'], r['resolution'], f"{r['throughput']:.2f}"] for r in results]
    headers = ["Model", "Batch Size", "Resolution", "Throughput (images/s)"]
    print(f"\n--- Results for {device.upper()} ---")
    print(tabulate(table_data, headers=headers, tablefmt="grid", numalign="right", stralign="left"))

def get_params(model):
    total = sum([param.nelement() for param in model.parameters()])
    return total / 1e6

def get_flops(model, inputs):
    flops = FlopCountAnalysis(model, inputs)
    return flops.total() / 1e6

def benchmark_flops(model_list):
    results = []
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    for n, _, resolution in model_list:
        try:
            inputs = torch.randn(1, 3, resolution, resolution, device=device)
            model = create_model(n, num_classes=1000)
            model.to(device)
            model.eval()
            params = get_params(model)
            flops = get_flops(model, inputs)
            results.append({ 'model': n, 'params': params, 'flops': flops})
        except Exception as e:
            print(f"Error processing {n}: {e}")

    return results

def print_benchmark_results(results):
    table_data = [[result['model'], f"{result['params']:.2f}", f"{result['flops']:.2f}"] for result in results]
    headers = ["Model", "Params (M)", "FLOPs (M)"]
    print(tabulate(table_data, headers=headers, tablefmt="grid", numalign="right", stralign="left"))

if __name__ == "__main__":
    benchmark_list = [
        ('edgevit_xxs', 512, 224),
        ('pvt_v2_b0', 512, 224),
        ('EfficientViT_M2', 512, 224),
        ('shufflenetv2_100', 512, 224),
        ('shufflenetv2_150', 512, 224),
        ('mobilenet_v1', 512, 224),
        ('mobilenet_v2', 512, 224),
        ('mobilenet_v2_1p4', 512, 224),
        ('mobilenetv3_large_100', 512, 224),
        ('mobileone_s0', 512, 224),
        ('emo_1m', 512, 224),
        ('emo_2m', 512, 224),
        ('emo_5m', 512, 224),
        ('mobilevitv2_050', 512, 256),
        ('mobilevit_xxs', 512, 256),
        ('mobilevit_xs', 512, 256),
        ('mobilevit_s', 512, 256),
        ('edgenext_xx_small', 512, 256),
        ('edgenext_x_small', 512, 256),
        ('edgenext_small', 512, 256),
        ('tinynext_t', 512, 224),
        ('tinynext_s', 512, 224),
        ('tinynext_m', 512, 224),
    ]
    flops_results = benchmark_flops(benchmark_list)
    print_benchmark_results(flops_results)
    for device in ['cuda:1', 'cpu']:
        throughput_results = benchmark_throughput(benchmark_list, device)
        print_throughput_results(throughput_results, device)