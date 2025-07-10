import os, torch, time
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


def print_benchmark_results(results):
    table_data = [[result['model'], f"{result['params']:.2f}", f"{result['flops']:.2f}"] for result in results]
    headers = ["Model", "Params (M)", "FLOPs (M)"]
    print(tabulate(table_data, headers=headers, tablefmt="grid", numalign="right", stralign="left"))

if __name__ == "__main__":
    benchmark_list = [
        ('tinynext_t', 512, 224),
        ('tinynext_s', 512, 224),
        ('tinynext_m', 512, 224),
        ('emo_1m', 512, 224),
        ('emo_2m', 512, 224),
        # ('emo_5m', 512, 224),
        ('mobilevit_xxs', 512, 256),
        ('mobilevit_xs', 256, 256),
        # ('mobilevit_s', 256, 256),
        ('edgenext_xx_small', 512, 256),
        ('edgenext_x_small', 512, 256),
        # ('edgenext_small', 512, 256),
    ]
    throughput_results = benchmark_throughput(benchmark_list, 'cuda')
    print_throughput_results(throughput_results, 'cuda')
    throughput_results = benchmark_throughput(benchmark_list, 'cpu')
    print_throughput_results(throughput_results, 'cpu')