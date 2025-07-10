from models.tinynext import tinynext_t, tinynext_s, tinynext_m


if __name__ == "__main__":
    import torch
    import time


    def print_params(model):
        total = sum([param.nelement() for param in model.parameters()])
        print('Number of params: %.2fM' % (total / 1e6))


    def func_calflops(model, shape):
        from calflops import calculate_flops
        calculate_flops(model, input_shape=shape)


    def func_profile(model, inputs):
        from thop import profile
        flops, params = profile(model, inputs=(inputs,))
        print('flops:\t{:.3f} M'.format(flops / 1e6))
        print('params:\t{:.3f} M'.format(params / 1e6))


    def func_torchsummary(model, shape):
        from torchsummary import summary
        summary(model, shape[1:])


    def func_torchinfo(model, shape):
        from torchinfo import summary
        summary(model, input_size=shape)


    def func_fvcore(model, inputs):
        from fvcore.nn import FlopCountAnalysis, flop_count_table
        flops = FlopCountAnalysis(model, inputs)
        print(flop_count_table(flops, max_depth=3))
        print('flops:\t{:.3f} M'.format(flops.total() / 1e6))


    def func_ptflops(model, shape):
        from ptflops import get_model_complexity_info
        macs, params = get_model_complexity_info(model, shape[1:])
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


    @torch.no_grad()
    def throughput(model, batch_size=128, resolution=224):
        torch.autograd.set_grad_enabled(False)
        T0 = 5
        T1 = 10 * 6  # forward min time
        inputs = torch.randn(batch_size, 3, resolution, resolution).cuda()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        start = time.time()
        while time.time() - start < T0:
            model(inputs)
        timing = []
        torch.cuda.synchronize()
        while sum(timing) < T1:
            start = time.time()
            model(inputs)
            torch.cuda.synchronize()
            timing.append(time.time() - start)
        timing = torch.as_tensor(timing, dtype=torch.float32)
        print(batch_size / timing.mean().item(), 'images/s @ batch size', batch_size)

    resolution = 224
    model = tinynext_t(num_classes=1000, distillation=False).cuda()
    model.eval()
    shape = (1, 3, resolution, resolution)
    inputs = torch.randn(size=shape).cuda()
    res = model(inputs)
    func_fvcore(model, inputs)
    throughput(model=model, resolution=resolution)
    # func_calflops(model, shape)
    # func_profile(model, inputs)
    # func_torchsummary(model, shape)
    # func_torchinfo(model, shape)
    # func_ptflops(model, shape)
