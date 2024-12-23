import warnings
from thop import profile


def check_flops(net, *args):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This API is being deprecated")
        flops, params = profile(net, inputs=(*args,), verbose=False)
    print(f"FLOPs={flops / 1e9}, params={params/1e6:.2f}")
    return flops, params
