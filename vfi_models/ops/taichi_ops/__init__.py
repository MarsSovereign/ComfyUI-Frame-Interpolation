import torch
import sys
import torch.multiprocessing as mp
from .worker_process import f
from .utils import to_shared_memory
# Ops not ported to taichi yet; exported so the shared import in ops/__init__.py works,
# and they raise NotImplementedError only if a model actually calls them
from .adacof import FunctionAdaCoF
from .correlation import ModuleCorrelation, FunctionCorrelation, _FunctionCorrelation
from .batch_edt import batch_edt

parent_conn, child_conn, process = None, None, None
def _device():
    # Imported lazily: the spawned worker imports this package too, and must not initialise
    # ComfyUI's model management (it would probe the GPU without ComfyUI's launch arguments)
    import comfy.model_management as model_management
    return model_management.get_torch_device()

def req_to_taichi_process(op_name, *tensors):
    global parent_conn, child_conn, process
    if parent_conn is None:
        # A private spawn context, so ComfyUI's global multiprocessing start method is left alone
        ctx = mp.get_context('spawn')
        parent_conn, child_conn = ctx.Pipe()
        process = ctx.Process(target=f, args=(child_conn, _device()), daemon=True)
        # spawn normally re-runs the parent's __main__ (ComfyUI's main.py) in the child, which
        # isn't import-safe there; the worker only needs its own module, so hide main's path
        main_module = sys.modules["__main__"]
        main_file, main_spec = getattr(main_module, "__file__", None), getattr(main_module, "__spec__", None)
        try:
            if main_file is not None:
                del main_module.__file__
            main_module.__spec__ = None
            process.start()
        finally:
            if main_file is not None:
                main_module.__file__ = main_file
            main_module.__spec__ = main_spec
    
    tensors = to_shared_memory(tensors)
    parent_conn.send((op_name, tensors))
    # Don't block forever if the worker died (e.g. taichi failed to start); restart it on the next call
    while not parent_conn.poll(1):
        if not process.is_alive():
            exitcode = process.exitcode
            parent_conn, child_conn, process = None, None, None
            raise RuntimeError(f"Taichi ops worker exited unexpectedly (exit code {exitcode}), see the console log for its error")
    result = parent_conn.recv()
    del tensors
    
    if type(result) not in [tuple, list]:
        raise Exception(result)
        
    return [tensor.to(_device()) for tensor in result]

def softsplat(
    tenIn: torch.Tensor, tenFlow: torch.Tensor, tenMetric: torch.Tensor, strMode: str
):
    assert strMode.split("-")[0] in ["sum", "avg", "linear", "soft"]

    if strMode == "sum":
        assert tenMetric is None
    if strMode == "avg":
        assert tenMetric is None
    if strMode.split("-")[0] == "linear":
        assert tenMetric is not None
    if strMode.split("-")[0] == "soft":
        assert tenMetric is not None

    if strMode == "avg":
        tenIn = torch.cat(
            [
                tenIn,
                tenIn.new_ones([tenIn.shape[0], 1, tenIn.shape[2], tenIn.shape[3]]),
            ],
            1,
        )

    elif strMode.split("-")[0] == "linear":
        tenIn = torch.cat([tenIn * tenMetric, tenMetric], 1)

    elif strMode.split("-")[0] == "soft":
        tenIn = torch.cat([tenIn * tenMetric.exp(), tenMetric.exp()], 1)

    # end

    tenOut = req_to_taichi_process("softsplat_out", tenIn, tenFlow)[0]

    if strMode.split("-")[0] in ["avg", "linear", "soft"]:
        tenNormalize = tenOut[:, -1:, :, :]

        if len(strMode.split("-")) == 1:
            tenNormalize = tenNormalize + 0.0000001

        elif strMode.split("-")[1] == "addeps":
            tenNormalize = tenNormalize + 0.0000001

        elif strMode.split("-")[1] == "zeroeps":
            tenNormalize[tenNormalize == 0.0] = 1.0

        elif strMode.split("-")[1] == "clipeps":
            tenNormalize = tenNormalize.clip(0.0000001, None)

        # end

        tenOut = tenOut[:, :-1, :, :] / tenNormalize
    # end

    return tenOut

def FunctionSoftsplat(tenInput, tenFlow, tenMetric, strType):
    assert tenMetric is None or tenMetric.shape[1] == 1
    assert strType in ["summation", "average", "linear", "softmax"]

    if strType == "average":
        tenInput = torch.cat(
            [
                tenInput,
                tenInput.new_ones(
                    tenInput.shape[0], 1, tenInput.shape[2], tenInput.shape[3]
                ),
            ],
            1,
        )

    elif strType == "linear":
        tenInput = torch.cat([tenInput * tenMetric, tenMetric], 1)

    elif strType == "softmax":
        tenInput = torch.cat([tenInput * tenMetric.exp(), tenMetric.exp()], 1)

    # end

    tenOutput = req_to_taichi_process("softsplat_out", tenInput, tenFlow)[0]

    if strType != "summation":
        tenNormalize = tenOutput[:, -1:, :, :]

        tenNormalize[tenNormalize == 0.0] = 1.0

        tenOutput = tenOutput[:, :-1, :, :] / tenNormalize
    # end

    return tenOutput


# end


class ModuleSoftsplat(torch.nn.Module):
    def __init__(self, strType):
        super().__init__()

        self.strType = strType

    # end

    def forward(self, tenInput, tenFlow, tenMetric):
        return FunctionSoftsplat(tenInput, tenFlow, tenMetric, self.strType)

def softsplat_func(tenIn, tenFlow):
    return req_to_taichi_process("softsplat_out", tenIn, tenFlow)[0]

class costvol_func:
    @staticmethod
    def apply(tenOne, tenTwo):
        return req_to_taichi_process("costvol_out", tenOne, tenTwo)[0]

class sepconv_func:
    @staticmethod
    def apply(tenIn, tenVer, tenHor):
        return req_to_taichi_process("sepconv_out", tenIn, tenVer, tenHor)[0]

def init():
    one_sample = torch.ones(1, 3, 16, 16, dtype=torch.float32, device=_device())
    softsplat_func(one_sample, one_sample)
    costvol_func.apply(one_sample, one_sample)
    sepconv_func.apply(one_sample, one_sample, one_sample)
