import torch
from torch.autograd import Function
import BinActivateFunc_cpp, BinActivateFunc_cuda

class BinActivateFunc(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        ctx.backend = BinActivateFunc_cuda if input.is_cuda else BinActivateFunc_cpp
        output = ctx.backend.forward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        ctx.backend.backward(ctx.input, grad_input)
        return grad_input

class BinActivateFunc2(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input[ctx.input > 1] = 0
        grad_input[ctx.input < -1] = 0
        return grad_input

class BinActivateFunc_bireal(Function):
    '''
    Proposed in "Bi-Real Net: Enhancing the Performance of 1-bit CNNs With Improved \
    Representational Capability and Advanced Training Algorithm" (ECCV2018)
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        ctx.backend = BinActivateFunc_cuda if input.is_cuda else BinActivateFunc_cpp
        output = ctx.backend.forward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        ctx.backend.clip_backward(ctx.input, grad_input)
        return grad_input