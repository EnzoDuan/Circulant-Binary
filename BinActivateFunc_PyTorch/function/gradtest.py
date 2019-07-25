import torch
from BinActivateFunc import BinActivateFunc, BinActivateFunc_bireal

test_size = 16

def test_bin():
    cuda0 = torch.device('cuda:0')
    A = BinActivateFunc.apply
    input = torch.randn(test_size, test_size, test_size, requires_grad=True, device=cuda0)
    print(input)
    output = A(input)
    print(output)
    grad_output = torch.randn(test_size, test_size, test_size).cuda()
    print(grad_output)
    output.backward(grad_output)
    print(input.grad)
    # manully checking...
    grad_output[input > 1] = 0
    grad_output[input < -1] = 0
    print(torch.equal(grad_output, input.grad))

def test_bireal():
    cuda0 = torch.device('cuda:0')
    A = BinActivateFunc_bireal.apply
    input = torch.randn(test_size, test_size, test_size, requires_grad=True, device=cuda0)
    print(input)
    output = A(input)
    print(output)
    grad_output = torch.randn(test_size, test_size, test_size).cuda()
    print(grad_output)
    output.backward(grad_output)
    print(input.grad)
    # manully checking...
    for i in range(grad_output.size(0)):
        for j in range(grad_output.size(1)):
            for k in range(grad_output.size(2)):
                x = input[i,j,k]
                if x > 1 or x < -1:
                    grad_output[i,j,k] = 0
                elif x < 0:
                    grad_output[i,j,k] *= 2 + 2 * x
                else:
                    grad_output[i,j,k] *= 2 - 2 * x
    print(torch.equal(grad_output, input.grad))


if __name__ == '__main__':
    # torch.manual_seed(618)
    test_bin()
    test_bireal()