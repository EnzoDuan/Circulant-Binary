#include <torch/torch.h>

#include <vector>

// CUDA forward declarations
int BinActivateFunc_cuda_backward(
    at::Tensor input,
    at::Tensor gradInput
);

int ClipActivateFunc_cuda_backward(
    at::Tensor input,
    at::Tensor gradInput
);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor BinActivateFunc_forward(
    at::Tensor input)
{
    CHECK_INPUT(input);
    return at::sign(input);
}

int BinActivateFunc_backward(
    at::Tensor input,
    at::Tensor gradInput)
{
    CHECK_INPUT(input);
    CHECK_INPUT(gradInput);

    BinActivateFunc_cuda_backward(input, gradInput);
    return 1;
}

int ClipActivateFunc_backward(
    at::Tensor input,
    at::Tensor gradInput)
{
    CHECK_INPUT(input);
    CHECK_INPUT(gradInput);

    ClipActivateFunc_cuda_backward(input, gradInput);
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &BinActivateFunc_forward, "BinActivateFunc forward (CUDA)");
  m.def("backward", &BinActivateFunc_backward, "BinActivateFunc backward (CUDA)");
  m.def("clip_backward", &ClipActivateFunc_backward, "ClipActivateFunc backward (CUDA)");
}
