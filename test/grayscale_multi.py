import pygpubench
import torch


def reference_kernel(data):
    output_gray, output_red, data = data
    weights = torch.tensor([0.2989, 0.5870, 0.1140],
                           device=data.device,
                           dtype=data.dtype)
    output_gray[...] = torch.sum(data * weights, dim=-1)
    output_red[...] = data[..., 0]


def generate_input(size: int, seed: int):
    """
    Generates random RGB image tensor of the specified size.
    Returns:
        Tensor of shape (size, size, 3) with values in [0, 1]
    """
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)

    x = torch.rand(
        size, size, 3, device="cuda", dtype=torch.float32, generator=gen
    ).contiguous()

    y_gray = torch.empty(size, size, device="cuda", dtype=torch.float32).contiguous()
    y_red = torch.empty(size, size, device="cuda", dtype=torch.float32).contiguous()

    return x, y_gray, y_red


def generate_test_case(**kwargs):
    x, y_gray, y_red = generate_input(**kwargs)
    expected_gray = torch.empty_like(y_gray)
    expected_red = torch.empty_like(y_red)
    reference_kernel((expected_gray, expected_red, x))
    return (
        pygpubench.out(y_gray, expected=(expected_gray, 1e-6, 1e-6)),
        pygpubench.out(y_red, expected=expected_red),
        x,
    )
if __name__ == "__main__":
    kernels = ["valid_custom_kernel_eager", "valid_custom_kernel_compiled", "valid_custom_kernel_stream"]
    for kernel in kernels:
        print(kernel)
        res = pygpubench.do_bench_isolated(
            f"submission_multi.{kernel}",
            generate_test_case,
            {"size": 1024},
            100,
            5,
            discard=True,
        )
        print("❌" if res.errors else "✅", pygpubench.basic_stats(res.time_us))

    broken = ["wrong_custom_kernel_backward_race", "wrong_custom_kernel_forward_race"]
    for kernel in broken:
        print(kernel)
        res = pygpubench.do_bench_isolated(
            f"submission_multi.{kernel}",
            generate_test_case,
            {"size": 1024},
            100,
            5,
            discard=True,
        )
        print("❌" if res.errors else "✅", pygpubench.basic_stats(res.time_us))

    print("done")
