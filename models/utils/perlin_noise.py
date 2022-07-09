# source: Alexandre Défossez, https://gist.github.com/adefossez/0646dbe9ed4005480a2407c62aac8869

import torch

def interp(t):
    return 3 * t**2 - 2 * t ** 3

def perlin(n, width, height, scale=10, device=None):
    gx, gy = torch.randn((2, n, width + 1, height + 1, 1, 1), device=device)
    xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
    ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)

    wx = 1 - interp(xs)
    wy = 1 - interp(ys)

    dots = 0
    dots += wx * wy * (gx[:, :-1, :-1] * xs + gy[:, :-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[:, 1:, :-1] * (1 - xs) + gy[:, 1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:, :-1, 1:] * xs - gy[:, :-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[:, 1:, 1:] * (1 - xs) - gy[:, 1:, 1:] * (1 - ys))

    return dots.permute(0, 1, 3, 2, 4).contiguous().view(n, width * scale, height * scale)

def perlin_ms(octaves=[1, 1, 1, 1], n=1, width=2, height=2, device=None):
    scale = 2 ** len(octaves)
    out = 0
    for oct in octaves:
        p = perlin(n, width, height, scale, device)
        out += p * oct
        scale //= 2
        width *= 2
        height *= 2
    return out