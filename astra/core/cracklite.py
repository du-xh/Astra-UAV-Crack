from __future__ import annotations
# Final Astra dependency. Third-party attribution: THIRD_PARTY_NOTICES.md.
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for BCHW feature maps."""

    def __init__(self, num_channels: int, eps: float=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias

class ConvBNAct(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, stride: int=1, padding: int | None=None, groups: int=1, act: bool=True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False), nn.BatchNorm2d(out_channels)]
        if act:
            layers.append(nn.GELU())
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class DirectionalDepthwiseConv(nn.Module):
    """Depthwise 1D strip convolution sampled along a fixed 2D direction."""

    def __init__(self, channels: int, length: int, degree: int):
        super().__init__()
        if length % 2 == 0:
            raise ValueError('Directional strip length must be odd.')
        self.length = length
        self.degree = degree % 180
        self.weight = nn.Parameter(torch.empty(channels, length))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.normal_(self.weight, mean=0.0, std=(2.0 / length) ** 0.5)

    @staticmethod
    def _offset_for_direction(offset: int, degree: int) -> tuple[int, int]:
        if degree == 0:
            return (0, offset)
        if degree == 90:
            return (offset, 0)
        if degree == 45:
            return (offset, offset)
        if degree == 135:
            return (offset, -offset)
        raise ValueError(f'Unsupported direction: {degree}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.length // 2
        _, channels, height, width = x.shape
        padded = F.pad(x, (pad, pad, pad, pad))
        out = x.new_zeros(x.shape)
        for idx, offset in enumerate(range(-pad, pad + 1)):
            dy, dx = self._offset_for_direction(offset, self.degree)
            y0 = pad + dy
            x0 = pad + dx
            shifted = padded[:, :, y0:y0 + height, x0:x0 + width]
            out = out + shifted * self.weight[:, idx].view(1, channels, 1, 1)
        return out + self.bias.view(1, channels, 1, 1)

class DirectionGuidedTopologyAggregation(nn.Module):
    """
    DGTA from the manuscript:
    orientation prior + tangent/normal directional strip aggregation + fallback.
    """
    DIRECTIONS = (0, 45, 90, 135)

    def __init__(self, dim: int, reduction: int=2, tangent_length: int=15, normal_length: int=5):
        super().__init__()
        reduced_dim = max(dim // reduction, 16)
        self.reduced_dim = reduced_dim
        self.reduce = ConvBNAct(dim, reduced_dim, kernel_size=1, padding=0)
        self.orientation_head = nn.Sequential(ConvBNAct(reduced_dim, reduced_dim, kernel_size=3), nn.Conv2d(reduced_dim, len(self.DIRECTIONS), kernel_size=1))
        self.confidence_head = nn.Sequential(ConvBNAct(reduced_dim, reduced_dim, kernel_size=3), nn.Conv2d(reduced_dim, 1, kernel_size=1))
        self.tangent_convs = nn.ModuleList([DirectionalDepthwiseConv(reduced_dim, tangent_length, degree) for degree in self.DIRECTIONS])
        self.normal_convs = nn.ModuleList([DirectionalDepthwiseConv(reduced_dim, normal_length, degree + 90) for degree in self.DIRECTIONS])
        self.fallback = ConvBNAct(reduced_dim, reduced_dim, kernel_size=3, groups=reduced_dim)
        self.out_proj = ConvBNAct(reduced_dim, dim, kernel_size=1, padding=0, act=False)

    def forward(self, x: torch.Tensor, return_maps: bool=False) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict]:
        u = self.reduce(x)
        orientation = torch.softmax(self.orientation_head(u), dim=1)
        confidence = torch.sigmoid(self.confidence_head(u))
        tangent = 0.0
        normal = 0.0
        for idx, (tangent_conv, normal_conv) in enumerate(zip(self.tangent_convs, self.normal_convs)):
            weight = orientation[:, idx:idx + 1]
            tangent = tangent + weight * tangent_conv(u)
            normal = normal + weight * normal_conv(u)
        fallback = self.fallback(u)
        fused = confidence * (tangent + normal) + (1.0 - confidence) * fallback
        out = self.out_proj(fused)
        if not return_maps:
            return (out, normal)
        maps = {'dgta_orientation': orientation, 'dgta_confidence': confidence, 'dgta_tangent': tangent, 'dgta_normal': normal, 'dgta_fallback': fallback}
        return (out, normal, maps)
