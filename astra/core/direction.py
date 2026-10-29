# Final Astra dependency. Third-party attribution: THIRD_PARTY_NOTICES.md.
import torch
from torch import nn
import torch.nn.functional as F
from astra.core.cracklite import LayerNorm2d, ConvBNAct, DirectionalDepthwiseConv, DirectionGuidedTopologyAggregation

class PackedDirectionalConv(DirectionalDepthwiseConv):
    """Exact fixed-direction sum as one depthwise convolution; no CUDA extension."""

    def __init__(self, channels, length, degree):
        super().__init__(channels, length, degree)
        if self.degree not in (0, 45, 90, 135):
            raise ValueError('Unsupported fixed direction')
        coordinate = torch.arange(length, dtype=torch.long)
        diagonal = coordinate * length + (coordinate if self.degree == 45 else coordinate.flip(0))
        self.register_buffer('diagonal_indices', diagonal[None].expand(channels, -1), persistent=False)

    def forward(self, x):
        pad = self.length // 2
        if self.degree == 0:
            kernel = self.weight[:, None, None, :]
            padding = (0, pad)
        elif self.degree == 90:
            kernel = self.weight[:, None, :, None]
            padding = (pad, 0)
        else:
            flat = self.weight.new_zeros((self.weight.shape[0], self.length * self.length))
            kernel = flat.scatter(1, self.diagonal_indices, self.weight).view(-1, 1, self.length, self.length)
            padding = (pad, pad)
        return F.conv2d(x, kernel, bias=self.bias, padding=padding, groups=x.shape[1])

def pack_directional_convolutions(module):
    for name, child in list(module.named_children()):
        if type(child) is DirectionalDepthwiseConv:
            replacement = PackedDirectionalConv(child.weight.shape[0], child.length, child.degree)
            replacement.weight = child.weight
            replacement.bias = child.bias
            replacement = replacement.to(device=child.weight.device, dtype=child.weight.dtype)
            setattr(module, name, replacement)
        else:
            pack_directional_convolutions(child)

class SemanticGuidedAggregation(DirectionGuidedTopologyAggregation):
    """Only change the source of orientation/confidence; keep author value paths."""

    def __init__(self, dim=32):
        super().__init__(dim=dim, reduction=2, tangent_length=15, normal_length=5)
        self.semantic_norm = LayerNorm2d(dim)
        self.semantic_reduce = ConvBNAct(dim, self.reduced_dim, kernel_size=1, padding=0)

    def forward(self, x, semantic):
        values = self.reduce(x)
        with torch.cuda.amp.autocast(enabled=False):
            normalized_semantic = self.semantic_norm(semantic.float())
        guidance = self.semantic_reduce(normalized_semantic)
        orientation = torch.softmax(self.orientation_head(guidance), dim=1)
        confidence = torch.sigmoid(self.confidence_head(guidance))
        tangent = 0.0
        normal = 0.0
        for index, (tangent_conv, normal_conv) in enumerate(zip(self.tangent_convs, self.normal_convs)):
            weight = orientation[:, index:index + 1]
            tangent = tangent + weight * tangent_conv(values)
            normal = normal + weight * normal_conv(values)
        fused = confidence * (tangent + normal) + (1.0 - confidence) * self.fallback(values)
        return (self.out_proj(fused), normal)

class DirectionContextResidual(nn.Module):

    def __init__(self, semantic_guidance=False):
        super().__init__()
        self.semantic_guidance = semantic_guidance
        self.norm = LayerNorm2d(32)
        self.aggregate = SemanticGuidedAggregation(32) if semantic_guidance else DirectionGuidedTopologyAggregation(32, reduction=2, tangent_length=15, normal_length=5)
        pack_directional_convolutions(self.aggregate)
        self.scale = nn.Parameter(torch.full((1, 32, 1, 1), 0.01))

    def forward(self, detail, semantic):
        with torch.cuda.amp.autocast(enabled=False):
            normalized = self.norm(detail.float())
        if self.semantic_guidance:
            context, _ = self.aggregate(normalized, semantic)
        else:
            context, _ = self.aggregate(normalized)
        return detail + self.scale.to(context.dtype) * context
