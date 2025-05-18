import torch
import torch.nn as nn

__all__ = ['SpdBlock']
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation
 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
 
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))
 
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
    
class SpdBlock(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, c1, c2, block_size=2):
        super().__init__()
        self.block_size = block_size  # 块大小
        self.c1 = c1
        self.c3 = c1 * (block_size ** 2)  # space-to-depth 后的通道数
        self.cv1 = Conv(self.c3, c2, 3, 1)

    def forward(self, x):
        bs = self.block_size
        N, C, H, W = x.size()  # 输入张量的维度
        assert H % bs == 0 and W % bs == 0, \
            f"空间维度必须能被 block_size 整除。得到的 H={H}, W={W}, block_size={bs}"
        a = torch.cat([x[..., ::bs, ::bs], x[..., 1::bs, ::bs], x[..., ::bs, 1::bs], x[..., 1::bs, 1::bs]], 1)
        a = self.cv1(a)
        return a