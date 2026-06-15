# CoderAI - in-process RIFE (IFNet_HDv3) frame interpolation.
# Vendored architecture matching the RIFE flownet.pkl weights (3 student IFBlocks
# of c=90, input 11ch = warped0+warped1+mask+flow; block_tea is training-only and
# unused at inference). Runs entirely in-process on torch — no subprocess.
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                  padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes),
    )


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super().__init__()
        self.conv0 = nn.Sequential(
            _conv(in_planes, c // 2, 3, 2, 1),
            _conv(c // 2, c, 3, 2, 1),
        )
        self.convblock0 = nn.Sequential(_conv(c, c), _conv(c, c))
        self.convblock1 = nn.Sequential(_conv(c, c), _conv(c, c))
        self.convblock2 = nn.Sequential(_conv(c, c), _conv(c, c))
        self.convblock3 = nn.Sequential(_conv(c, c), _conv(c, c))
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 4, 2, 1),
            nn.PReLU(c // 2),
            nn.ConvTranspose2d(c // 2, 4, 4, 2, 1),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 4, 2, 1),
            nn.PReLU(c // 2),
            nn.ConvTranspose2d(c // 2, 1, 4, 2, 1),
        )

    def forward(self, x, flow, scale=1):
        x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear",
                          align_corners=False, recompute_scale_factor=False)
        flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear",
                             align_corners=False, recompute_scale_factor=False) * (1. / scale)
        feat = self.conv0(torch.cat((x, flow), 1))
        feat = self.convblock0(feat) + feat
        feat = self.convblock1(feat) + feat
        feat = self.convblock2(feat) + feat
        feat = self.convblock3(feat) + feat
        flow = self.conv1(feat)
        mask = self.conv2(feat)
        flow = F.interpolate(flow, scale_factor=scale, mode="bilinear",
                             align_corners=False, recompute_scale_factor=False) * scale
        mask = F.interpolate(mask, scale_factor=scale, mode="bilinear",
                             align_corners=False, recompute_scale_factor=False)
        return flow, mask


_backwarp_cache = {}


def _warp(img, flow):
    """Backward-warp img by flow (B,2,H,W) via grid_sample."""
    B, _, H, W = img.shape
    dev = img.device
    key = (B, H, W, dev)
    grid = _backwarp_cache.get(key)
    if grid is None:
        hor = torch.linspace(-1.0, 1.0, W, device=dev).view(1, 1, 1, W).expand(B, -1, H, -1)
        ver = torch.linspace(-1.0, 1.0, H, device=dev).view(1, 1, H, 1).expand(B, -1, -1, W)
        grid = torch.cat([hor, ver], 1)  # (B,2,H,W)
        _backwarp_cache[key] = grid
    flow = torch.cat([flow[:, 0:1] / ((W - 1.0) / 2.0),
                      flow[:, 1:2] / ((H - 1.0) / 2.0)], 1)
    g = (grid + flow).permute(0, 2, 3, 1)
    return F.grid_sample(img, g, mode="bilinear", padding_mode="border",
                         align_corners=True)


class IFNet(nn.Module):
    """RIFE IFNet_HDv3 — predicts bidirectional flow + blend mask and returns the
    t=0.5 interpolated frame. Higher fps multipliers are produced by recursive
    bisection (interpolate the midpoints) by the caller."""

    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(11, c=90)
        self.block1 = IFBlock(11, c=90)
        self.block2 = IFBlock(11, c=90)
        self.block_tea = IFBlock(14, c=90)  # training-only; loaded but unused

    @torch.no_grad()
    def interpolate(self, img0, img1, scale_list=(4, 2, 1)):
        B, _, H, W = img0.shape
        flow = torch.zeros(B, 4, H, W, device=img0.device, dtype=img0.dtype)
        mask = torch.zeros(B, 1, H, W, device=img0.device, dtype=img0.dtype)
        warped0, warped1 = img0, img1
        for i, block in enumerate((self.block0, self.block1, self.block2)):
            fd, md = block(torch.cat((warped0, warped1, mask), 1), flow,
                           scale=scale_list[i])
            flow = flow + fd
            mask = mask + md
            warped0 = _warp(img0, flow[:, :2])
            warped1 = _warp(img1, flow[:, 2:4])
        m = torch.sigmoid(mask)
        return warped0 * m + warped1 * (1 - m)


def load_ifnet(weights_path: str, device):
    """Build IFNet and load RIFE flownet weights (.pkl/.pth, possibly 'module.'-
    prefixed). Returns the eval model on `device`."""
    net = IFNet()
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    clean = {}
    for k, v in sd.items():
        nk = k[7:] if k.startswith("module.") else k
        clean[nk] = v
    net.load_state_dict(clean, strict=True)
    net.eval().to(device)
    return net
