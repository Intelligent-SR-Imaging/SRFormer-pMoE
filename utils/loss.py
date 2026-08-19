import torch

class LPIPSLoss(torch.nn.Module):
    def __init__(self,loss_weight=1.0,xz=True,yz=True):
        super(LPIPSLoss, self).__init__()
        import pyiqa
        self.model = pyiqa.create_metric('lpips-vgg', as_loss=True)
        self.loss_weight = loss_weight
        self.xz = xz
        self.yz = yz

    def forward(self, x, y):
        b,c,nx,ny = x.shape
        x2d = x.contiguous().view(b*c, 1, nx, ny).contiguous()
        y2d = y.contiguous().view(b*c, 1, nx, ny).contiguous()
        loss = self.model(x2d, y2d)
        if self.xz:
            loss += self.model(x.transpose(1,3).contiguous().view(b*ny, 1, c, nx),
                               y.transpose(1,3).contiguous().view(b*ny, 1, c, nx))
        if self.yz:
            loss += self.model(x.transpose(1,2).contiguous().view(b*nx, 1, c, ny),
                               y.transpose(1,2).contiguous().view(b*nx, 1, c, ny))

        return loss*self.loss_weight


def PSNR(re_img, tar_img):
    imdff = torch.clamp(re_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    rmse = (imdff ** 2).mean().sqrt()
    ps = 20 * torch.log10(1 / rmse)
    return ps


def linear_transform(transform, target, device):
    size = transform.shape
    length = size[0] * size[1] * size[2]
    x = transform.reshape((length, 1))
    y = target.reshape((length, 1))
    X = torch.cat([x, torch.ones(length, 1).to(device)], 1)
    XT = X.permute(1, 0)
    # c = torch.solve(XT.mm(y), XT.mm(X))[0]
    if y.dtype == torch.float16:
        y=y.float()
    c = torch.linalg.solve(XT.mm(X), XT.mm(y))
    re = (c[0] * x + c[1]).reshape(size)
    return re


def batch_PSNR_linear_transform(restored, target, device):
    PSNR_list = []
    for re, t in zip(restored, target):
        if torch.max(re) == torch.min(re):
            print("zero in val restore")
        else:
            re = linear_transform(re, t, device)
        psnr = PSNR(re, t)
        PSNR_list.append(psnr)
    return sum(PSNR_list)
