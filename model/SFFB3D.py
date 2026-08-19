import torch.nn as nn
import torch

def apply_complex(fr, fi, input, dtype = torch.complex64):
    return (fr(input.real)-fi(input.imag)).type(dtype) \
            + 1j*(fr(input.imag)+fi(input.real)).type(dtype)


class ComplexConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv3d, self).__init__()
        self.conv_r = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        return apply_complex(self.conv_r, self.conv_i, input)


class FourierUnit3D(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit3D, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv3d(in_channels=in_channels * 2,
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.relu = torch.nn.LeakyReLU(inplace=True)
        self.fft_norm = fft_norm

    def forward(self, x):
        if x.dtype == torch.float16:
            half_f = True
        else:
            half_f = False
        if half_f:
            x = x.float()
        batch = x.shape[0]
        # (batch, c, d, h, w/2+1, 2)
        fft_dim = (-3, -2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 5, 2, 3, 4).contiguous()  # (batch, c, 2,d, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        if half_f:
            ffted=ffted.half()
        ffted = self.conv_layer(ffted)  # (batch, c*2, d, h, w/2+1)
        ffted = self.relu(ffted)
        if half_f:
            ffted=ffted.float()
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 5, 2).contiguous()  # (batch,c, d, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
        if half_f:
            output = output.half()
        return output

from torch.nn.functional import leaky_relu

def complex_leaky_relu(input):
    return leaky_relu(input.real).type(torch.complex64)+1j*leaky_relu(input.imag).type(torch.complex64)

class FourierUnit3D_complexConv(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit3D_complexConv, self).__init__()
        self.groups = groups

        self.conv_layer = ComplexConv3d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]
        # (batch, c, d, h, w/2+1, 2)
        fft_dim = (-3, -2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = self.conv_layer(ffted)
        ffted = complex_leaky_relu(ffted)
        ifft_shape_slice = x.shape[-3:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        return output

class FourierUnit2D(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit2D, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv3d(in_channels=in_channels * 2,
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.relu = torch.nn.LeakyReLU(inplace=True)
        self.fft_norm = fft_norm

    def forward(self, x):
        if x.dtype == torch.float16:
            half_f = True
        else:
            half_f = False
        if half_f:
            x = x.float()
        batch = x.shape[0]
        # (batch, c, d, h, w/2+1, 2)
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 5, 2, 3, 4).contiguous()  # (batch, c, 2,d, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        if half_f:
            ffted=ffted.half()
        ffted = self.conv_layer(ffted)  # (batch, c*2, d, h, w/2+1)
        ffted = self.relu(ffted)
        if half_f:
            ffted=ffted.float()
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 5, 2).contiguous()  # (batch,c, d, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
        if half_f:
            output = output.half()
        return output

class SFFB2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pre_specturm_feature = nn.Sequential(
            nn.Conv3d(dim,dim,3,1,1),
            nn.LeakyReLU(inplace=True),
        )

        self.specturm_feature = nn.Sequential(
            nn.Conv3d(dim, dim, 3, 1, 1),
            FourierUnit2D(dim, dim)
        )

        self.post_specturm_feature = nn.Conv3d(dim,dim,1,1,0)

        self.res = nn.Sequential(
            nn.Conv3d(dim, dim, 3, 1, 1),
            nn.Conv3d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim, dim, 3, 1, 1),
        )

        self.out_proj = nn.Conv3d(dim * 2, dim, 1, 1, 0)

    def forward(self, x):
        x_fft = self.pre_specturm_feature(x)
        x_fft = self.specturm_feature(x_fft) + x_fft
        x_fft = self.post_specturm_feature(x_fft)

        x_res = self.res(x) + x

        x_out = torch.cat([x_fft, x_res],1)
        x_out = self.out_proj(x_out)

        return x_out

class SFFB3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pre_specturm_feature = nn.Sequential(
            nn.Conv3d(dim,dim,3,1,1),
            nn.LeakyReLU(inplace=True),
        )

        self.specturm_feature = nn.Sequential(
            nn.Conv3d(dim, dim, 3, 1, 1),
            FourierUnit3D(dim, dim)
        )

        self.post_specturm_feature = nn.Conv3d(dim,dim,1,1,0)

        self.res = nn.Sequential(
            nn.Conv3d(dim, dim, 3, 1, 1),
            nn.Conv3d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim, dim, 3, 1, 1),
        )

        self.out_proj = nn.Conv3d(dim * 2, dim, 1, 1, 0)

    def forward(self, x):
        x_fft = self.pre_specturm_feature(x)
        x_fft = self.specturm_feature(x_fft) + x_fft
        x_fft = self.post_specturm_feature(x_fft)

        x_res = self.res(x) + x

        x_out = torch.cat([x_fft, x_res],1)
        x_out = self.out_proj(x_out)

        return x_out

class SFFB3D_complex_conv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pre_specturm_feature = nn.Sequential(
            nn.Conv3d(dim,dim,3,1,1),
            nn.LeakyReLU(inplace=True),
        )

        self.specturm_feature = nn.Sequential(
            nn.Conv3d(dim, dim, 3, 1, 1),
            FourierUnit3D_complexConv(dim,dim)
        )

        self.post_specturm_feature = nn.Conv3d(dim,dim,1,1,0)

        self.res = nn.Sequential(
            nn.Conv3d(dim, dim, 3, 1, 1),
            nn.Conv3d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim, dim, 3, 1, 1),
        )

        self.out_proj = nn.Conv3d(dim * 2, dim, 1, 1, 0)

    def forward(self, x):
        x_fft = self.pre_specturm_feature(x)
        x_fft = self.specturm_feature(x_fft) + x_fft
        x_fft = self.post_specturm_feature(x_fft)

        x_res = self.res(x) + x

        x_out = torch.cat([x_fft, x_res],1)
        x_out = self.out_proj(x_out)

        return x_out

if __name__ == "__main__":
    input = torch.rand(2,4,4,16,16)
    layer = SFFB2D(4)
    print(input.shape)
    print(layer(input).shape)