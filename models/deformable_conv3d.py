from torch.autograd import Variable, Function
import torch
from torch import nn
import numpy as np

class DeformConv3DWrapper(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilation=1, groups=1, offset_groups=1):
        super().__init__()
        offset_channels = 3 * kernel_size * kernel_size * kernel_size
        self.conv3d_offset = nn.Conv3d(
            in_channels,
            offset_channels * offset_groups,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
        )
        self.conv3d = DeformConv3D(
            in_channels, out_channels, 
            kernel_size=kernel_size, padding=dilation, bias=False)

    def forward(self, x):
        offset = self.conv3d_offset(x)
        return self.conv3d(x, offset)

class DeformConv3D(nn.Module):
    def __init__(self, inc, outc=[], kernel_size=3, padding=1, bias=None):
        super(DeformConv3D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        #self.zero_padding = nn.functional.pad(padding)
        self.conv_kernel = nn.Conv3d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

    def forward(self, x, offset):
        dtype = offset.data.type()
        ks = self.kernel_size

        # Out_channels = 3 * kernel_size_x * kernel_size_y * kernel_size_z
        M = offset.size(1)
        N = M // 3 # Number of channels

        if self.padding != 0:
            # For simplicity we pad from both sides in all 3 dimensions
            padding_use = (self.padding, self.padding, self.padding, self.padding ,self.padding, self.padding)
            x = nn.functional.pad(x, padding_use, "constant", 0)

        # Get input dimensions
        b, c, h, w, d = x.size()
        shape = (h, w, d)

        # interpolation points p (Eq. 2)
        # ------------------------------
        # (b, 3N, h, w, d)
        p = self._get_p(offset, dtype)

        # (b, h, w, d, 3N)
        p = p.contiguous().permute(0, 2, 3, 4, 1) # p = p_0 + p_n + offset

        # Use grid_sample to interpolate
        # ------------------------------
        for ii in range(N):
            # Normalize flow field to rake values in the range [-1, 1]
            flow = p[..., [t for t in range(ii, M, N)]]
            for jj in range(3):
                flow[..., jj] = 2 * flow[..., jj] / (shape[jj] - 1) - 1

            # Push through the spatial transformer
            tmp = nn.functional.grid_sample(input=x, grid=flow, mode='bilinear', padding_mode='border', align_corners=True).contiguous()
            tmp = tmp.unsqueeze(dim=-1)

            # Aggregate
            if ii == 0:
                xt = tmp
            else:
                xt = torch.cat((xt, tmp), dim=-1)

        # For simplicity, ks is a scalar, implying kernel has same dimensions in all directions
        x_offset = self._reshape_x_offset(xt, ks)
        out = self.conv_kernel(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y, p_n_z = np.meshgrid(range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
                                          range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
                                          range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
                                          indexing='ij')
        # (2N, 1)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten(), p_n_z.flatten()))
        p_n = np.reshape(p_n, (1, 3*N, 1, 1, 1))
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)

        return p_n

    @staticmethod
    def _get_p_0(h, w, d, N, dtype):
        #p_0_x, p_0_y, p_0_z = np.meshgrid(range(1, h + 1), range(1, w + 1), range(1, d + 1), indexing='ij') # 1,...,N
        p_0_x, p_0_y, p_0_z = np.meshgrid(range(0, h), range(0, w), range(0, d), indexing='ij') # 0,...N-1
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0_z = p_0_z.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y, p_0_z), axis=1)
        p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w, d = offset.size(1)//3, offset.size(2), offset.size(3), offset.size(4)

        # (1, 3N, 1, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 3N, h, w, d)
        p_0 = self._get_p_0(h, w, d, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, d, _ = q.size()
        ny = x.size(3) # Padded dimension y
        nz = x.size(4)  # Padded dimension z
        c = x.size(1) # Number of channels in input x
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, d, N)
        offset_x = q[..., :N]
        offset_y = q[..., N:2*N]
        offset_z = q[..., 2*N:]
        # Convert subscripts to linear indices (i.e. Matlab's sub2ind)
        index = offset_x * ny * nz + offset_y * nz + offset_z
        # (b, c, h*w*d*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, d, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        '''
        This function arranges the interpolated x values in consecutive 3d blocks of size
        kernel_size x kernel_size x kernel_size. Since the Conv3d stride is equal to kernel_size, the convolution
        will happen only for the offset cubes and output the results in the proper locations
        Note: We assume kernel size is the same for all dimensions (cube)
        '''
        b, c, h, w, d, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks*ks].contiguous().view(b, c, h, w, d*ks*ks) for s in range(0, N, ks*ks)], dim=-1)
        N = x_offset.size(4)
        x_offset = torch.cat([x_offset[..., s:s + d*ks*ks].contiguous().view(b, c, h, w*ks, d*ks) for s in range(0, N, d*ks*ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks, d*ks)

        return x_offset


if __name__ == "__main__":
    model = DeformConv3DWrapper(10, 10, kernel_size=3)
    x = torch.randn(1,10,16,8,8)
    print(model(x).shape)
