import torch
from torch.autograd import Variable
import numpy as np
from torch.nn import Conv2d, Conv1d, ConvTranspose2d

# F =  [[-1, -1, 1], [1, -1, -1], [-1, 1, -1]]
# G =  [[1, 1], [-1, -1]]
# # F = torch.tensor ([[-1, -1, 1], [1, -1, -1], [-1, 1, -1]])
# # G = torch.tensor ([[1, 1], [-1, -1]]
# # )
# C = np.convolve2D(F,G)

input = torch.randn(1, 2, 1, 2)
print(input)

m = ConvTranspose2d(2, 2, (2,2), stride=1, padding=0 )
#m1 = Conv2d(in_channels = 2, out_channels= 2, kernel_size=(2,2), stride=1, padding=0)

#conv2d.weight.data
print(m(input))
