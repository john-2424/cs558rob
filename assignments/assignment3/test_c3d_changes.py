from Model.AE.CAE import Encoder
import torch

enc2d = Encoder(2800)
x2d = torch.randn(4, 2800)
print(enc2d(x2d).shape)

enc3d = Encoder(6000)
x3d = torch.randn(4, 6000)
print(enc3d(x3d).shape)