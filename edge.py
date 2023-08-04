import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open('./img/53_pred.png')
np_img = np.asarray(img)
img_tensor = torch.tensor(np_img).unsqueeze(dim=0).unsqueeze(dim=0).float()


kernel = torch.ones(1, 1, 3, 3)

a = torch.nn.functional.conv2d(img_tensor.float(), kernel, padding=0)
padded_a = torch.nn.functional.pad(a, (1, 1, 1, 1), mode='constant', value=0)

# 取出边缘点
edge_mask = padded_a == 9 * img_tensor
print(edge_mask)

maps = torch.where(edge_mask, torch.zeros_like(img_tensor), torch.ones_like(img_tensor))
maps[:, :, 1, :] = 0  # 第一行
maps[:, :, -2, :] = 0  # 最后一行
maps[:, :, :, 1] = 0  # 第一列
maps[:, :, :, -2] = 0  # 最后一列
img1 = maps.permute(0, 2, 3, 1).squeeze(dim=0)
img1 = img1.numpy()
plt.imshow(img1.astype('uint8'))
plt.show()