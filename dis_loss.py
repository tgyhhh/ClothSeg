import torch
import torch.nn as nn
import numpy as np

torch.set_printoptions(threshold=np.inf)

def distance_to_nearest_nonzero(a, b):
    """
    计算矩阵a不同batch中每个非0元素索引位置到矩阵b中不同batch值为1的元素的索引位置的最近欧几里得距离的均值。
    参数:
        a: torch.Tensor, shape为 (b, c, h, w)。
        b: torch.Tensor, shape为 (b, c, h, w)。b中的元素必须是0或1。
    返回:
        torch.Tensor, shape为 (1,)。返回的张量中的元素表示矩阵a中每个非0元素索引位置到矩阵b中值为1的元素的索引位置的最近欧几里得距离的均值。
    """
    device = a.device
    a_indices = torch.nonzero(a, as_tuple=True)
    b_1_indices = torch.nonzero(b == 1, as_tuple=True)
    if len(a_indices[0]) == 0 or len(b_1_indices[0]) == 0:
        return torch.tensor(0.0)  # 如果a或b中没有非0或值为1的元素，则返回0
    a_coords = torch.stack(a_indices[2:], dim=1).float()  # 获取a中非0元素的坐标，转换为浮点数张量
    b_1_coords = torch.stack(b_1_indices[2:], dim=1).float()  # 获取b中值为1的元素的坐标，转换为浮点数张量
    batch_number_a = a_indices[0]
    batch_number_b = b_1_indices[0]
    unique_batch_number = torch.unique(batch_number_a)
    distances_list = []
    for i in range(len(unique_batch_number)):
        batch_step_a = (batch_number_a == i)
        batch_step_b = (batch_number_b == i)
        a_coords_new = a_coords[batch_step_a]
        b_1_coords_new = b_1_coords[batch_step_b]
        # 计算欧几里得距离
        distances = torch.norm(a_coords_new.unsqueeze(1) - b_1_coords_new.unsqueeze(0), p=2, dim=2)  # 计算所有a_coords和b_1_coords之间的欧几里得距离
        if len(distances) == 0 or distances.numel() == 0:
            nearest_distances = torch.tensor(0.0)
            nearest_distances = nearest_distances.to(device)
        else:
            nearest_distances = torch.min(distances, dim=1).values
            nearest_distances = torch.mean(nearest_distances)
        distances_list.append(nearest_distances)
    return torch.mean(torch.stack(distances_list))

class Label_Distance_Loss(nn.Module):
    def __init__(self):
        super(Label_Distance_Loss, self).__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, label=None):
        prediction = x.argmax(1)  # (2,480,480)
        # print(prediction)

        matrix_tensor = prediction.unsqueeze(dim=1)  # (2,1,480,480)
        _, channel, _, _ = matrix_tensor.shape
        device = x.device
        kernel = torch.ones(1, channel, 3, 3)
        kernel = kernel.to(device)

        a = torch.nn.functional.conv2d(matrix_tensor.float(), kernel, padding=0)
        padded_a = torch.nn.functional.pad(a, (1, 1, 1, 1), mode='constant', value=0)

        # 取出边缘点
        edge_mask = padded_a == 9 * matrix_tensor

        maps = torch.where(edge_mask, torch.zeros_like(matrix_tensor), torch.ones_like(matrix_tensor))
        maps[:, :, 0, :] = 0  # 第一行
        maps[:, :, -1, :] = 0  # 最后一行
        maps[:, :, :, 0] = 0  # 第一列
        maps[:, :, :, -1] = 0  # 最后一列
        maps = maps * matrix_tensor

        label = label.unsqueeze(dim=1)  # (2,1,480,480)
        b = torch.nn.functional.conv2d(label.float(), kernel, padding=0)
        padded_b = torch.nn.functional.pad(b, (1, 1, 1, 1), mode='constant', value=0)
        # 取出边缘点
        edge_mask_b = padded_b == 9 * label

        mapsb = torch.where(edge_mask_b, torch.zeros_like(label), torch.ones_like(label))
        mapsb[:, :, 0, :] = 0  # 第一行
        mapsb[:, :, -1, :] = 0  # 最后一行
        mapsb[:, :, :, 0] = 0  # 第一列
        mapsb[:, :, :, -1] = 0  # 最后一列

        loss_distance = distance_to_nearest_nonzero(maps, mapsb)
        return loss_distance