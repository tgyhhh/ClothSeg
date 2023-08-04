import torch
from torch import Tensor
from models.base import BaseModel
from models.heads import FPNHead
from models.heads import FPF
class CustomVIT_Pre(BaseModel):
    def __init__(self, backbone: str = 'ResT-S', num_classes: int = 19) -> None:
        super().__init__(backbone, num_classes)
        self.decode_head = FPF(self.backbone.channels, 128, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: Tensor):
        y = self.backbone(x)
        out_local, out_global, out_projection, out_feat = self.decode_head(y)   # 4x reduction in image size
        return out_local, out_global, out_projection, out_feat


if __name__ == '__main__':
    model = CustomVIT('ResT-S', 19)
    model.init_pretrained('checkpoints/backbones/rest/rest_small.pth')
    x = torch.zeros(2, 3, 512, 512)
    y = model(x)
    print(y.shape)
        

