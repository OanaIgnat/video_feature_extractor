import sys
import torch as th
import torchvision.models as models

from s3dg import S3D
from videocnn.models import resnext
from torch import nn


class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return th.mean(x, dim=[-2, -1])


def get_model(args):
    assert args.type in ['2d', '3d', 's3d']
    if args.type == '2d':
        print('Loading 2D-ResNet-152 ...')
        model = models.resnet152(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-2], GlobalAvgPool())
        model = model.cuda()
    elif args.type == '3d':
        print('Loading 3D-ResneXt-101 ...')
        model = resnext.resnet101(
            num_classes=400,
            shortcut_type='B',
            cardinality=32,
            sample_size=112,
            sample_duration=16,
            last_fc=False)
        model = model.cuda()
        model_data = th.load(args.resnext101_model_path)
        model.load_state_dict(model_data)
    else:
        print('Loading S3D ...')
        model = S3D(
            'model/s3d_dict.npy',
            num_classes=512
         )
        model = model.cuda()
        model_data = th.load(args.s3d_model_path)
        model.load_state_dict(model_data)
        # device = th.device('cuda:0')
        # model.to(device)
    model.eval()
    print('loaded')
    return model
