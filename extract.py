import torch as th
import math
import numpy as np

from sklearn import preprocessing
from video_loader import VideoLoader
from torch.utils.data import DataLoader
import argparse
from model import get_model
from preprocessing import Preprocessing
from random_sequence_shuffler import RandomSequenceSampler
import torch.nn.functional as F

# python extract.py --csv=input.csv --type=3d --batch_size=64 --num_decoding_thread=4
parser = argparse.ArgumentParser(description='Easy video feature extractor')

parser.add_argument(
    '--csv',
    type=str,
    help='input csv with video input path')
parser.add_argument('--batch_size', type=int, default=64,
                            help='batch size')
parser.add_argument('--type', type=str, default='s3d',
                            help='CNN type')
parser.add_argument('--half_precision', type=int, default=1,
                            help='output half precision float')
parser.add_argument('--only_preprocess', type=int, default=0,
                            help='only save the preprocessed video - no final features')
parser.add_argument('--num_decoding_thread', type=int, default=4,
                            help='Num parallel thread for video decoding')
parser.add_argument('--l2_normalize', type=int, default=1,
                            help='l2 normalize feature')
parser.add_argument('--resnext101_model_path', type=str, default='model/resnext101.pth',
                            help='Resnext model path')
parser.add_argument('--s3d_model_path', type=str, default='model/s3d_howto100m.pth',
                            help='Resnext model path')
args = parser.parse_args()

dataset = VideoLoader(
    args.csv,
    framerate=1 if args.type == '2d' else 24,
    size=224 if args.type == '2d' else 112,
    centercrop=(args.type == '3d'),
)
n_dataset = len(dataset)
sampler = RandomSequenceSampler(n_dataset, 10)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=args.num_decoding_thread,
    sampler=sampler if n_dataset > 10 else None,
)
preprocess = Preprocessing(args.type)
model = get_model(args)

with th.no_grad():
    for k, data in enumerate(loader):
        input_file = data['input'][0]
        output_file = data['output'][0]
        print(data['video'].shape)
        if len(data['video'].shape) > 3:
            print('Computing features of video {}/{}: {}'.format(
                k + 1, n_dataset, input_file))
            video = data['video'].squeeze()
            if len(video.shape) == 4:
                # video = preprocess(video)

                if args.only_preprocess:
                    print("Saving only the preprocessed video")
                    normalized_video = F.normalize(video, dim=1)
                    normalized_video = normalized_video.cpu().numpy()
                    # normalized_video = preprocessing.normalize(normalized_video)
                    # normalized_video = video / video.sum(0).expand_as(video)
                    np.save(output_file, normalized_video)

                else:
                    if args.type == '3d':
                        video = F.normalize(video, dim=1)

                    n_chunk = len(video)
                    if args.type == '3d':
                        features = th.cuda.FloatTensor(n_chunk, 2048).fill_(0)
                    elif args.type == 's3d':
                        features = th.cuda.FloatTensor(n_chunk, 1024).fill_(0)
                    n_iter = int(math.ceil(n_chunk / float(args.batch_size)))
                    for i in range(n_iter):
                        min_ind = i * args.batch_size
                        max_ind = (i + 1) * args.batch_size
                        video_batch = video[min_ind:max_ind].cuda()
                        batch_features = model(video_batch)
                        if args.type == 's3d':
                            batch_features = batch_features['mixed_5c']
                        if args.l2_normalize:
                            batch_features = F.normalize(batch_features, dim=1)
                        features[min_ind:max_ind] = batch_features
                    features = features.cpu().numpy()
                    if args.half_precision:
                        features = features.astype('float16')
                    np.save(output_file, features)
        else:
            print('Video {} already processed.'.format(input_file))
