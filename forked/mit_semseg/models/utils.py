import sys
import os
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    if map_location is None:
        map_location = DEVICE
    return torch.load(cached_file, map_location=map_location)
