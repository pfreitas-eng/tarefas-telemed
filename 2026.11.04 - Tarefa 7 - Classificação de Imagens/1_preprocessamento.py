import json
import os
import zipfile

import numpy as np
import torch
from numpy.lib import format as npformat
from torchvision import tv_tensors
from torchvision.transforms import v2

source_data_dir = './source_data'
image_size = 224
chunk_size = 500
splits = ['train', 'val', 'test']


def iter_raw_chunks(split):
    # lê o array 'image' em blocos, pois o raw descompactado (float64, 3 canais) não cabe na memória de uma vez
    path = os.path.join(source_data_dir, f'Dataset5_raw_{split}.npz')
    with zipfile.ZipFile(path) as zf, zf.open('image.npy') as fh:
        version = npformat.read_magic(fh)
        if version == (1, 0):
            shape, _, dtype = npformat.read_array_header_1_0(fh)
        else:
            shape, _, dtype = npformat.read_array_header_2_0(fh)
        bytes_per_image = int(np.prod(shape[1:])) * dtype.itemsize
        for start in range(0, shape[0], chunk_size):
            n = min(chunk_size, shape[0] - start)
            buffer = fh.read(n * bytes_per_image)
            yield np.frombuffer(buffer, dtype=dtype).reshape((n,) + shape[1:])


def load_labels(split):
    with np.load(os.path.join(source_data_dir, f'Dataset5_raw_{split}.npz')) as data:
        return data['image_name'], data['image_label'].astype(np.int64).ravel()


def first_channel_01(chunk):
    # 1. seleção do primeiro canal   2. normalização entre 0 e 1
    return chunk[..., 0] / 255.0


# 3. média e desvio padrão calculados apenas com o conjunto de treinamento
total, total_sq, count = 0.0, 0.0, 0
for chunk in iter_raw_chunks('train'):
    block = first_channel_01(chunk)
    total += block.sum()
    total_sq += np.square(block).sum()
    count += block.size
train_mean = total / count
train_std = np.sqrt(total_sq / count - train_mean ** 2)
print(f"Média treino: {train_mean:.6f} | Desvio treino: {train_std:.6f}")

stats_path = os.path.join(source_data_dir, 'Dataset5_clean_stats.json')
with open(stats_path, 'w', encoding='utf-8') as f:
    json.dump({'mean': float(train_mean), 'std': float(train_std),
               'source': 'Dataset5_raw_train.npz, primeiro canal / 255'}, f, indent=4)
print("Estatísticas salvas em:", stats_path)

# 4. normalização com a média e o desvio do treino
normalize = v2.Compose([
    v2.ToDtype(torch.float32, scale=False),
    v2.Normalize(mean=[float(train_mean)], std=[float(train_std)]),
])

for split in splits:
    image_names, labels = load_labels(split)
    clean = np.empty((len(labels), image_size, image_size), dtype=np.float32)
    start = 0
    for chunk in iter_raw_chunks(split):
        # (n, H, W) -> (n, 1, H, W) para o formato de canal esperado pelo torchvision
        block = tv_tensors.Image(torch.from_numpy(first_channel_01(chunk)[:, None]))
        clean[start:start + len(chunk)] = normalize(block).numpy()[:, 0]
        start += len(chunk)

    output_path = os.path.join(source_data_dir, f'Dataset5_clean_{split}.npz')
    np.savez(output_path, image=clean, image_name=image_names, image_label=labels)
    print(f"Salvo {output_path} | imagens: {len(labels)} | shape: {clean.shape}")
    del clean
