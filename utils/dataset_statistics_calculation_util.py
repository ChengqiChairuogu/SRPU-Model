import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
import sys

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs import json_config as cfg_json_gen
from configs import base as cfg_base

class StatsDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.samples = data['samples']
        self.datasets_info = data['datasets_info']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        dataset = sample['dataset']
        frame = sample['frames'][0]
        raw_root = self.datasets_info[dataset]['raw_image_root'].replace('\\', '/')
        
        image_path = project_root / raw_root / frame
        
        if not image_path.exists():
            return np.zeros((cfg_base.IMAGE_HEIGHT, cfg_base.IMAGE_WIDTH, cfg_base.INPUT_DEPTH), dtype=np.float32)
        
        image = np.array(Image.open(image_path).convert('L')) / 255.0
        image_stack = np.stack([image] * cfg_base.INPUT_DEPTH, axis=-1)
        return image_stack

def calculate_mean_std(dataloader):
    mean = 0.0
    std = 0.0
    n_samples = 0
    for images in tqdm(dataloader):
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        n_samples += batch_size
    mean /= n_samples
    std /= n_samples
    return mean.tolist(), std.tolist()

def main():
    json_path = project_root / cfg_json_gen.JSON_OUTPUT_DIR_NAME / 'master_labeled_dataset.json'
    dataset = StatsDataset(json_path)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
    mean, std = calculate_mean_std(dataloader)
    stats = {'mean': mean, 'std': std, 'input_depth_at_calculation': cfg_base.INPUT_DEPTH}
    output_path = project_root / 'json/dataset_stats.json'
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"统计保存到: {output_path}")

if __name__ == '__main__':
    main() 