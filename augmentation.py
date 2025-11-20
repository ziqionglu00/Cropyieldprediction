import torch
from sklearn import preprocessing
from einops import rearrange
from dataload import Sentinel_Dataset
import random

class DataWrapper(object):
    def __init__(self, img_size=224, s=1, kernel_size=9, train=True):
        self.img_size = img_size
        self.s = s
        self.kernel_size = kernel_size
        self.mode=train
    def __call__(self, x):
        x = x.to(torch.float32)
        if self.mode:
            xi=self.augment_sample(x) 
        else:
            xi=self.sample(x)
        return xi

    def safe_evi(self,nir, red, blue, epsilon=1e-8):
        denominator = nir + 6 * red - 7.5 * blue + 1
        safe_denominator = torch.where(
            denominator == 0, 
            torch.tensor(epsilon, device=denominator.device), 
            denominator
        )
        return 2.5 * (nir - red) / safe_denominator

    def random_horizontal_flip(self,sample, p=0.5):
        if random.random() < p:
            sample = torch.flip(sample, dims=[3])
        return sample

    def random_vertical_flip(self,sample, p=0.5):
        if random.random() < p:
            sample = torch.flip(sample, dims=[2])
        return sample

    def random_rotation(self,sample):
        k = random.choice([0, 1, 2, 3])
        if k:
            sample = torch.rot90(sample, k, dims=[2, 3])
        return sample

    def random_color_jitter(self,sample, brightness=0.2, contrast=0.2):
        factor_brightness = 1.0 + random.uniform(-brightness, brightness)
        factor_contrast   = 1.0 + random.uniform(-contrast, contrast)
        
        sample = sample * factor_brightness

        global_mean = sample.mean(dim=[0, 2, 3], keepdim=True) 
        sample = global_mean + factor_contrast * (sample - global_mean)
        return sample

    def augment_sample(self,sample):
        mean=[0.06034789, 0.08395179, 0.07924612, 0.12489378, 0.24519444,
            0.29666891, 0.30588894, 0.32263374, 0.24204728, 0.15821346]
        std=[0.06244765, 0.05826417, 0.07020638, 0.06704885, 0.08744965,
             0.11501483, 0.11429801, 0.11636653, 0.09800738, 0.09861727]

        mean_tensor = torch.tensor(mean, dtype=sample.dtype, device=sample.device).view(1,-1, 1, 1)
        std_tensor = torch.tensor(std, dtype=sample.dtype, device=sample.device).view(1,-1, 1, 1)

        sample = self.random_horizontal_flip(sample, p=0.5) 
        sample = self.random_vertical_flip(sample, p=0.5) 
        sample = self.random_rotation(sample)

        s2 = (sample - mean_tensor) / (std_tensor + 1e-8)

        return s2


    def sample(self,sample):

        mean=[0.06034789, 0.08395179, 0.07924612, 0.12489378, 0.24519444,
            0.29666891, 0.30588894, 0.32263374, 0.24204728, 0.15821346]
        std=[0.06244765, 0.05826417, 0.07020638, 0.06704885, 0.08744965,
             0.11501483, 0.11429801, 0.11636653, 0.09800738, 0.09861727]

        mean_tensor = torch.tensor(mean, dtype=sample.dtype, device=sample.device).view(1,-1, 1, 1)
        std_tensor = torch.tensor(std, dtype=sample.dtype, device=sample.device).view(1,-1, 1, 1)

        sample = (sample - mean_tensor) / (std_tensor + 1e-8)

        return sample


    def augment_batch(self,batch):

        augmented = []
        for sample in batch:
            augmented.append(self.augment_sample(sample))
        return torch.stack(augmented)


if __name__ == '__main__':
    image_dir = "yieldpredicion/data"
    label_path = "yieldpredicion/yieldlabel.txt"
    dataset = Sentinel_Dataset(image_dir, label_path)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    it = iter(train_loader)
    x, f, y = next(it)

    x = x[:, :, :22, :, :, :]

    x = rearrange(x, 'b t g h w c -> (b t g) c h w')

    wrapper = DataWrapper()

    xi, xj = wrapper(x)
    print(xi.shape)
    print(xj.shape)
