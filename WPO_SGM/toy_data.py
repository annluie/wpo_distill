import numpy as np
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle

#for cifar10
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random

# set a seed to get the same data every time
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Optional: For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Dataset iterator
def inf_train_gen(data, rng=None, batch_size=200):
    if rng is None:
        rng = np.random.RandomState()

    if data == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data

    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        return data

    elif data == "rings":
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        X = util_shuffle(X, random_state=rng)

        # Add noise
        X = X + rng.normal(scale=0.08, size=X.shape)

        return X.astype("float32")

    elif data == "moons":
        data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.01)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        return data

    elif data == "8gaussians":
        scale = 4.
        centers = [(5, 0), (-5, 0), (0, 5), (0, -5), (5. / np.sqrt(2), 5. / np.sqrt(2)),
                   (5. / np.sqrt(2), -5. / np.sqrt(2)), (-5. / np.sqrt(2),
                                                         5. / np.sqrt(2)), (-5. / np.sqrt(2), -5. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))

    elif data == "2spirals":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return x

    elif data == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return np.concatenate([x1[:, None], x2[:, None]], 1) * 2

    elif data == "line":
        x = rng.rand(batch_size) * 5 - 2.5
        y = x
        return np.stack((x, y), 1)
    elif data == "cos":
        x = rng.rand(batch_size) * 5 - 2.5
        y = np.sin(x) * 2.5
        return np.stack((x, y), 1)
    elif data == 'swissroll_6D_xy1':
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        z2 =  data[:,0] + data[:,1] + rng.randn(batch_size) * 0.1
        z1 =  2*data[:,0]  + rng.randn(batch_size) * 0.1
        z4 =  rng.randn(batch_size) * 1.0
        z3 =  data[:,0] - 2*data[:,1] + rng.randn(batch_size) * 0.1
        return np.stack((data[:,0], data[:,1], z1, z2, z3, z4), 1)
    elif data =='cifar10':
        '''      # Basic transform: convert to tensor and normalize to [-1, 1]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) # Normalize from [0,1] to [-1,1]
        ])  
        
        #normalize to have mean 0 and std 1 for each channel
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # Normalize to mean=0, std=1
        ])
        b = batch_size
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=b, shuffle=True, num_workers=4)

        # flatten images to [batch_size, 3072] for score model input (vs [batch_size, 3, 32, 32])
        for images, _ in train_loader:
            flat_images = images.view(images.size(0), -1)  # shape: [64, 3072]
            #print(flat_images.min(), flat_images.max())    # should be around -1 to 1
            break
        return flat_images
        '''
        if 'train_loader' not in locals():
            # Normalize to mean=0, std=1 for each channel
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            b = batch_size
            train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            train_loader = DataLoader(train_dataset, batch_size=b, shuffle=False, num_workers=1)

        # Flatten one batch of images to [batch_size, 3072] for model input
        for images, _ in train_loader:
            flat_images = images.view(images.size(0), -1)
            # flat_images: [batch_size, 3072]
            #print("Mean:", flat_images.mean().item())
            #print("Std:", flat_images.std().item())
            break
        return flat_images
    else:
        return inf_train_gen("8gaussians", rng, batch_size)
