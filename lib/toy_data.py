import numpy as np
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle


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
    elif data == "3d_cos":
        z = rng.rand(batch_size) * 5 - 2.5
        x = np.sin(3*z) * 2.5
        y = np.cos(3*z) * 2.5
        x = x + rng.randn(batch_size) * 0.05
        y = y + rng.randn(batch_size) * 0.2    
        return np.stack((x, y, z), 1)
    elif data == "3d_1spiral":
        n = np.sqrt(np.random.rand(batch_size)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size) * 0.5
        z = np.random.rand(batch_size) * 2 - 1.0
        out = np.stack((d1x, d1y, z), 1)
        out = out * 0.5
        out += np.random.randn(*out.shape) * 0.05
        return out
    elif data == '3Dswissroll':
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        z = rng.randn(batch_size) * 1.0    
        return np.stack((data[:,0], data[:,1], z), 1)
    elif data == 'swissroll_xy1':
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        z =  data[:,0] + data[:,1] + rng.randn(batch_size) * 0.1    
        return np.stack((data[:,0], data[:,1], z), 1)
    elif data == 'swissroll_xy5':
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        z =  data[:,0] + data[:,1] + rng.randn(batch_size) * 0.5    
        return np.stack((data[:,0], data[:,1], z), 1)
    elif data == 'swissroll_6D_xy1':
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        z2 =  data[:,0] + data[:,1] + rng.randn(batch_size) * 0.1
        z1 =  2*data[:,0]  + rng.randn(batch_size) * 0.1
        z4 =  rng.randn(batch_size) * 1.0
        z3 =  data[:,0] - 2*data[:,1] + rng.randn(batch_size) * 0.1
        return np.stack((data[:,0], data[:,1], z1, z2, z3, z4), 1)
    elif data == 'swissroll_xy01':
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        z =  data[:,0] + data[:,1] + rng.randn(batch_size) * 0.01    
        return np.stack((data[:,0], data[:,1], z), 1)
    elif data == 'swissroll_flat01':
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        z =  rng.randn(batch_size) * 0.01    
        return np.stack((data[:,0], data[:,1], z), 1)
    elif data == 'swissroll_n_8gaussians':
        #swissroll + 8gaussians
        #swissroll portion
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        #gaussians portion
        scale = 2.
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

        return np.stack((data[:,0], data[:,1], dataset[:,0], dataset[:,1]), 1)
    
    elif data == 'checkerboard_n_2spirals':
        # 2spirals
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        x = x * 0.5
        # checkerboard
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
         # Ensure x1 and x[:, 0] have the same number of dimensions
        x1 = x1[:, None]  # Ensure x1 has two dimensions
        x2 = x2[:, None]
        x3 = x[:, 0][:, None]
        x4 = x[:, 1][:, None]

        # print(x1.shape, x2.shape, x3.shape, x4.shape)
        return np.concatenate([x1, x2, x3,x4], 1) * 2


    elif data == 'moons_n_2spirals':

        #moons
        data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.02)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        # 2spirals
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        x = x * 0.5
        return np.concatenate([data[:,0][:, None], data[:,1][:, None], x[:,0][:, None], x[:,1][:, None]], 1)
    
    else:
        return inf_train_gen("8gaussians", rng, batch_size)
