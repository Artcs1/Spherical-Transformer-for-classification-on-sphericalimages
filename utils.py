import numpy as np
import cv2

from scipy.spatial.transform import Rotation as R

class GaussianBlur(object):
    def __init__(self, p=1, kernel_size=None, sigma_min=0.1, sigma_max=2.0):
        self.p = p
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.kernel_size = kernel_size   

    def __call__(self, pic):
        if np.random.rand(1) > self.p:
            return pic 
        
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        
        if self.kernel_size == None:
            self.kernel_size = (1+2*np.ceil(2*sigma)).astype('int')  

        pic = cv2.GaussianBlur(np.array(pic), (self.kernel_size, self.kernel_size), sigma)
        return np.array(pic)

class GaussianNoise(object):
    def __init__(self, p=1, mean=None, std=None):
        self.p = p
        self.std = std
        self.mean = mean
        if self.mean == None:
            self.mean = np.random.uniform(0.0, 0.001)
        if self.std == None:
            self.std = np.random.uniform(0.0, 0.03)
        
    def __call__(self, pic):
        if np.random.rand(1) > self.p:
            return pic
        return pic + np.random.randn(*pic.shape) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Cutout(object):
    """
    Applies cutout augmentation
    """

    def __init__(self, p=1, max_size=None, n_squares=None):
        self.p = p
        if max_size == None:
            self.size = np.random.randint(10, 50)
        if n_squares == None:
            self.n_squares = np.random.randint(1, 5)

    def __call__(self, pic):
        """
        Args:
            pic (np.ndarray): Image to be cut
        """
        h, w, _ = pic.shape
        new_image = pic
        if np.random.rand(1) < self.p:
            for _ in range(self.n_squares):
                y = np.random.randint(h)
                x = np.random.randint(w)
                y1 = np.clip(y - self.size // 2, 0, h)
                y2 = np.clip(y + self.size // 2, 0, h)
                x1 = np.clip(x - self.size // 2, 0, w)
                x2 = np.clip(x + self.size // 2, 0, w)
                new_image[y1:y2, x1:x2, :] = 0

        return new_image

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'size={0}'.format(self.size)
        format_string += ')'
        return format_string

class CircularHorizontalShift(object):

    """
    Executes a circular horizontal shift 
    """

    def __init__(self, p=0.5, shift_length=None):
        self.p = p
        self.shift_length = shift_length

    def __call__(self, pic):
        """
        Args:
            pic (np.ndarray): Image to be shifted
        """
        if self.shift_length == None:
            self.shift_length = np.random.randint(0, pic.shape[2])

        shifted_img = pic
        if np.random.rand(1) < self.p:
            shifted_img = np.concatenate((pic[:, :, self.shift_length:], pic[:, :, :self.shift_length]), axis=2)
        
        return shifted_img
    
    def __repr__(self):
        format_string = self.__class__.__name__+ '('
        format_string += 'shift_length={0}'.format(self.shift_length)
        format_string += ')'
        return format_string

class Rotate(object):
    def __init__(self, p=0.5, rz=180, rx=15, ry=15):
        self.p = p
        self.rz = rz
        self.rx = rx
        self.ry = ry

    def __call__(self, pic):
        """
        Args:
            pic (np.ndarray): Image to be shifted
        """
        if np.random.rand(1) > self.p:
            return pic

        self.rz = np.random.uniform(-self.rz,self.rz)
        self.rx = np.random.uniform(-self.rx,self.rx)
        self.ry = np.random.uniform(-self.ry,self.ry)
        self.r = R.from_euler("zxy", [self.rz, self.rx, self.ry], degrees=True).as_matrix()

        colors = pic
        dim = pic.shape
        phi, theta = np.meshgrid(np.linspace(0, np.pi, num=dim[0], endpoint=False),
                                 np.linspace(0, 2 * np.pi, num=dim[1], endpoint=False))
        coordSph = np.stack([(np.sin(phi) * np.cos(theta)).T, (np.sin(phi) * np.sin(theta)).T, np.cos(phi).T], axis=2)

        eps = 1e-8
        data = np.array(np.dot(coordSph.reshape((dim[0] * dim[1], 3)), self.r))
        coordSph = data.reshape((dim[0] * dim[1], 3))

        x, y, z = data[:, ].T
        z = np.clip(z, -1 + eps, 1 - eps)

        phi = np.arccos(z)
        theta = np.arctan2(y, x)
        theta[theta < 0] += 2 * np.pi
        theta = dim[1] / (2 * np.pi) * theta
        phi = dim[0] / np.pi * phi

        mapped = np.stack([theta.reshape(dim[0], dim[1]), phi.reshape(dim[0], dim[1])], axis=2).astype(np.float32)

        colors = cv2.remap(colors, mapped, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        return np.array(colors)


class Deterministic_Rotate(object):
    def __init__(self, rz=180, rx=15, ry=15):
        self.rz = rz
        self.rx = rx
        self.ry = ry

    def __call__(self, pic):
        """
        Args:
            pic (np.ndarray): Image to be shifted
        """
        self.r = R.from_euler("zxy", [self.rz, self.rx, self.ry], degrees=True).as_matrix()

        colors = pic
        dim = pic.shape
        phi, theta = np.meshgrid(np.linspace(0, np.pi, num=dim[0], endpoint=False),
                                 np.linspace(0, 2 * np.pi, num=dim[1], endpoint=False))
        coordSph = np.stack([(np.sin(phi) * np.cos(theta)).T, (np.sin(phi) * np.sin(theta)).T, np.cos(phi).T], axis=2)

        eps = 1e-8
        data = np.array(np.dot(coordSph.reshape((dim[0] * dim[1], 3)), self.r))
        coordSph = data.reshape((dim[0] * dim[1], 3))

        x, y, z = data[:, ].T
        z = np.clip(z, -1 + eps, 1 - eps)

        phi = np.arccos(z)
        theta = np.arctan2(y, x)
        theta[theta < 0] += 2 * np.pi
        theta = dim[1] / (2 * np.pi) * theta
        phi = dim[0] / np.pi * phi

        mapped = np.stack([theta.reshape(dim[0], dim[1]), phi.reshape(dim[0], dim[1])], axis=2).astype(np.float32)

        colors = cv2.remap(colors, mapped, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        return np.array(colors)


