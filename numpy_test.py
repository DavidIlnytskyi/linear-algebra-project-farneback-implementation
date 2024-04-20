import numpy as np

sigma = 4

kernel_size = int(4 * 4 + 1)
positions = np.arange(-kernel_size, kernel_size + 1, dtype=int)
gaussian_kernel = np.exp(-(positions**2) / (2 * sigma**2))

print(gaussian_kernel)