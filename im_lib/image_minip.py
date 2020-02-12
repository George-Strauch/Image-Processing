import math
import numpy as np
import random
from numba import njit

@njit
def randPixImage(width=600, height=600):
    image = np.empty(shape=(height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            for k in range(3):
                image[i, j, k] = random.uniform(0, 255)
    return image


@njit
def scale_down(img, sq_factor):
    height, width, _ = img.shape
    new_h = height//sq_factor
    new_w = width//sq_factor
    new_img = []

    for i in range(new_h):
        row = []
        for j in range(new_w):
            rgb = []
            for color in range(3):
                # make pixel of color: color
                sum_var = 0
                for a in range(sq_factor):
                    for b in range(sq_factor):
                        sum_var += img[i*sq_factor+a, j*sq_factor+b, color]

                rgb.append(np.uint8(sum_var / sq_factor**2))

            row.append(rgb)

        new_img.append(row)

    return np.array(new_img)


# applies kernel both horizontally and vertically
@njit
def apply_1d_kernel(img, kernel):
    mid = len(kernel) // 2
    height, width, _ = img.shape
    ker_len = len(kernel)
    tmp_ker = kernel

    new_image = np.empty(shape=(height, width, 3), dtype=np.uint8)

    # first go horizontally
    for row in range(height):
        for pixel in range(width):
            if mid > pixel:
                start = mid-pixel
                finish_p1 = ker_len

            elif ker_len-mid > width-pixel:
                start = 0
                finish_p1 = mid+(width-pixel)

            else:
                start = 0
                finish_p1 = ker_len
            # print("row, pix: ", row, pixel, " --- start, finish(+1): ", start, finish_p1)
            for color in range(3):
                sum_var = 0
                for i in range(start, finish_p1):
                    sum_var += tmp_ker[i] * img[row, pixel+i-mid, color]
                new_image[row, pixel, color] = sum_var

    # then go vertically
    for row in range(height):
        if mid > row:
            start = mid - row
            finish_p1 = ker_len

        elif ker_len - mid > height - row:
            start = 0
            finish_p1 = mid + (height - row)

        else:
            start = 0
            finish_p1 = ker_len

        for pixel in range(width):
            for color in range(3):
                sum_var = 0
                for i in range(start, finish_p1):
                    sum_var += kernel[i] * new_image[row+i-mid, pixel, color]
                new_image[row, pixel, color] = sum_var

    return new_image



# # applies kernel with loss of pixels on borders
@njit
def apply_kernel(img, kernel):
    height, width, _ = img.shape
    ker_len = kernel.shape[0]
    new_image = np.empty(shape=(height-ker_len, width-ker_len, 3), dtype=np.uint8)

    for row_ind in range(height-ker_len):
        for pix_ind in range(width-ker_len):
            for color in range(3):
                sm = 0
                for i in range(ker_len):
                    for v in range(ker_len):
                        sm = sm + kernel[i, v] * img[row_ind+i, pix_ind+v, color]
                new_image[row_ind, pix_ind, color] = sm

    return new_image


@njit
def normalize_1d_kernel(kernel):
    weight_sum = 0
    l = len(kernel)
    # get sum of weights
    for i in range(l):
            weight_sum += kernel[i]

    if 1-weight_sum < 0.03:
        return kernel

    for i in range(l):
        kernel[i] = kernel[i] / weight_sum

    return kernel


@njit
def normalize_kernel(kernel):
    weight_sum = 0
    h, w = kernel.shape
    # get sum of weights
    for i in range(h):
        for k in range(w):
            weight_sum += kernel[i, k]

    print('weight is: ', weight_sum)

    for i in range(h):
        for k in range(w):
            kernel[i, k] = kernel[i, k] / weight_sum

    return kernel


@njit
def norm_dist(std, dist):
    return (1 / (std*(2*math.pi)**.5)) * (math.e**(-1*(dist**2) / (2*(std**2))))


@njit
def norm_dist_2d(std, x, y):
    return (1 / (std*(2*math.pi)**.5)) * (math.e**(-1*(x**2 + y**2) / (2*(std**2))))


# std = standard deviation
@njit
def make_gauss_blur_kernel(std, sq_size=3):
    kernel = np.empty(shape=(sq_size, sq_size))
    mid = sq_size//2
    for i in range(sq_size):
        for j in range(sq_size):
            # dist = ((mid-i)**2 + (mid-j)**2)**.5
            # kernel[i, j] = round(norm_dist(std, dist), 5)
            kernel[i, j] = norm_dist_2d(std, (mid-i), (mid-j))

    return normalize_kernel(kernel)


@njit
def make_1d_gaussian_kernel(std, l=3):
    kernel = np.empty(shape=(l,))
    mid = l//2
    for i in range(l):
        kernel[i] = norm_dist(std, (i-mid))
    return normalize_1d_kernel(kernel)


# # returns an array of 1d kernels to apply
# @njit
# def make_lens_blur_kernel(num_kerels, raduis):
#     kernels = []
#
#     for k in range(num_kerels):
#         current_ker = []





@njit
def make_lens_blur_kernel_2D(diameter):
    kernel = []
    for a in range(diameter):
        row = []
        for b in range(diameter):
            dist_sq = (a-(diameter//2))**2 + (b-(diameter//2))**2
            val = 0.54100
            val += 0.66318 * np.cos(dist_sq)
            val -= 0.20942 * np.cos(3*dist_sq)
            val += 0.10471 * np.cos(5*dist_sq)
            val -= 0.08726 * np.cos(7*dist_sq)
            row.append(val)

        kernel.append(row)

    return normalize_kernel(np.array(kernel))




@njit
def make_lazy_lens_blur(l):
    bad_kernel = np.empty(shape=(l, l))
    for y in range(l):
        for x in range(l):
            if ((x - (l / 2)) ** 2 + (y - (l / 2)) ** 2) < (l / 2) ** 2:
                bad_kernel[x, y] = 1
            else:
                bad_kernel[x, y] = 1
    return normalize_kernel(bad_kernel)





























