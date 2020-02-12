from PIL import Image
import numpy as np
import im_lib.image_minip as im


def get_image_and_array(file):
    img = Image.open(file)
    im_array = np.array(img)

    im_array = im.scale_down(im_array, 2)
    img = Image.fromarray(im_array)

    return img, im_array
# --------------------------------------------------------------




img, im_array = get_image_and_array('images/city.jpg')
img.show(title="normal")


l = 41

bad_kernel = im.make_lens_blur_kernel_2D(l//2)
new_image_array = im.apply_kernel(im_array, bad_kernel)
new_image = Image.fromarray(new_image_array)
new_image.show(title='fourier lens blur')



bad_kernel = im.make_lazy_lens_blur(l//2)
new_image_array = im.apply_kernel(im_array, bad_kernel)
new_image = Image.fromarray(new_image_array)
new_image.show(title='programmed circle lens blur')



k = im.make_1d_gaussian_kernel(l//3, l)
new_image_array = im.apply_1d_kernel(im_array, k)
new_image = Image.fromarray(new_image_array)
new_image.show("gaussian blur")













