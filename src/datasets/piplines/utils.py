import cv2
import numpy as np
import numbers

cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}


def imresize(img,
             size,
             return_scale=False,
             interpolation='bilinear',
             out=None):
    """Resize image to a given size.

    Args:
        img (ndarray): The input image.
        size (tuple[int]): Target size (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        out (ndarray): The output destination.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or `resized_img`.
    """
    h, w = img.shape[:2]

    resized_img = cv2.resize(img, size, dst=out, interpolation=cv2_interp_codes[interpolation])
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale


def imrescale(img,
              scale,
              return_scale=False,
              interpolation='bilinear'):
    """Resize image while keeping the aspect ratio.

    Args:
        img (ndarray): The input image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image.
        interpolation (str): Same as :func:`resize`.

    Returns:
        ndarray: The rescaled image.
    """
    h, w = img.shape[:2]
    new_size, scale_factor = rescale_size((w, h), scale, return_scale=True)
    rescaled_img = imresize(img, new_size, interpolation=interpolation)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img


def rescale_size(old_size, scale, return_scale=False):
    """Calculate the new size to be rescaled to.

    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.

    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(f'Scale must be a number or tuple of int, but got {type(scale)}')

    new_size = _scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


def _scale_size(size, scale):
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    w, h = size
    return int(w * float(scale) + 0.5), int(h * float(scale) + 0.5)


def impad(img,
          shape,
          pad_val,
          padding=None,
          padding_mode='constant'):

    assert (shape is not None) ^ (padding is not None)
    if shape is not None:
        width = max(shape[1] - img.shape[1], 0)
        height = max(shape[0] - img.shape[0], 0)
        padding = (0, 0, width, height)

    # check pad_val
    if isinstance(pad_val, tuple):
        assert len(pad_val) == img.shape[-1]
    elif not isinstance(pad_val, numbers.Number):
        raise TypeError('pad_val must be a int or a tuple. '
                        f'But received {type(pad_val)}')

    # check padding
    if isinstance(padding, tuple) and len(padding) in [2, 4]:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
    elif isinstance(padding, numbers.Number):
        padding = (padding, padding, padding, padding)
    else:
        raise ValueError('Padding must be a int or a 2, or 4 element tuple.' f'But received {padding}')

    # check padding mode
    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

    border_type = {
        'constant': cv2.BORDER_CONSTANT,
        'edge': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT_101,
        'symmetric': cv2.BORDER_REFLECT
    }
    img = cv2.copyMakeBorder(
        img,
        padding[1],
        padding[3],
        padding[0],
        padding[2],
        border_type[padding_mode],
        value=pad_val)

    return img


def impad_to_multiple(img,
                      divisor,
                      pad_val=0):

    pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
    pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
    return impad(img, shape=(pad_h, pad_w), pad_val=pad_val)


def imnormalize(img, mean, std, to_rgb=True, is_zero_to_one=False):

    img = img.copy().astype(np.float32)
    return imnormalize_(img, mean, std, to_rgb, is_zero_to_one=is_zero_to_one)


def imnormalize_(img, mean, std, to_rgb=True, is_zero_to_one=False):
    if not is_zero_to_one:  # convert to [0, 1]
        img = img/255

    # cv2 inplace normalization does not accept uint8
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # The channel order of the images read by opencv defaults to BGR
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img


def convert_color_factory(src, dst):

    code = getattr(cv2, f'COLOR_{src.upper()}2{dst.upper()}')

    def convert_color(img):
        out_img = cv2.cvtColor(img, code)
        return out_img

    return convert_color


bgr2rgb = convert_color_factory('bgr', 'rgb')

rgb2bgr = convert_color_factory('rgb', 'bgr')

bgr2hsv = convert_color_factory('bgr', 'hsv')

hsv2bgr = convert_color_factory('hsv', 'bgr')

bgr2hls = convert_color_factory('bgr', 'hls')

hls2bgr = convert_color_factory('hls', 'bgr')