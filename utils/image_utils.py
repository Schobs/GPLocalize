import numpy as np

def load_and_resize_image(image_path, coords, datatype_load, load_im_size):
    """Load image and resize it to the specified size. Also resize the coordinates to match the new image size.

    Args:
        image_path (str): _description_
        coords ([ints]): _description_

    Returns:
        _type_: _description_
    """

    original_image = datatype_load(image_path)
    original_size = np.expand_dims(np.array(list(original_image.size)), 1)
    if list(original_image.size) != load_im_size:
        resizing_factor = [
            list(original_image.size)[0] / load_im_size[0],
            list(original_image.size)[1] / load_im_size[1],
        ]
        resized_factor = np.expand_dims(np.array(resizing_factor), axis=0)
    else:
        resizing_factor = [1, 1]
        resized_factor = np.expand_dims(np.array(resizing_factor), axis=0)

    # potentially resize the coords
    coords = np.round(coords * [1 / resizing_factor[0], 1 / resizing_factor[1]])
    image = np.expand_dims(
        normalize_image(original_image.resize(load_im_size)), axis=0
    )

    return resized_factor, original_size, image, coords

def normalize_image(image):
    """Adds small epsilon to std to avoid divide by zero

    Args:
        image (_type_): _description_

    Returns:
        _type_: _description_
    """

    norm_image = (image - np.mean(image)) / (np.std(image) + 1e-100)

    return norm_image