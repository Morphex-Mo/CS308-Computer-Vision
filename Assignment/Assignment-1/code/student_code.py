import numpy as np

def my_imfilter(image, filter):
  """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
    with matrices is fine and encouraged. Using opencv or similar to do the
    filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
    it may take an absurdly long time to run. You will need to get a function
    that takes a reasonable amount of time to run so that the TAs can verify
    your code works.
  - Remember these are RGB images, accounting for the final image dimension.
  """

  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  ############################
  ### TODO: YOUR CODE HERE ###
  # Unify grayscale/color processing to a 3D array (H, W, C).
  is_gray = (image.ndim == 2)
  if is_gray:
    image_work = image[:, :, np.newaxis]
  else:
    image_work = image

  image_work = image_work.astype(np.float64)
  filt = filter.astype(np.float64)

  height, width, channels = image_work.shape
  k_h, k_w = filt.shape
  pad_h = k_h // 2
  pad_w = k_w // 2

  padded = np.pad(
    image_work,
    ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
    mode='reflect'
  )

  filtered_image = np.zeros_like(image_work, dtype=np.float64)

  # Naive convolution over spatial dimensions and channels.
  for i in range(height):
    for j in range(width):
      patch = padded[i:i + k_h, j:j + k_w, :]
      for c in range(channels):
        filtered_image[i, j, c] = np.sum(patch[:, :, c] * filt)

  if is_gray:
    filtered_image = filtered_image[:, :, 0]

  ### END OF STUDENT CODE ####
  ############################

  return filtered_image

def create_hybrid_image(image1, image2, filter):
  """
  Takes two images and creates a hybrid image. Returns the low
  frequency content of image1, the high frequency content of
  image 2, and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  Returns
  - low_frequencies: numpy nd-array of dim (m, n, c)
  - high_frequencies: numpy nd-array of dim (m, n, c)
  - hybrid_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """

  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]

  ############################
  ### TODO: YOUR CODE HERE ###
  low_frequencies = my_imfilter(image1, filter)

  image2_low = my_imfilter(image2, filter)
  high_frequencies = image2 - image2_low

  hybrid_image = low_frequencies + high_frequencies
  hybrid_image = np.clip(hybrid_image, 0.0, 1.0)

  ### END OF STUDENT CODE ####
  ############################

  return low_frequencies, high_frequencies, hybrid_image
