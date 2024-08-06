import numpy as np
import imageio.v3 as iio

import sys

def img_toBW(filename, thresh: float | int = 0.5):
    """
    Converts an image to a binary image based on a threshold value.
    """

    #Reads and converts the image to a numpy array
    img = iio.imread(filename)
    img = np.asarray(img)

    #Defines the shape of the image and creates empty BW
    shape = (img.shape[0], img.shape[1])
    BW = np.empty(shape)

    #Creates the BW
    if img.ndim == 2: # Already grayscale/BW
        BW = img > (thresh * 255) if isinstance(thresh, float) else thresh
    else: # RGB format
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                #Uses the luminocity formula 
                luminocity = 0.2126 * img[i, j, 0] + 0.7152 * img[i, j, 1] + 0.0722 * img[i, j, 2]
                BW[i, j] = luminocity > (thresh * 255) if isinstance(thresh, float) else thresh

    return BW

def main():
    # Create the image
    filename = "image.png"
    BW = img_toBW(filename)

    return 0

if __name__ == '__main__':
    if hasattr(sys, 'ps1'):  # Check if in interactive mode
        main()
    else:
        sys.exit(main())