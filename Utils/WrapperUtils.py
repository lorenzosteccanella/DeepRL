import cv2
import numpy as np
#from skimage.transform import downscale_local_mean
#from skimage.measure import compare_ssim


def normalize(image):
    im = image / 255
    return im


def get_pixels_from_obs(observation):
    return observation


def np_in_list(array, list):
    for element in list:
        if np.array_equal(element, array):
            return True
    return False


#def ssim_in_list(array, list, multichannel):
#    for element in list:
#        if ObsPixelWrapper.SSIM_equal(element, array, multichannel):
#            return True
#    return False


#def SSIM_equal(abstract_state_1, abstract_state_2, multichannel, verbose=False):
#    x = round(compare_ssim(abstract_state_1, abstract_state_2, 3, multichannel=multichannel, gaussian_weights=True,
#                           use_sample_covariance=False), 3)
#    # print(x)
#    if x == 1:
#        return True
#    else:
#        if verbose:
#            print(x)
#        False


def sample_colors(image, threshold):
    img = cv2.medianBlur(image, 1)
    # img = np.clip(img, treshold, treshold)

    img_quantizied = np.floor_divide(img, threshold) * threshold
    # _, img = cv2.threshold(img_quantizied, 100, 255, cv2.THRESH_BINARY)

    return img_quantizied


def make_gray_scale(image):
    im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    stacked_img = np.stack((im,) * 3, axis=-1)
    return stacked_img


def hash_numpy_array_equal(a, b):
    return hash(a.tostring()) == hash(b.tostring())


#def make_downsampled_image(image, zone_size_x, zone_size_y):
#    len_x = image.shape[0]
#    len_y = image.shape[1]
#
#    if (len_x % zone_size_x == 0) and (len_y % zone_size_y == 0):
#        if (len(image.shape)) == 3:
#            img_blurred = downscale_local_mean(image, (zone_size_y, zone_size_x, 1)).astype(np.uint8)
#        else:
#            img_blurred = downscale_local_mean(image, (zone_size_y, zone_size_x)).astype(np.uint8)
#        return img_blurred
#
#    else:
#        raise Exception("The image " + str(len_x) + "x" + str(len_y) +
#                        " can not be fragmented into zones " + str(zone_size_x) + "x" + str(zone_size_y))
