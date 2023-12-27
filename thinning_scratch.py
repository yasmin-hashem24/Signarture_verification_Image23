from scipy import ndimage as ndi
from commonfunctions import  *


G123_LUT = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                     0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
                     0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                     0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                     1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0,
                     0, 1, 1, 0, 0, 1, 0, 0, 0], dtype=bool)

G123P_LUT = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0,
                      0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                      1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
                      0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
def thin_customized(image, max_num_iter=None):
    skel = np.asarray(image, dtype=bool).astype(np.uint8)
    
    mask = np.array([[8, 4, 2],
                     [16, 0, 1],
                     [32, 64, 128]], dtype=np.uint8)

    max_num_iter = max_num_iter or np.inf
    num_iter = 0
    n_pts_old, n_pts_new = np.inf, np.sum(skel)

    while n_pts_old != n_pts_new and num_iter < max_num_iter:
        n_pts_old = n_pts_new

        for lut in [G123_LUT, G123P_LUT]:
            N = ndi.correlate(skel, mask, mode='constant')
            D = np.take(lut, N)
            skel[D] = 0

        n_pts_new = np.sum(skel)
        num_iter += 1

    return skel.astype(bool)


def thin_customized_parallel(image, max_num_iter=None, num_threads=2):
    skel = np.asarray(image, dtype=bool).astype(np.uint8)
    mask = np.array([[8, 4, 2],
                     [16, 0, 1],
                     [32, 64, 128]], dtype=np.uint8)

    max_num_iter = max_num_iter or np.inf

    global result
    result = []
    result_lock = threading.Lock()

    threads = []

    for _ in range(num_threads):
        thread = threading.Thread(target=thin_customized, args=(image, skel.copy(), mask, G123_LUT, max_num_iter, result_lock))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return result[0]

def GaussianBlurImage(image, sigma):
    filter_size = 2 * int(4 * sigma + 0.5) + 1
    gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
    m = filter_size // 2
    n = filter_size // 2
    
    for x in range(-m, m + 1):
        for y in range(-n, n + 1):
            x1 = 2 * np.pi * (sigma**2)
            x2 = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            gaussian_filter[x + m, y + n] = (1 / x1) * x2
    
    im_filtered = convolve2d(image, gaussian_filter, mode='same', boundary='wrap')
    return im_filtered



def GaussianBlurImage_parallel(image, sigma, num_threads=2):
    global result
    result = []
    result_lock = threading.Lock()

    threads = []

    for _ in range(num_threads):
        thread = threading.Thread(target=GaussianBlurImage, args=(image, sigma, result_lock, result))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return result[0]

