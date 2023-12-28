from commonfunctions import *
from HogScratch import *

class PreprocessorScratch:
    def __init__(self):
        self.G123_LUT = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
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

        self.G123P_LUT = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
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
        
    '''This is the same as the one in preprocessor'''
    def img2gray(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) > 3:
            img = img[:, :, 0:3, 0]
        if len(img.shape) == 2:
            return img
        else:
            gray_img = rgb2gray(img)
            return gray_img
    
    '''This function still needs implementing'''
    def contrastEnhancemet(self, img: np.ndarray) -> None:
        enhanced = img.copy()
        low, high = np.percentile(img, [0.2, 99.8])
        enhanced = rescale_intensity(img, in_range=(low, high))
        return enhanced

    '''Function done'''
    def skewCorrection(self, img: np.ndarray) -> np.ndarray:
        skew_corrected_imgs = transform.resize(img, (128, 256), mode='reflect', anti_aliasing=True)
        return skew_corrected_imgs

    '''This function replaces noise removal'''
    def gaussian(self, img: np.ndarray, sigma: float):
        filter_size = 2 * int(4 * sigma + 0.5) + 1
        gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
        m = filter_size // 2
        n = filter_size // 2
        
        for x in range(-m, m + 1):
            for y in range(-n, n + 1):
                x1 = 2 * np.pi * (sigma**2)
                x2 = np.exp(-(x**2 + y**2) / (2 * sigma**2))
                gaussian_filter[x + m, y + n] = (1 / x1) * x2
        
        im_filtered = convolve2d(img, gaussian_filter, mode='same', boundary='wrap')
        return im_filtered
    
    '''Implemented local thresholding'''
    def localThresholding(self, img, block_size, offset):
    
        binary_image = np.zeros(img.shape, dtype=bool)
        
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                
                start_row = max(0, i - block_size // 2)
                end_row = min(img.shape[0], i + block_size // 2 + 1)
                start_col = max(0, j - block_size // 2)
                end_col = min(img.shape[1], j + block_size // 2 + 1)

                local_mean = np.mean(img[start_row:end_row, start_col:end_col])

                local_threshold = local_mean - offset
                binary_image[i, j] = img[i, j] < local_threshold
        return binary_image.astype(np.uint8)

    '''utility function for calling selected version of local thresholfing'''
    def binarize(self, img: np.ndarray) -> np.ndarray:
        block_size = 21
        binarized_img = self.localThresholding(img, block_size, 0.1)
        return binarized_img

    '''Sequential version of thinning'''
    def thinningCustomized(self, img: np.ndarray, max_num_iter: int=None):
        skel = np.asarray(img, dtype=bool).astype(np.uint8)
        
        mask = np.array([[8, 4, 2],
                        [16, 0, 1],
                        [32, 64, 128]], dtype=np.uint8)

        max_num_iter = max_num_iter or np.inf
        num_iter = 0
        n_pts_old, n_pts_new = np.inf, np.sum(skel)

        while n_pts_old != n_pts_new and num_iter < max_num_iter:
            n_pts_old = n_pts_new

            for lut in [self.G123_LUT, self.G123P_LUT]:
                N = ndi.correlate(skel, mask, mode='constant')
                D = np.take(lut, N)
                skel[D] = 0

            n_pts_new = np.sum(skel)
            num_iter += 1

        return skel.astype(bool)
    
    def erosion(self,img: np.ndarray, SE: np.ndarray) -> np.ndarray:

        width,height=SE.shape
        window_width = width
        window_height = height

        edge_x = window_width // 2
        edge_y = window_height // 2

        output_image = np.zeros(img.shape)

        for x in range(edge_x, img.shape[1] - edge_x):
            for y in range(edge_y, img.shape[0] - edge_y):
                flag = 0
                for fx in range(0, window_width):
                    for fy in range(0, window_height):
                        if SE[fy][fx] == 1 and img[y + fy - edge_y][x + fx - edge_x] != 1:
                            flag = 1
                            break
                    if flag == 1:
                        break
                    if flag == 0:
                        output_image[y][x] = 1
        return output_image

    def dilation(self,img: np.ndarray, SE: np.ndarray) -> np.ndarray:
        width,height=SE.shape
        window_width = width
        window_height = height

        edge_x = window_width // 2
        edge_y = window_height // 2
        
        output_image_dialation = np.zeros(img.shape)

        for x in range(edge_x, img.shape[1] - edge_x):
            for y in range(edge_y, img.shape[0] - edge_y):
                flag = 0
                for fx in range(0, window_width):
                    for fy in range(0, window_height):
                        if SE[fy][fx] == 1 and img[y + fy - edge_y][x + fx - edge_x] == 1:
                            flag = 1
                            break
                    if flag == 1:
                        break
                if flag == 1:
                    output_image_dialation[y][x] = 1
        return output_image_dialation

    def opening(self,img: np.ndarray, SE: np.ndarray) -> np.ndarray:
        eroded_img = self.erosion(img, SE)
        dialated_img = self.dilation(eroded_img, SE)
        return dialated_img

    '''utility function to use customized hog'''
    def HOGFeatureExtraction(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.uint8)
        features = compute_hog_features(img)
        return features

#################################### PARAELIZED FUNCTIONS ####################################

    '''This is the parallelized version of noise removal'''
    def gaussianParallelized(self, img: np.ndarray, sigma: float, num_threads: int=2):
        global result
        result = []
        result_lock = threading.Lock()

        threads = []

        for _ in range(num_threads):
            thread = threading.Thread(target=self.gaussian, args=(img, sigma, result_lock, result))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return result[0]
    

    '''Parallelized version of local thresholding'''
    def localThresholdingSegment(self, image: np.ndarray, block_size: int, offset: float) -> np.ndarray:
        image = image * 255
        binary_image = np.zeros(image.shape, dtype=bool)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                start_row = max(0, i - block_size // 2)
                end_row = min(image.shape[0], i + block_size // 2 + 1)
                start_col = max(0, j - block_size // 2)
                end_col = min(image.shape[1], j + block_size // 2 + 1)

                local_mean = np.mean(image[start_row:end_row, start_col:end_col])

                local_threshold = local_mean - offset

                binary_image[i, j] = image[i, j] < local_threshold

        return binary_image


    '''Parallelized version of local thresholding'''
    def localThresholdingParallelized(self, image: np.ndarray, block_size: int, offset: float) -> np.ndarray:
        r, c = image.shape
        image_sections = [image[0:r//2, 0:c//2],  image[0:r//2, c//2:c], image[r//2:r, 0:c//2], image[r//2:r, c//2:c]]
        
        threads = list()
        sections_list = [0] * len(image_sections) 
        for i in range(len(image_sections)):
            thread = threading.Thread(target=self.localThresholdingImg, args=(image_sections[i], block_size, offset, sections_list, i, ))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
        
        
        image[0:r//2, 0:c//2] = sections_list[0]
        image[0:r//2, c//2:c] = sections_list[1]
        image[r//2:r, 0:c//2] = sections_list[2]
        image[r//2:r, c//2:c] = sections_list[3]
        return image

    '''Parallelized version of local thresholding'''
    def localThresholdingImg(self, image: np.ndarray, block_size: int, offset: float, sections_list: list[np.ndarray], index) -> None:
        segmented_image = self.localThresholdingSegment(image, block_size, offset)
        
        sections_list[index] = segmented_image
    
    '''Parallel version of thinning'''
    def thinningCustomizedParallelized(self, img: np.ndarray, max_num_iter: int=None, num_threads: int=2):
        skel = np.asarray(img, dtype=bool).astype(np.uint8)
        mask = np.array([[8, 4, 2],
                        [16, 0, 1],
                        [32, 64, 128]], dtype=np.uint8)

        max_num_iter = max_num_iter or np.inf

        global result
        result = []
        result_lock = threading.Lock()

        threads = []

        for _ in range(num_threads):
            thread = threading.Thread(target=self.thinningCustomized, args=(img, skel.copy(), mask, self.G123_LUT, max_num_iter, result_lock))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return result[0]
    
    '''utility function to use customized hog parallel version'''
    def HOGFeatureExtractionParallelized(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.uint8)
        features = compute_hog_features_parallel(img)
        return features

#############################################################################################################

    def preprocess_parallel(self, img: np.ndarray) -> np.ndarray:
        gray_img = self.img2gray(img)
        clean_img = self.gaussianParallelized(gray_img, 1.2)
        enhanced_img = self.contrastEnhancemet(clean_img)
        binarized_img = self.localThresholdingParallelized(enhanced_img, 25, 0.1)
        skew_corrected_img = self.skewCorrection(binarized_img)
        skeletonized_img = self.thinningCustomizedParallelized(skew_corrected_img)
        return skeletonized_img

    def preproccess_from_scratch(self, img: np.ndarray) -> np.ndarray:
        gray_img = self.img2gray(img)
        clean_img = self.gaussian(gray_img, 1.2)
        enhanced_img = self.contrastEnhancemet(clean_img)
        binarized_img = self.binarize(enhanced_img)
        skew_corrected_img = self.skewCorrection(binarized_img)
        skeletonized_img = self.thinningCustomized(skew_corrected_img)
        return skeletonized_img


    def preprocess(self, img: np.ndarray) -> np.ndarray:
        gray_img = self.img2gray(img)
        clean_img = self.gaussian(gray_img, 1.2)
        enhanced_img = self.contrastEnhancemet(clean_img)
        binarized_img = self.binarize(enhanced_img)
        skew_corrected_img = self.skewCorrection(binarized_img)
        skeletonized_img = self.thinningCustomized(skew_corrected_img)
        show_images([skeletonized_img])
        return skeletonized_img