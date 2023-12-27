from commonfunctions import *

class Preprocessor:
    def __init__(self):
        pass
    
    def img2gray(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) > 3:
            img = img[:, :, 0:3, 0]
        if len(img.shape) == 2:
            return img
        else:
            gray_img = rgb2gray(img)
            return gray_img
    
    def noiseRemoval(self, img: np.ndarray) -> np.ndarray:
        '''Implement the gaussian filter'''
        clean_img = gaussian(img, 1.2)
        return clean_img
    
    def contrastEnhancemet(self, img: np.ndarray) -> None:
        enhanced = img.copy()
        low, high = np.percentile(img, [0.2, 99.8])
        enhanced = rescale_intensity(img, in_range=(low, high))
        return enhanced
    
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
    
    def binarize(self, img: np.ndarray) -> np.ndarray:
        block_size = 25
        binarized_img = self.localThresholding(img, block_size, 0.1)
        return binarized_img
    
    def skewCorrection(self, img: np.ndarray) -> np.ndarray:
        skew_corrected_imgs = transform.resize(img, (128, 256), mode='reflect', anti_aliasing=True)
        return skew_corrected_imgs
    
    def HOGFeatureExtractionSkimage(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.uint8)
        features = hog(img, orientations=9, pixels_per_cell=(9, 9), cells_per_block=(3,3), transform_sqrt=True, block_norm="L1")
        return features
    
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        gray_img = self.img2gray(img)
        clean_img = self.noiseRemoval(gray_img)
        enhanced_img = self.contrastEnhancemet(clean_img)
        binarized_img = self.binarize(enhanced_img)
        opened_img = opening(binarized_img, rectangle(5,5))
        skew_corrected_img = self.skewCorrection(opened_img)
        skeletonized_img = skeletonize(skew_corrected_img)
        show_images([opened_img, skeletonized_img])
        return skeletonized_img 