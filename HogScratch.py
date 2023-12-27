from commonfunctions import  *

def compute_gradient(img: np.ndarray, grad_filter: np.ndarray) -> np.ndarray:

    ts = grad_filter.shape[0]

    new_img = np.zeros((img.shape[0] + ts - 1, img.shape[1] + ts - 1))

    new_img[int((ts-1)/2.0):img.shape[0] + int((ts-1)/2.0), 
            int((ts-1)/2.0):img.shape[1] + int((ts-1)/2.0)] = img

    result = np.zeros((new_img.shape))
    
    for r in np.uint16(np.arange((ts-1)/2.0, img.shape[0]+(ts-1)/2.0)):
        for c in np.uint16(np.arange((ts-1)/2.0, img.shape[1]+(ts-1)/2.0)):
            curr_region = new_img[r-np.uint16((ts-1)/2.0):r+np.uint16((ts-1)/2.0)+1, 
                                  c-np.uint16((ts-1)/2.0):c+np.uint16((ts-1)/2.0)+1]
            curr_result = curr_region * grad_filter
            score = np.sum(curr_result)
            result[r, c] = score

    result_img = result[np.uint16((ts-1)/2.0):result.shape[0]-np.uint16((ts-1)/2.0), 
                        np.uint16((ts-1)/2.0):result.shape[1]-np.uint16((ts-1)/2.0)]

    return result_img

def compute_gradient_magnitude(horizontal_gradient: np.ndarray, vertical_gradient: np.ndarray) -> np.ndarray:

    magnitude=np.zeros([horizontal_gradient.shape[0],horizontal_gradient.shape[1]])
    for x in range(horizontal_gradient.shape[0]):
        for y in range(horizontal_gradient.shape[1]):
            S=np.sqrt(np.power(horizontal_gradient[x][y],2)+np.power(vertical_gradient[x][y],2))
            magnitude[x][y]=S
    return magnitude 

def compute_gradient_direction(horizontal_gradient: np.ndarray, vertical_gradient: np.ndarray) -> np.ndarray:
    
    magnitude=np.zeros([horizontal_gradient.shape[0],horizontal_gradient.shape[1]])
    for x in range(horizontal_gradient.shape[0]):
        for y in range(horizontal_gradient.shape[1]):
            S=np.arctan(vertical_gradient[x][y]/(horizontal_gradient[x][y]+(1e-5)))
            deg=np.rad2deg(S)
            modeled=np.mod(deg,180)
            magnitude[x][y]=modeled
            

    return magnitude 

def find_nearest_bins(curr_direction: float, hist_bins: np.ndarray) -> (int, int):

    diff_arr = np.zeros([len(hist_bins)])
    dict = {}
    if (curr_direction > hist_bins[-1]):
        return (len(hist_bins)-1,0)
    elif(curr_direction < hist_bins[0]):
        return (0,len(hist_bins)-1)
    for i in range(len(hist_bins)):
        diff=np.abs(curr_direction-hist_bins[i])
        diff_arr[i] = diff
        dict[diff]=i
    min1, min2 = np.partition(diff_arr,1)[0:2]
    index1 = dict[min1] 
    index2 = dict[min2]
    if(index1<index2):
        return(index1,index2)
    else:
        return (index2,index1)

def update_histogram_bins(
        HOG_cell_hist: np.ndarray, 
        curr_direction: float, 
        curr_magnitude: float, 
        first_bin_idx: int, 
        second_bin_idx: int, 
        hist_bins: np.ndarray
    ) -> None:

    if (hist_bins[first_bin_idx]==curr_magnitude):
        HOG_cell_hist[first_bin_idx]+=curr_direction
    else:
        if(hist_bins[first_bin_idx]==0):
            perc = (curr_magnitude/hist_bins[second_bin_idx])
            HOG_cell_hist[second_bin_idx]+=(perc*curr_direction)
            HOG_cell_hist[first_bin_idx]+=((1-perc)*curr_direction)
        else:
            perc = (curr_magnitude/hist_bins[first_bin_idx])
            HOG_cell_hist[first_bin_idx]+=(perc*curr_direction)
            HOG_cell_hist[second_bin_idx]+=((1-perc)*curr_direction)


def calculate_histogram_per_cell(
        cell_direction: np.ndarray, 
        cell_magnitude: np.ndarray, 
        hist_bins: np.ndarray
    ) -> np.ndarray:

    histo = np.zeros(len(hist_bins))
    for x in range(cell_direction.shape[0]):
        for y in range(cell_direction.shape[1]):
            i,j=find_nearest_bins(cell_direction[x][y],hist_bins)
            update_histogram_bins(histo,cell_direction[x][y],cell_magnitude[x][y],i,j,hist_bins)
    return histo

'''utility function to be used'''
def compute_hog_features(image: np.ndarray) -> np.ndarray:
    
    image_sqrt = np.sqrt(image.astype(float))

    
    horizontal_mask = np.array([-1, 0, 1])
    vertical_mask = np.array([[-1], [0], [1]])

    # Compute gradients
    horizontal_gradient = compute_gradient(image_sqrt, horizontal_mask)
    vertical_gradient = compute_gradient(image_sqrt, vertical_mask)

    # Compute gradient magnitude and direction
    grad_magnitude = compute_gradient_magnitude(horizontal_gradient, vertical_gradient)
    grad_direction = compute_gradient_direction(horizontal_gradient, vertical_gradient)

    # Define histogram bins
    hist_bins = np.array([10, 30, 50, 70, 90, 110, 130, 150, 170])

    # Compute histograms for each cell
    cells_histogram = np.zeros((grad_magnitude.shape[0] // 9, grad_magnitude.shape[1] // 9, 9))
    for r in range(0, grad_magnitude.shape[0] // 9 * 9, 9):
        for c in range(0, grad_magnitude.shape[1] // 9 * 9, 9):
            cell_direction = grad_direction[r:r+9, c:c+9]
            cell_magnitude = grad_magnitude[r:r+9, c:c+9]
            cells_histogram[r // 9, c // 9] = calculate_histogram_per_cell(
                cell_direction, cell_magnitude, hist_bins)

    # Normalize and concatenate histograms (L1 normalization)
    features_list = []
    for r in range(cells_histogram.shape[0] - 1):
        for c in range(cells_histogram.shape[1] - 1):
            histogram_18x18 = np.reshape(cells_histogram[r:r+2, c:c+2], (36,))
            histogram_18x18_normalized = histogram_18x18 / (np.sum(np.abs(histogram_18x18)) + 1e-5)
            features_list.append(histogram_18x18_normalized)

    return np.concatenate(features_list, axis=0)


def calculate_histogram_per_cell_parallel(result, cell_direction, cell_magnitude, hist_bins, idx, lock):
    histogram = calculate_histogram_per_cell(cell_direction, cell_magnitude, hist_bins)
    with lock:
        result[idx] = histogram  

'''utility function to be used parallelized'''
def compute_hog_features_parallel(image: np.ndarray) -> np.ndarray:
    image_sqrt = np.sqrt(image.astype(float))

    horizontal_mask = np.array([-1, 0, 1])
    vertical_mask = np.array([[-1], [0], [1]])

    horizontal_gradient = compute_gradient(image_sqrt, horizontal_mask)
    vertical_gradient = compute_gradient(image_sqrt, vertical_mask)

    grad_magnitude = compute_gradient_magnitude(horizontal_gradient, vertical_gradient)
    grad_direction = compute_gradient_direction(horizontal_gradient, vertical_gradient)

    hist_bins = np.array([10, 30, 50, 70, 90, 110, 130, 150, 170])

    cells_histogram = np.zeros((grad_magnitude.shape[0] // 9, grad_magnitude.shape[1] // 9, 9))

    threads = []
    results = [None] * len(cells_histogram.ravel())
    idx = 0
    lock = threading.Lock()

    for r in range(0, grad_magnitude.shape[0] // 9 * 9, 9):
        for c in range(0, grad_magnitude.shape[1] // 9 * 9, 9):
            cell_direction = grad_direction[r:r+9, c:c+9]
            cell_magnitude = grad_magnitude[r:r+9, c:c+9]
            thread = threading.Thread(target=calculate_histogram_per_cell_parallel,
                                      args=(results, cell_direction, cell_magnitude, hist_bins, idx, lock))
            threads.append(thread)
            thread.start()
            idx += 1


    for thread in threads:
        thread.join()

    for idx, (r, c) in enumerate(zip(range(0, grad_magnitude.shape[0] // 9 * 9, 9),
                                     range(0, grad_magnitude.shape[1] // 9 * 9, 9))):
        cells_histogram[r // 9, c // 9] = results[idx]

    features_list = []
    for r in range(cells_histogram.shape[0] - 1):
        for c in range(cells_histogram.shape[1] - 1):
            histogram_18x18 = np.reshape(cells_histogram[r:r+2, c:c+2], (36,))
            histogram_18x18_normalized = histogram_18x18 / (np.sum(np.abs(histogram_18x18)) + 1e-5)
            features_list.append(histogram_18x18_normalized)

    return np.concatenate(features_list, axis=0)