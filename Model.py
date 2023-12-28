from commonfunctions import *
from Preprocessor import Preprocessor
from PreprocessorScratch import PreprocessorScratch
class Model:
    def __init__(self):
        self.real_dataset = list()
        self.forged_dataset = list()
        self.HOG_features = list()
        self.HOG_labels = list()
        self.model = None
        self.preprocessor = Preprocessor()
        self.preprocessorScratch = PreprocessorScratch()

    def readDataset(self, dir: str) -> None:
        dataset = list()
        read_imgs = os.listdir(dir)
        for i in range(len(read_imgs)):
            image = io.imread(dir + read_imgs[i]).astype(np.uint8)
            dataset.append(self.preprocessor.preprocess(image))
        return dataset
    
    def extractFeatures(self, imgs: list[np.ndarray]) -> np.ndarray:
        extracted_features = list()
        for i in range(len(imgs)):
            extracted_features.append(self.preprocessor.HOGFeatureExtractionSkimage(imgs[i]))
        return extracted_features
    
    def train(self, real_dir: str = "", forged_dir: str = "") -> None:
        self.forged_dataset = self.readDataset(forged_dir)
        self.real_dataset = self.readDataset(real_dir)
        
        forged_extracted_features = self.extractFeatures(self.forged_dataset)
        real_extracted_features = self.extractFeatures(self.real_dataset)
        self.HOG_features = forged_extracted_features + real_extracted_features

        HOG_labels_forged = [0 for _ in range(len(forged_extracted_features))]
        HOG_labels_real = [1 for _ in range(len(real_extracted_features))]
        self.HOG_labels = HOG_labels_forged + HOG_labels_real 

        x_train, x_test, y_train, y_test = train_test_split(self.HOG_features, self.HOG_labels, test_size = 0.2, random_state = 42)
        SVMmodel = svm.SVC(kernel='linear')
        SVMmodel.fit(x_train, y_train)

        accuracy = SVMmodel.score(x_test, y_test)
        print("SVM using HOG as feature descriptor.", 'accuracy:', accuracy * 100, '%')
        accuracy = SVMmodel.score(x_train, y_train)
        print("SVM using HOG as feature descriptor. Testing the train accuracy", 'accuracy:', accuracy * 100, '%')

        dump(SVMmodel, 'SVMmodel.pkl')

    def loadModel(self, model_path: str = "") -> None:
        self.model = load(model_path + 'SVMmodel.pkl')
    
    def predict(self, img_path: str = "") -> bool:
        img_to_be_predicted = io.imread(img_path).astype(np.uint8)

        extraced_features = self.extractFeatures([self.preprocessor.preprocess(img_to_be_predicted)])
        
        test_prediction = self.model.predict(extraced_features)
        return test_prediction