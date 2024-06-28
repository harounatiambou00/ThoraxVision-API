import tensorflow as tf
from PIL import Image
import numpy as np
import io

class Model:
    model = None

    def __init__(self):
        try:
            self.model = tf.keras.models.load_model('./thoraxvision_xray_image_classification_model_v2.h5')
        finally:
            print(self.model.input_shape)

    def preprocess_image(self, image):
        if image.mode != "RGB":
            image = image.convert("RGB")  
        image = image.resize((224, 224))
        image = np.array(image)
        image = image / 255.0 
        return np.expand_dims(image, axis=0)

    def preprocess_patient_data(self, patient_data):
        return np.expand_dims(patient_data, axis=0)

    def normalize_patient_data(self, patient_data):
        HEIGHT_MEAN = 170
        HEIGHT_STD = 10
        WEIGHT_MEAN = 70
        WEIGHT_STD = 15
        AGE_MEAN = 50
        AGE_STD = 20

        def normalize(value, mean, std):
            return (value - mean) / std

        normalized_patient_data = patient_data
        #Taille
        normalized_patient_data[6] = normalize(patient_data[6], HEIGHT_MEAN, HEIGHT_STD)
        #Age
        normalized_patient_data[9] = normalize(patient_data[9], AGE_MEAN, AGE_STD)
                #Poid
        normalized_patient_data[12] = normalize(patient_data[12], WEIGHT_MEAN, WEIGHT_STD)
        return np.array(normalized_patient_data)

    def predict(self, image_file, patient_data):
        image = Image.open(io.BytesIO(image_file.read()))
        image = self.preprocess_image(image=image)
        normalized_patient_data = self.normalize_patient_data(patient_data)
        normalized_patient_data = self.preprocess_patient_data(patient_data=normalized_patient_data)
        print("Normalizedatient data : ", patient_data)
        if self.model is not None:
            classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
            prediction = self.model.predict([image, normalized_patient_data])[0]
            print(prediction)
            threshold = 0.1
            predicted_labels = {classes[i]: float(prediction[i]) for i in range(len(classes))}
            chosen_labels = [label for label, prob in predicted_labels.items() if prob >= threshold]
            return chosen_labels
        else:
            return

