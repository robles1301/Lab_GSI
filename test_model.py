import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 150, 150
batch_size = 1  

test_dir = 'data/test'

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False 
)

model = load_model('modelo_neumonia.h5')

loss, accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f'Pérdida en test: {loss:.4f}')
print(f'Exactitud en test: {accuracy:.4f}')

predictions = model.predict(test_generator, steps=test_generator.samples // batch_size)

# print("Predicciones (0: NORMAL, 1: PNEUMONIA):")
# print(predictions)

# Interpretacion de los resultados
predicted_classes = np.where(predictions > 0.5, 1, 0) # Redondear a 0 o 1
# print("Clases predichas:")
# print(predicted_classes)
# Guardar las predicciones en un archivo CSV
predicted_classes = predicted_classes.flatten()
filenames = test_generator.filenames
results = pd.DataFrame({"Filename": filenames, "Predicted Class": predicted_classes})
results.to_csv("predictions.csv", index=False)

# Mostrar un porcentaje de exito:
success_rate = np.mean(predicted_classes == test_generator.classes)
print(f"Tasa de éxito: {success_rate:.2%}")
print("Predicciones guardadas en predictions.csv")
