import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import MobileNetV2

train_dir = 'data/train'
val_dir = 'data/validation'
test_dir = 'data/test'
img_width, img_height = 150, 150
batch_size = 32
epochs = 10

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()  # Muestra la arquitectura

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    epochs=epochs
)

model.save('modelo_neumonia.h5')

# Gráficas de precisión y pérdida
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Exactitud')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida')
plt.legend()
plt.show()

# Métricas con sklearn
test_generator = val_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int)

print("Informe de Clasificación:")
print(classification_report(test_generator.classes, predicted_classes))

print("Matriz de Confusión:")
print(confusion_matrix(test_generator.classes, predicted_classes))

# Extracción de características con MobileNetV2
mobilenet = MobileNetV2(
    input_shape=(img_width, img_height, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

# Generar características
train_features = mobilenet.predict(train_generator)
test_features = mobilenet.predict(test_generator)

# Entrenar Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(train_features, train_generator.classes)

# Evaluar
y_pred_rf = rf.predict(test_features)
print(classification_report(test_generator.classes, y_pred_rf))