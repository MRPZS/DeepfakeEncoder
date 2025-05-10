import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import json
import numpy as np
from sklearn.metrics import roc_curve

def compute_eer(y_true, y_score):
    fpr, tpr, threshold = roc_curve(y_true, y_score)

    # заменяем np.inf на max + eps
    eps = 1e-3
    threshold[0] = max(threshold[1:]) + eps

    fnr = 1 - tpr
    eer_index = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fnr[eer_index]
    return eer

def date_generator(images_root_path, labels_path):
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)

    image_paths = []
    image_labels = []

    for image_name, label in labels.items():
        image_path = os.path.join(images_root_path, image_name)
        if os.path.exists(image_path): # проверка на случай, если пути не существует
            image_paths.append(image_path)
            image_labels.append(label)
        else:
            return print("Путь указан неверно")

    df = pd.DataFrame({
        'filename' : image_paths,
        'label' : image_labels
        })
    df['label'] = df['label'].map({0 : 'real', 1 : 'fake'})
    df = df.iloc[:4000]
    print(df.tail())

    train = df.sample(frac=0.8, random_state = 42)
    val = df.drop(train.index)

    # создадим генераторы для наших данных
    train_datagen = ImageDataGenerator(rescale = 1./255)
    val_datagen = ImageDataGenerator(rescale = 1./255)

    train_gen = train_datagen.flow_from_dataframe(
        dataframe = train,
        x_col = 'filename',
        y_col = 'label',
        target_size = (224, 224),
        batch_size = 32,
        class_mode = 'binary'
        )

    val_gen = val_datagen.flow_from_dataframe(
        dataframe = val,
        x_col = 'filename',
        y_col = 'label',
        target_size = (224, 224),
        batch_size = 32,
        class_mode = 'binary'
        )

    return train_gen, val_gen

# генерируем данные
images_root_path = r'data\train\images'
labels_path = r'data\train\meta.json'  
train_gen, val_gen = date_generator(images_root_path, labels_path)


# Создание модели
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    layers.Flatten(),    
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')

    ])

# Компиляция модели
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Обучение модели
history = model.fit(
    train_gen,
    batch_size=64,
    epochs=10,
    validation_data=val_gen
)

# Оценка модели
test_loss, test_acc = model.evaluate(val_gen)
print(f'Test accuracy: {test_acc}')
print(f'Test loss: {test_loss}')

# Получаем предсказания модели на валидационных данных
y_pred = model.predict(val_gen)
# Получаем истинные метки
y_true = val_gen.labels

eer = compute_eer(y_true, y_pred)
print(f'Equal Error Rate (EER): {eer:.4f}')

'''
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
'''
