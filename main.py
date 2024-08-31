import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model

from graphs import plot_training_history


def create_binary_generators(data_dir, target_classes, img_size, batch_size):
    """Создание генераторов данных для бинарной классификации, включая тестовый генератор."""

    def binary_flow_from_directory(directory, classes, img_size, batch_size, subset):
        datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.3)
        generator = datagen.flow_from_directory(
            directory,
            target_size=img_size,
            batch_size=batch_size,
            classes=classes,
            class_mode='binary',
            subset=subset,
            shuffle=True
        )
        return generator

    # Генераторы для обучения и валидации
    train_generator = binary_flow_from_directory(data_dir, target_classes, img_size, batch_size, 'training')
    val_generator = binary_flow_from_directory(data_dir, target_classes, img_size, batch_size, 'validation')

    # Генератор для тестирования (без аугментации)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        classes=target_classes,
        class_mode='binary',
        shuffle=False
    )

    return train_generator, val_generator, test_generator


def create_multiclass_generators(data_dir, img_size, batch_size):
    """Создание генераторов данных для многоклассовой классификации, включая тестовый генератор."""

    # Генератор данных с аугментацией для обучающей выборки
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.3  # 30% данных для валидации
    )

    # Генератор для обучающей выборки
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',  # Режим для многоклассовой классификации
        subset='training'
    )

    # Генератор для валидационной выборки (без аугментации)
    val_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.3)

    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # Генератор для тестовой выборки (без аугментации)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # Не перемешиваем данные, чтобы можно было соотнести результаты
    )

    return train_generator, val_generator, test_generator


def create_binary_classification_model(input_shape):
    """Создание простой модели для бинарной классификации."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def compile_and_train_binary_model(model, train_generator, val_generator, epochs=10):
    """Компиляция и обучение бинарной модели."""
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs
    )
    return model, history


def create_multiclass_classification_model(input_shape, num_classes):
    """Создание модели многоклассового классификатора на основе предобученной модели."""

    # Загрузка предобученной модели ResNet50, исключая верхние слои
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Заморозка базовой модели для сохранения предобученных весов
    base_model.trainable = False

    # Добавление новых слоев
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Добавление глобального слоя усреднения
    x = Dense(1024, activation='relu')(x)  # Полносвязный слой с 1024 нейронами
    predictions = Dense(num_classes, activation='softmax')(
        x)  # Выходной слой с softmax для многоклассовой классификации

    # Создание полной модели
    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def compile_and_train_multiclass_model(model, train_generator, val_generator, epochs=10):
    """Компиляция и обучение многоклассовой модели."""

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs
    )

    return model, history


def evaluate_model(model, test_generator):
    """Оценка модели на тестовом наборе данных."""
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Тестовая потеря: {test_loss}")
    print(f"Тестовая точность: {test_acc}")
    return test_loss, test_acc


def save_model(model, model_name):
    """Сохранение модели в формате Keras."""
    model.save(f"{model_name}.keras")
    print(f"Модель сохранена как {model_name}.keras")


if __name__ == "__main__":
    # Пути к данным и параметры
    data_dir = 'data/'
    categories = ['Pepper_healthy', 'Potato_healthy', 'Tomato_healthy']
    target_classes = ['Tomato_healthy', 'Potato_healthy']  # Имена папок для целевых классов
    input_shape = (256, 256, 3)  # Размер изображений 256x256
    img_size = (256, 256)
    batch_size = 32
    num_classes = 3
    epochs = 10

    # --- Бинарная классификация ---
    target_classes_binary = ['Tomato_healthy', 'Potato_healthy']
    train_generator_binary, val_generator_binary, test_generator_binary = create_binary_generators(
        data_dir, target_classes_binary, img_size, batch_size
    )

    model_binary = create_binary_classification_model(input_shape)
    model_binary, history_binary = compile_and_train_binary_model(model_binary, train_generator_binary,
                                                                  val_generator_binary, epochs)
    plot_training_history(history_binary, 'Бинарная модель')

    # --- Многоклассовая классификация ---
    train_generator_multiclass, val_generator_multiclass, test_generator_multiclass = create_multiclass_generators(
        data_dir, img_size, batch_size
    )

    model_multiclass = create_multiclass_classification_model(input_shape, num_classes)
    model_multiclass, history_multiclass = compile_and_train_multiclass_model(model_multiclass,
                                                                              train_generator_multiclass,
                                                                              val_generator_multiclass, epochs)
    plot_training_history(history_multiclass, 'Многоклассовая модель')

    # --- Оценка и сохранение моделей ---
    print("Оценка бинарной модели:")
    evaluate_model(model_binary, test_generator_binary)

    print("Оценка многоклассовой модели:")
    evaluate_model(model_multiclass, test_generator_multiclass)

    save_model(model_binary, "models/binary_classifier")
    save_model(model_multiclass, "models/multiclass_classifier")
