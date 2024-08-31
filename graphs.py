import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid", font="DejaVu Sans", rc={"axes.labelsize": 14, "axes.titlesize": 16})


def plot_training_history(history, model_name):
    """
    Построение графика изменения потерь (loss) на обучении и валидации по эпохам.

    Аргументы:
        history (History): Объект истории обучения модели, содержащий значения потерь и метрик по эпохам.
        model_name (str): Название модели, которое будет использовано в заголовке графика и имени файла.

    Возвращает:
        None: График сохраняется в файл и не возвращает никаких значений.
    """

    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("Set2")
    plt.plot(history.history['loss'], label='Потери на обучении', color=palette[0], linewidth=2)
    plt.plot(history.history['val_loss'], label='Потери на валидации', color=palette[1], linewidth=2)

    plt.title(f'{model_name} - Потери', fontsize=16, fontweight='bold')
    plt.xlabel('Эпохи', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    model_name_sanitized = model_name.replace(" ", "_")
    file_path = f'img/graphs_img/{model_name_sanitized}_training_history.png'
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'График сохранён в {file_path}')
