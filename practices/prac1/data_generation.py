import numpy as np
import pandas as pd


def generate_data(num_samples=1000, L=10.0, E=1e10, I=1e-8):
    # Генерация случайных значений для силы и позиции
    F = np.random.uniform(10, 100.0, num_samples)
    x = np.random.uniform(0, L, num_samples)

    # Расчет прогиба балки
    y = (F * x * (L - x)) / (3 * E * I)

    # Создание DataFrame
    data = pd.DataFrame({
        'F': F,
        'x': x,
        'y': y
    })

    return data


# Генерация данных
train_data = generate_data(num_samples=100_000)
val_data = generate_data(num_samples=5_000)
test_data = generate_data(num_samples=5_000)

# Сохранение данных в CSV файлы
train_data.to_csv('train.csv', index=False)
val_data.to_csv('val.csv', index=False)
test_data.to_csv('test.csv', index=False)
