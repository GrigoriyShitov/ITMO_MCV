import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

# Загрузка изображения
filename = "dot_and_hole"
img = cv.imread(f'data/{filename}.jpg', cv.IMREAD_GRAYSCALE)

# Параметры для дилатации
kernel = np.ones((3, 3), np.uint8)

# Дилатация с использованием OpenCV
dilated_img_cv = cv.dilate(img, kernel, iterations=3)

# Отображение результатов
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Dilated Image (OpenCV)')
plt.imshow(dilated_img_cv, cmap='gray')
plt.axis('off')


def dilate_manual(image, kernel):
    # Получаем размеры изображения и ядра
    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Создаем выходное изображение с теми же размерами
    dilated_image = np.zeros_like(image)

    # Вычисляем отступы для ядра
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Проходим по каждому пикселю изображения
    for i in range(pad_height, img_height - pad_height):
        for j in range(pad_width, img_width - pad_width):
            # Проверяем область вокруг текущего пикселя
            region = image[i - pad_height:i + pad_height + 1, j - pad_width:j + pad_width + 1]
            # Если хотя бы один пиксель в области равен 255 (белый), устанавливаем пиксель в выходном изображении в 255
            if np.any(region * kernel):
                dilated_image[i, j] = 255

    return dilated_image


# Преобразуем изображение в бинарное (0 и 255)
_, binary_img = cv.threshold(img, 128, 255, cv.THRESH_BINARY)

# Создаем ядро размером 3x3
kernel = np.ones((3, 3), np.uint8)

# Дилатация с использованием собственного алгоритма
dilated_img_manual = dilate_manual(binary_img, kernel)
for i in range(3):
    dilated_img_manual = dilate_manual(dilated_img_manual, kernel)
# Отображение результатов

plt.subplot(1, 3, 3)
plt.title('Dilated Image (Manual)')
plt.imshow(dilated_img_manual, cmap='gray')
plt.axis('off')

plt.show()