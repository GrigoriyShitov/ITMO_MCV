import torch
from torchvision.models.alexnet import alexnet
from torchvision import transforms
from PIL import Image
import time

# Функция для предобработки изображения
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Изменение размера изображения до 224x224
        transforms.ToTensor(),  # Преобразование в тензор
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация
    ])
    image = Image.open(image_path)  # Открываем изображение
    image = transform(image).unsqueeze(0)  # Преобразуем и добавляем batch размер
    return image.cuda()

# Функция для получения предсказания
def predict(image_path):
    # Загружаем и подготавливаем изображение
    image = preprocess_image(image_path)

    # Прогоняем изображение через модель
    timest = time.time()
    with torch.no_grad():  # Отключаем автоматическое вычисление градиентов
        output = model(image)
    print(f"Inference time: {time.time()-timest:.4f} seconds")

    # Получаем индекс класса с максимальной вероятностью
    _, predicted_class = torch.max(output, 1)

    # Возвращаем предсказание
    return predicted_class.item()

# Загружаем модель
timest = time.time()
model = alexnet(pretrained=True).eval().cuda()
print("Model load time: {:.4f} seconds".format(time.time()-timest))

# Пример использования
image_path = "/content/imgs/cat.jpeg"  # Укажите путь к вашему изображению
predicted_class = predict(image_path)
print(f"Predicted class index: {predicted_class}")
