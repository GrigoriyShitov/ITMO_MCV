import torch
from torch2trt import torch2trt
from torch2trt import TRTModule
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

# Функция для получения предсказания с TensorRT
def predict_with_trt(image_path):
    # Загружаем и подготавливаем изображение
    image = preprocess_image(image_path)

    # Прогоняем изображение через модель TensorRT
    timest = time.time()
    with torch.no_grad():
        output = model_trt(image)
    print(f"Inference time (TRT): {time.time()-timest:.4f} seconds")

    # Получаем индекс класса с максимальной вероятностью
    _, predicted_class = torch.max(output, 1)

    # Возвращаем предсказание
    return predicted_class.item()

# Загрузка предварительно обученной модели AlexNet и её конвертация в TensorRT (один раз)
def convert_model_to_trt():
    model = alexnet(pretrained=True).eval().cuda()

    # Примерное изображение для конвертации
    x1 = torch.ones((1, 3, 224, 224)).cuda()

    # Конвертируем модель в TensorRT
    timest = time.time()
    model_trt = torch2trt(model, [x1])
    print(f"Model conversion time to TRT: {time.time() - timest:.4f} seconds")

    # Сохраняем конвертированную модель
    torch.save(model_trt.state_dict(), 'alexnet_trt.pth')

# Загружаем модель TensorRT
def load_trt_model():
    model_trt = TRTModule()
    timest = time.time()
    model_trt.load_state_dict(torch.load('alexnet_trt.pth'))
    print(f"TRT Model load time: {time.time() - timest:.4f} seconds")
    return model_trt

# Конвертируем модель в TensorRT один раз
# Закомментируйте это, если модель уже конвертирована и сохранена
#convert_model_to_trt()

# Загружаем готовую модель TensorRT
model_trt = load_trt_model()

# Пример использования с изображением
image_path = "/content/imgs/cat.jpeg"  # Укажите путь к вашему изображению
predicted_class_trt = predict_with_trt(image_path)
print(f"Predicted class index (TRT): {predicted_class_trt}")
