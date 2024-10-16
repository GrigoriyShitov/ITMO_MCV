ЛР2

**Цель работы:**

Изучить основы оптимизации и векторизации алгоритмов компьютерного
зрения на базе процессорной системы ARM Cortex A57 MPCore + NEON. 

Вариант:

2. Сложение и вычитание двух изображений

# 1. Теоретическая база 

ARM Cortex A57: Это высокопроизводительный процессор с поддержкой 64-битной архитектуры ARMv8-A и SIMD-инструкций NEON, что позволяет эффективно обрабатывать векторные данные.

NEON — это расширение архитектуры ARM, предназначенное для обработки данных с использованием SIMD (Single Instruction, Multiple Data). Это означает, что NEON позволяет выполнять одну и ту же операцию над несколькими данными одновременно, что значительно увеличивает производительность при обработке мультимедиа, графики и других вычислительно интенсивных задач.
Основные характеристики NEON

SIMD Архитектура: 

* NEON поддерживает выполнение одной инструкции над несколькими данными, что позволяет эффективно использовать ресурсы процессора.

* Регистры: NEON использует 128-битные регистры (обозначаемые как Q0-Q31), которые могут содержать несколько 8-, 16-, 32- или 64-битных элементов. Например, один 128-битный регистр может хранить четыре 32-битных числа.

* Поддержка различных типов данных: NEON может работать как с целыми числами, так и с числами с плавающей точкой, что делает его универсальным инструментом для различных приложений.

* Эффективность: Использование NEON может значительно увеличить производительность по сравнению с традиционными методами обработки данных. Например, в некоторых случаях производительность может увеличиваться в 1.6–2.5 раза по сравнению с предыдущими архитектурами ARM12.

# 2. Описание разработанной системы (алгоритмы, принципы работы, архитектура) 

image_processing.cpp
```cpp 
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
using namespace std;
void addImages(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& result) {
    for (int i = 0; i < img1.rows; ++i) {
        for (int j = 0; j < img1.cols; ++j) {
            result.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(img1.at<cv::Vec3b>(i, j)[0] + img2.at<cv::Vec3b>(i, j)[0]);
            result.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(img1.at<cv::Vec3b>(i, j)[1] + img2.at<cv::Vec3b>(i, j)[1]);
            result.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(img1.at<cv::Vec3b>(i, j)[2] + img2.at<cv::Vec3b>(i, j)[2]);
        }
    }
}

void subtractImages(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& result) {
    for (int i = 0; i < img1.rows; ++i) {
        for (int j = 0; j < img1.cols; ++j) {
            result.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(img1.at<cv::Vec3b>(i, j)[0] - img2.at<cv::Vec3b>(i, j)[0]);
            result.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(img1.at<cv::Vec3b>(i, j)[1] - img2.at<cv::Vec3b>(i, j)[1]);
            result.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(img1.at<cv::Vec3b>(i, j)[2] - img2.at<cv::Vec3b>(i, j)[2]);
        }
    }
}

int main() {
    cv::Mat img1 = cv::imread("Lenna_gs.png");
    cv::Mat img2 = cv::imread("tree_gs.png");
    cv::Mat result(img1.size(), img1.type());
    auto start = chrono::high_resolution_clock::now();
    addImages(img1, img2, result);
    auto end = chrono::high_resolution_clock::now();
    cv::imwrite("added_image.jpg", result);
    cout << "Time taken to addition: " 
         << chrono::duration_cast<chrono::microseconds>(end - start).count() 
         << " microseconds" << endl;

    start = chrono::high_resolution_clock::now();
    subtractImages(img1, img2, result);
    end = chrono::high_resolution_clock::now();
    cv::imwrite("subtracted_image.jpg", result);
    cout << "Time taken to subsctraction: " 
         << chrono::duration_cast<chrono::microseconds>(end - start).count() 
         << " microseconds" << endl;
    return 0;
}
```