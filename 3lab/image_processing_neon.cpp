#include <arm_neon.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace std;

// Функция сложения двух изображений с использованием NEON
void addImagesWithNeon(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& result) {
    for (int i = 0; i < img1.rows; ++i) {
        for (int j = 0; j < img1.cols; j += 8) { // Обрабатываем по 8 пикселей за раз
            uint8x8_t pix1_r = vld1_u8(&img1.at<cv::Vec3b>(i,j)[0]); // Красный компонент
            uint8x8_t pix1_g = vld1_u8(&img1.at<cv::Vec3b>(i,j)[1]); // Зеленый компонент
            uint8x8_t pix1_b = vld1_u8(&img1.at<cv::Vec3b>(i,j)[2]); // Синий компонент

            uint8x8_t pix2_r = vld1_u8(&img2.at<cv::Vec3b>(i,j)[0]);
            uint8x8_t pix2_g = vld1_u8(&img2.at<cv::Vec3b>(i,j)[1]);
            uint8x8_t pix2_b = vld1_u8(&img2.at<cv::Vec3b>(i,j)[2]);

            // Сложение
            uint8x8_t result_r = vqadd_u8(pix1_r, pix2_r);
            uint8x8_t result_g = vqadd_u8(pix1_g, pix2_g);
            uint8x8_t result_b = vqadd_u8(pix1_b, pix2_b);

            vst1_u8(&result.at<cv::Vec3b>(i,j)[0], result_r);
            vst1_u8(&result.at<cv::Vec3b>(i,j)[1], result_g);
            vst1_u8(&result.at<cv::Vec3b>(i,j)[2], result_b);
        }
    }
}

// Функция вычитания двух изображений с использованием NEON
void subtractImagesWithNeon(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& result) {
    for (int i = 0; i < img1.rows; ++i) {
        for (int j = 0; j < img1.cols; j += 8) { // Обрабатываем по 8 пикселей за раз
            uint8x8_t pix1_r = vld1_u8(&img1.at<cv::Vec3b>(i,j)[0]); // Красный компонент
            uint8x8_t pix1_g = vld1_u8(&img1.at<cv::Vec3b>(i,j)[1]); // Зеленый компонент
            uint8x8_t pix1_b = vld1_u8(&img1.at<cv::Vec3b>(i,j)[2]); // Синий компонент

            uint8x8_t pix2_r = vld1_u8(&img2.at<cv::Vec3b>(i,j)[0]);
            uint8x8_t pix2_g = vld1_u8(&img2.at<cv::Vec3b>(i,j)[1]);
            uint8x8_t pix2_b = vld1_u8(&img2.at<cv::Vec3b>(i,j)[2]);

            // Вычитание
            uint8x8_t result_r = vqsub_u8(pix1_r, pix2_r);
            uint8x8_t result_g = vqsub_u8(pix1_g, pix2_g);
            uint8x8_t result_b = vqsub_u8(pix1_b, pix2_b);

            vst1_u8(&result.at<cv::Vec3b>(i,j)[0], result_r);
            vst1_u8(&result.at<cv::Vec3b>(i,j)[1], result_g);
            vst1_u8(&result.at<cv::Vec3b>(i,j)[2], result_b);
        }
    }
}

int main() {
    // Загрузка изображений
    cv::Mat img1 = cv::imread("Lenna_gs.png");
    cv::Mat img2 = cv::imread("tree_gs.png");
    cv::Mat result(img1.size(), img1.type());

    // Сложение изображений с использованием NEON
    auto start = chrono::high_resolution_clock::now();
    addImagesWithNeon(img1, img2, result);
    auto end = chrono::high_resolution_clock::now();
    cv::imwrite("added_image_neon.jpg", result);
    cout << "Time taken to addition with NEON: " 
         << chrono::duration_cast<chrono::microseconds>(end - start).count() 
         << " microseconds" << endl;

    // Вычитание изображений с использованием NEON
    start = chrono::high_resolution_clock::now();
    subtractImagesWithNeon(img1, img2, result);
    end = chrono::high_resolution_clock::now();
    cv::imwrite("subtracted_image_neon.jpg", result);
    cout << "Time taken to subtraction with NEON: " 
         << chrono::duration_cast<chrono::microseconds>(end - start).count() 
         << " microseconds" << endl;

    return 0;
}
