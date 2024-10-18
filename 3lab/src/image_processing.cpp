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