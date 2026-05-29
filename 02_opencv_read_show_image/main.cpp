#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    // Use the first command line argument as the image path.
    // If no argument is given, use the shared sample image at the repository root.
    const std::string image_path = argc > 1 ? argv[1] : "../assets/dog.webp";

    // cv::imread loads an image from disk into a cv::Mat object.
    // cv::IMREAD_COLOR tells OpenCV to load the image as a 3-channel color image.
    const cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);

    // An empty cv::Mat means the image could not be loaded.
    if (image.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return 1;
    }

    std::cout << "Loaded image: " << image_path << std::endl;
    std::cout << "Width: " << image.cols << ", height: " << image.rows << std::endl;

    // cv::imshow opens a window and displays the image.
    // cv::waitKey(0) keeps the window open until any key is pressed.
    cv::imshow("OpenCV Image Viewer", image);
    cv::waitKey(0);

    return 0;
}
