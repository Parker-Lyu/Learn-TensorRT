#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

constexpr const char* kDefaultImagePath = "../assets/img.jpeg";

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [--no-display] [image_path]\n";
}

bool graphical_display_is_available() {
#if defined(__linux__)
    // OpenCV HighGUI needs a display server on Linux. Checking before imshow
    // avoids an opaque GUI-backend failure in headless containers and CI.
    const char* const display = std::getenv("DISPLAY");
    const char* const wayland_display = std::getenv("WAYLAND_DISPLAY");
    return (display != nullptr && display[0] != '\0') ||
           (wayland_display != nullptr && wayland_display[0] != '\0');
#else
    return true;
#endif
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        print_usage(argv[0]);
        return 0;
    }

    const bool display_image = argc <= 1 || std::string(argv[1]) != "--no-display";
    const int image_argument_index = display_image ? 1 : 2;
    if (argc > image_argument_index + 1 ||
        (argc > image_argument_index && argv[image_argument_index][0] == '-')) {
        std::cerr << "Invalid command-line arguments.\n";
        print_usage(argv[0]);
        return 2;
    }

    // With no explicit path, use the repository's shared sample image. The
    // README runs the executable from this lesson directory, making this path
    // independent of the build-directory name.
    const std::string image_path =
        argc > image_argument_index ? argv[image_argument_index] : kDefaultImagePath;

    // cv::imread loads an image from disk into a cv::Mat object.
    // IMREAD_COLOR requests an 8-bit, three-channel BGR image.
    cv::Mat image;
    try {
        image = cv::imread(image_path, cv::IMREAD_COLOR);
    } catch (const cv::Exception& error) {
        std::cerr << "OpenCV could not read the image: " << error.what() << '\n';
        return 1;
    }

    // An empty cv::Mat means the image could not be loaded.
    if (image.empty()) {
        std::cerr << "Failed to read image: " << image_path << '\n';
        return 1;
    }

    std::cout << "Loaded image: " << image_path << '\n';
    std::cout << "Width: " << image.cols << ", height: " << image.rows
              << ", channels: " << image.channels() << '\n';

    if (!display_image) {
        std::cout << "Display skipped (--no-display).\n";
        return 0;
    }

    if (!graphical_display_is_available()) {
        std::cerr << "No graphical display was detected. Run with --no-display in a headless "
                     "container or CI environment.\n";
        return 3;
    }

    // cv::imshow opens a window and displays the image.
    // cv::waitKey(0) keeps the window open until any key is pressed.
    try {
        cv::imshow("OpenCV Image Viewer", image);
        static_cast<void>(cv::waitKey(0));
    } catch (const cv::Exception& error) {
        std::cerr << "OpenCV could not display the image: " << error.what() << '\n';
        return 3;
    }

    return 0;
}
