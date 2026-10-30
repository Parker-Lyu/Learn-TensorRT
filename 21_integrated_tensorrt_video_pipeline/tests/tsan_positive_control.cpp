// Deliberately incorrect fixtures prove that TSAN instrumentation and reporting
// are active. Never link this executable into a lesson or a production binary.
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::string mode(argv[1]);
    if (mode == "race") {
        int shared = 0;
        std::thread first([&] { shared = 1; });
        std::thread second([&] { shared = 2; });
        first.join();
        second.join();
        std::cout << shared << '\n';
    } else if (mode == "lock-order") {
        std::mutex first, second;
        {
            std::lock_guard<std::mutex> a(first);
            std::lock_guard<std::mutex> b(second);
        }
        {
            std::lock_guard<std::mutex> b(second);
            std::lock_guard<std::mutex> a(first);
        }
    } else {
        return 2;
    }
    return 0;  // The runner requires TSAN to report the intended bug and exit 66.
}
