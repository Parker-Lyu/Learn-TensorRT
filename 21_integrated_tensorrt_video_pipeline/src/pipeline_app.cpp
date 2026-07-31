#include "config.hpp"
#include "integrated_pipeline.hpp"

#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        return lesson21::run_integrated_pipeline(lesson21::parse_config(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
