#include "batch_layout.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        require(lesson14::input_batch_offset(2, 3, 4, 5) == 120, "input offset mismatch");
        require(lesson14::output_batch_offset(3, {4, 84, 8400}) == 2116800,
                "output offset mismatch");
        const auto input = lesson14::make_batched_input(4, 3, 2, 2);
        require(input.size() == 48, "batched tensor size mismatch");
        require(input[0] != input[12] && input[12] != input[24],
                "sample markers overlap across batch offsets");
        bool threw = false;
        try { (void)lesson14::checked_volume({1, -1, 4}); } catch (...) { threw = true; }
        require(threw, "invalid dynamic dimension was accepted");
        std::cout << "All batch layout tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
