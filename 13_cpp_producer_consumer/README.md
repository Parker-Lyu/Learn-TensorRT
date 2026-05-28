# 13 - C++ Producer Consumer

Goal: build the threading pattern used by real camera inference pipelines.

Topics:

- `std::thread`
- `std::mutex`
- `std::condition_variable`
- Thread-safe queue
- Bounded queue
- Producer-consumer pattern
- Backpressure
- Frame dropping strategy
- Graceful shutdown

Acceptance criteria:

- One producer thread reads frames into a bounded queue.
- One consumer thread pops frames and simulates or runs inference.
- The program handles producer FPS higher than consumer FPS.
- The program exits cleanly without deadlock.
- You can explain latency versus throughput trade-offs in queue sizing.
