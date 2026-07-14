#include "cuda_buffer.hpp"
#include "katas.hpp"
#include <iostream>
int main(){
    const lesson23::Box a{0,0,10,10,0.9F,0},b{5,5,15,15,0.8F,0};
    lesson23::CudaBuffer buffer(1024);
    std::cout<<"IoU="<<lesson23::iou(a,b)<<" CUDA_bytes="<<buffer.size()<<'\n';
}
