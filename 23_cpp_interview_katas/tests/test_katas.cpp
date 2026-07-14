#include "cuda_buffer.hpp"
#include "katas.hpp"
#include <cmath>
#include <cstdlib>
#include <future>
#include <iostream>
#include <stdexcept>

void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(){try{
    using namespace lesson23;
    require(iou({0,0,10,10,0,0},{0,0,10,10,0,0})==1.0F,"identical IoU");
    require(iou({0,0,0,0,0,0},{0,0,1,1,0,0})==0.0F,"degenerate IoU");
    auto kept=nms({{0,0,10,10,.9F,0},{1,1,9,9,.8F,0},{1,1,9,9,.7F,1}},.5F);
    require(kept.size()==2,"class-aware NMS"); require(nms({},.5F).empty(),"empty NMS");
    Image image{2,2,1,{0,10,20,30}}; require(std::abs(bilinear_sample(image,.5F,.5F,0)-15)<1e-6,"bilinear");
    Image rgb{1,2,3,{1,2,3,4,5,6}}; require(hwc_to_chw(rgb)==std::vector<float>({1,4,2,5,3,6}),"reorder");
    auto mapped=map_from_letterbox({10,20,110,220,0,0},{2,10,20,50,100});
    require(mapped.x1==0&&mapped.y1==0&&mapped.x2==50&&mapped.y2==100,"letterbox clamp");
    require(top_k_indices({1,3,3,2},3)==std::vector<std::size_t>({1,2,3}),"top-k stable");
    RingBuffer<int> ring(2); require(ring.push(1)&&ring.push(2)&&!ring.push(3),"ring full");
    require(ring.pop()==1&&ring.push(3)&&ring.pop()==2&&ring.pop()==3&&!ring.pop(),"ring wrap");
    BoundedQueue<int> queue(1); queue.push(7); auto blocked=std::async(std::launch::async,[&]{return queue.push(8);});
    queue.close(); require(!blocked.get(),"close wakes blocked producer"); require(queue.pop()==7&&!queue.pop(),"queue drain");
    CudaBuffer first(64); CudaBuffer second(std::move(first)); require(!first.get()&&second.size()==64,"CUDA move");
    std::cout<<"All C++ katas passed\n"; return EXIT_SUCCESS;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return EXIT_FAILURE;}}
