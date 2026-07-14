#include "katas.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace lesson23 {
float iou(const Box& a, const Box& b) {
    const float intersection = std::max(0.0F, std::min(a.x2,b.x2)-std::max(a.x1,b.x1)) *
                               std::max(0.0F, std::min(a.y2,b.y2)-std::max(a.y1,b.y1));
    const float area_a = std::max(0.0F,a.x2-a.x1)*std::max(0.0F,a.y2-a.y1);
    const float area_b = std::max(0.0F,b.x2-b.x1)*std::max(0.0F,b.y2-b.y1);
    const float denominator = area_a + area_b - intersection;
    return denominator > 0.0F ? intersection / denominator : 0.0F;
}
std::vector<Box> nms(std::vector<Box> boxes, float threshold) {
    if (threshold < 0.0F || threshold > 1.0F) throw std::invalid_argument("NMS threshold must be in [0,1]");
    std::stable_sort(boxes.begin(), boxes.end(), [](const Box& a,const Box& b){return a.score>b.score;});
    std::vector<Box> kept;
    for (const auto& candidate : boxes) {
        const bool suppressed = std::any_of(kept.begin(),kept.end(),[&](const Box& selected){
            return selected.class_id == candidate.class_id && iou(selected,candidate) > threshold;});
        if (!suppressed) kept.push_back(candidate);
    }
    return kept;
}
std::vector<std::size_t> top_k_indices(const std::vector<float>& scores, std::size_t k) {
    k = std::min(k, scores.size()); std::vector<std::size_t> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin()+k, indices.end(), [&](auto a,auto b){
        return scores[a] == scores[b] ? a < b : scores[a] > scores[b];});
    indices.resize(k); return indices;
}
float bilinear_sample(const Image& image, float x, float y, int channel) {
    if (image.width <= 0 || image.height <= 0 || image.channels <= 0 || channel < 0 || channel >= image.channels ||
        image.data_hwc.size() != static_cast<std::size_t>(image.width*image.height*image.channels))
        throw std::invalid_argument("invalid image or channel");
    x=std::clamp(x,0.0F,static_cast<float>(image.width-1)); y=std::clamp(y,0.0F,static_cast<float>(image.height-1));
    const int x0=static_cast<int>(std::floor(x)), y0=static_cast<int>(std::floor(y));
    const int x1=std::min(x0+1,image.width-1), y1=std::min(y0+1,image.height-1);
    auto at=[&](int px,int py){return image.data_hwc[(py*image.width+px)*image.channels+channel];};
    const float dx=x-x0,dy=y-y0;
    return (1-dy)*((1-dx)*at(x0,y0)+dx*at(x1,y0))+dy*((1-dx)*at(x0,y1)+dx*at(x1,y1));
}
std::vector<float> hwc_to_chw(const Image& image) {
    if (image.width<=0||image.height<=0||image.channels<=0||
        image.data_hwc.size()!=static_cast<std::size_t>(image.width*image.height*image.channels))
        throw std::invalid_argument("invalid HWC image");
    const std::size_t plane=static_cast<std::size_t>(image.width)*image.height;
    std::vector<float> output(image.data_hwc.size());
    for(int y=0;y<image.height;++y) for(int x=0;x<image.width;++x) for(int c=0;c<image.channels;++c)
        output[c*plane+y*image.width+x]=image.data_hwc[(y*image.width+x)*image.channels+c];
    return output;
}
Box map_from_letterbox(const Box& box,const Letterbox& t) {
    if(t.scale<=0||t.original_width<=0||t.original_height<=0) throw std::invalid_argument("invalid letterbox transform");
    Box out=box; out.x1=std::clamp((box.x1-t.pad_x)/t.scale,0.0F,static_cast<float>(t.original_width));
    out.x2=std::clamp((box.x2-t.pad_x)/t.scale,0.0F,static_cast<float>(t.original_width));
    out.y1=std::clamp((box.y1-t.pad_y)/t.scale,0.0F,static_cast<float>(t.original_height));
    out.y2=std::clamp((box.y2-t.pad_y)/t.scale,0.0F,static_cast<float>(t.original_height)); return out;
}
}  // namespace lesson23
