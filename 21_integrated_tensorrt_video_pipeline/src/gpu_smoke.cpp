#include "tensorrt_backend.hpp"
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <iostream>
#include <numeric>
int main(int argc,char**argv){if(argc<3){std::cerr<<"usage: integrated_tensorrt_gpu_smoke ENGINE IMAGE [BATCH]\n";return 2;}try{int n=argc>3?std::stoi(argv[3]):1;if(n<1||n>4)throw std::invalid_argument("batch must be 1..4");cv::Mat image=cv::imread(argv[2]);if(image.empty())throw std::runtime_error("cannot read image");lesson21::TensorRtBackend backend(argv[1],2,{640,640});std::vector<cv::Mat> images(n,image);lesson21::BatchMetadata m;m.batch_id=1;for(int i=0;i<n;++i)m.frames.push_back({0,(std::uint64_t)i,(std::size_t)i,lesson21::Clock::now(),{1,0,0,image.cols,image.rows}});backend.submit(0,images,std::move(m));auto r=backend.collect(0);double sum=std::accumulate(r.output.begin(),r.output.end(),0.0);std::cout<<"backend=tensorrt batch="<<n<<" output_elements="<<r.output.size()<<" checksum="<<sum<<" preprocess_ms="<<r.preprocess_ms<<" inference_ms="<<r.inference_ms<<" d2h_ms="<<r.d2h_ms<<"\n";return 0;}catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<'\n';return 1;}}
