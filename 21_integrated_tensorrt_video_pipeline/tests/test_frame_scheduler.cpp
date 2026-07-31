#include "frame_scheduler.hpp"
#include <opencv2/core.hpp>
#include <cstdlib>
#include <iostream>
#include <set>
#include <stdexcept>
#define CHECK(x) do{if(!(x))throw std::runtime_error("check failed: " #x);}while(false)
int main(){try{cv::Mat a(8,8,CV_8UC3,cv::Scalar(1,2,3)),b(8,8,CV_8UC3,cv::Scalar(4,5,6));lesson21::FrameScheduler scheduler(std::vector<cv::Mat>{a,b},4,2,lesson21::OverloadPolicy::Block);scheduler.start();std::set<std::pair<size_t,std::uint64_t>> ids;while(!scheduler.done()){auto batch=scheduler.next_batch(3,std::chrono::milliseconds(5));CHECK(batch.size()<=3);for(size_t i=0;i<batch.size();++i){CHECK(batch[i].metadata.batch_index==i);ids.emplace(batch[i].metadata.stream_id,batch[i].metadata.frame_id);}}scheduler.stop(false);CHECK(ids.size()==8);CHECK(scheduler.captured()==8);CHECK(scheduler.evicted()==0);CHECK(scheduler.queue_peak()<=2);std::cout<<"frame scheduler tests passed\n";return EXIT_SUCCESS;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return EXIT_FAILURE;}}
