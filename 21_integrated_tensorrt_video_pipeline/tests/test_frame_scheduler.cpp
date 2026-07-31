#include "frame_scheduler.hpp"
#include <opencv2/core.hpp>
#include <cstdlib>
#include <memory>
#include <iostream>
#include <set>
#include <stdexcept>
#define CHECK(x) do{if(!(x))throw std::runtime_error("check failed: " #x);}while(false)
namespace {
class FailingSource final : public lesson21::FrameSource {
public:
 bool read(cv::Mat& frame) override {
  if (reads_++ != 0) throw std::runtime_error("expected source failure");
  frame=cv::Mat(8,8,CV_8UC3,cv::Scalar(1,2,3));return true;
 }
 std::string name() const override{return "failing";}
private: int reads_{0};
};
}
int main(){try{cv::Mat a(8,8,CV_8UC3,cv::Scalar(1,2,3)),b(8,8,CV_8UC3,cv::Scalar(4,5,6));lesson21::FrameScheduler scheduler(std::vector<cv::Mat>{a,b},4,2,lesson21::OverloadPolicy::Block);scheduler.start();std::set<std::pair<size_t,std::uint64_t>> ids;while(!scheduler.done()){auto batch=scheduler.next_batch(3,std::chrono::milliseconds(5));CHECK(batch.size()<=3);for(size_t i=0;i<batch.size();++i){CHECK(batch[i].metadata.batch_index==i);ids.emplace(batch[i].metadata.stream_id,batch[i].metadata.frame_id);}}scheduler.stop(false);CHECK(ids.size()==8);CHECK(scheduler.captured()==8);CHECK(scheduler.evicted()==0);CHECK(scheduler.queue_peak()<=2);std::vector<std::unique_ptr<lesson21::FrameSource>> failing;failing.push_back(std::make_unique<FailingSource>());lesson21::FrameScheduler broken(std::move(failing),2,lesson21::OverloadPolicy::Block);broken.start();bool observed=false;for(int attempt=0;attempt<100&&!observed;++attempt){try{broken.next_batch(1,std::chrono::milliseconds(1));}catch(const std::runtime_error&){observed=true;}}broken.stop(true);CHECK(observed);std::cout<<"frame scheduler tests passed\n";return EXIT_SUCCESS;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return EXIT_FAILURE;}}
