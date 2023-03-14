
#include <iostream>
#include <vector>

#include <chrono>
#include <thread>

#include "spdlog/spdlog.h"
//#include "spdlog/sinks/stdout_color_sinks.h"

#include "util_infer.h"
#include "corex_infer.h"
using namespace std;

// create color multi threaded logger
auto console = spdlog::stdout_color_mt("console");

// // voice data loading & feature extracting task
// void do_dummy(int id) {
//         spdlog::get("console")->info("capture & prepare voice data");
//         corex::frame_t_ *buffer = (corex::frame_t_ *)new char[sizeof(corex::frame_t_) + 20];
//         buffer->type = 0;
//         buffer->src = id;
//         buffer->len = 20;
//         // enumerate wave files
//         for (int i = 1; i <= 2; i++) {
//                 spdlog::get("console")->info("prepare a wave file");
//                 // prepare wave data batch by batch
//                 for (int j = 1; j <= 3; j++) {
//                         spdlog::get("console")->info("extrace a frame data");
//                         buffer->seq = (j == 3 ? -j : j);
//                         buffer->data[0] = j + 10 * i;

//                         util_infer::run_dummy(buffer);

//                         std::this_thread::sleep_for(std::chrono::seconds(2));

//                 }
//         }
//         spdlog::get("console")->info("<<<<<<< done >>>>>>>>>>");
// }

// // post-processing task for voice recognition evaluation
// void post_analysis(int id) {
//         spdlog::get("console")->info("post_analysis is running");
//         std::this_thread::sleep_for(std::chrono::seconds(2));

//         std::list<corex::frame_t_*> res;

//         int lanes = 1; // the number of methods for any given wave file
//         while (util_infer::get_rep(res, lanes)) {
//                 spdlog::get("console")->info("get all inference results for the same wave file");
//                 // scores
//                 for (auto it = res.begin(); it != res.end(); it++) {
//                         auto ele = *it;
//                         spdlog::get("console")->info("frame data seq:{} val:[{},,,]", ele->seq, ele->data[0]);

//                         // free memory after the last step
//                         delete []ele;
//                 }
//         }
//         spdlog::get("console")->info("post_analysis is quiting");
// }

void func(int th) {
        int32_t len = 60;
        float *input = new float[len];
        for(int i=0;i<len;i++) input[i] = i*0.1f;
        float *output = NULL;

        int lanes = 2; // the number of methods for any given wave file
        util_infer::run_dummy(input, len, th);
        
        // another method
	util_infer::run_dummy(input, len, th);

        // wait for the inference results

        std::this_thread::sleep_for(std::chrono::seconds(2));

        std::list<corex::frame_t_*> res;

        util_infer::get_rep(th, lanes, res);

        spdlog::get("console")->info("get all inference results for the same wave file");
        // scores
        for (auto it = res.begin(); it != res.end(); it++) {
                auto ele = *it;
                spdlog::get("console")->info("frame data src:{} type:{}, seq:{} len:{}", ele->src, ele->type, ele->seq, ele->len);
                spdlog::get("console")->info("val:[{},{},{},]", ele->data[0], ele->data[1], ele->data[2]);

                // free memory after the last step
                delete []ele;
        }
        delete []input;
}

int main() {
        spdlog::get("console")->info("preparing sth");

        const int n = 2;
        util_infer::init("config.json");

        std::vector<thread> threads(n);
        for (int i = 0; i < n; i++) {
                threads[i] = thread(func, i + 1);
        }
        // spawn n threads:
        // for (int i = 0; i < n; i++) {
        //         threads[i] = thread(do_dummy, i + 1);
        // }

        // thread th_cook = thread(post_analysis, 0);

        // for (auto& th : threads) {
        //         th.join();
        // }
        // th_cook.join();

        // spdlog::get("console")->info("done!");

        // wait
        std::cout << "press ctrl-c to quit" << std::endl;
        cin.get();
        return 0;
}
