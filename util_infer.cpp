#include<pthread.h>

#include "util_infer.h"

#include "corex_infer.h"

#include "spdlog/spdlog.h"
//#include "spdlog/sinks/stdout_color_sinks.h"

#include <chrono>
#include <thread>

namespace util_infer {
auto console = spdlog::stdout_color_mt("util_infer");

corex::inference infer_("config.json");

bool init(const char *conf)
{
	// create color multi threaded logger
	spdlog::get("util_infer")->info("init is called");
	return true;
}

// call dummmy model inference
// parameter in_data is input for inference
bool run_dummy(const float *data, int32_t len, int32_t th)
{
	spdlog::get("util_infer")->info("dummpy infer");
	corex::inference &infer = infer_;

	infer.query_engine("dummy")->call(data, len, th);

	return true;
}

bool get_rep(int th, int lanes, std::list<corex::frame_t_*> &pool) {
	int cnt = 0;
	corex::channel *chan = infer_.query_ochannel(th);
	while (cnt < lanes) {
		corex::frame_t_ *msg;
		chan->get(msg);
		pool.push_back(msg);

		if (msg->seq < 0) { // last frame arrives
			cnt++;			// merge the parallel results
		}
		spdlog::get("util_infer")->info("get_rep receives a msg. seq:{} src:{}. total size {}", msg->seq, msg->src, cnt);
	}
	spdlog::get("util_infer")->info("last frame result arrives");

	return true;
}

// corex::inference &get_infer() {
// 	return infer_;
// }

};
