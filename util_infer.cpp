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
bool run_dummy(const corex::frame_t_ *in_data)
{
	spdlog::get("util_infer")->info("dummpy infer");
	corex::inference &infer = get_infer();
	corex::msg_t_ msg;
	size_t len = sizeof(corex::frame_t_) + in_data->len;
	msg.data = (corex::frame_t_ *)new char[len];
	memcpy(msg.data, in_data, len);

	corex::channel *chan = infer.query_channel("dummy");
	chan->put(msg);

	return true;
}

bool get_rep(std::list<corex::frame_t_*> &res, int lanes) {
	static std::list<corex::frame_t_ *> pool;
	std::this_thread::sleep_for(std::chrono::seconds(2));
	int cnt = 0;
	corex::channel *chan = util_infer::get_infer().query_channel("sink", 1);
	if (chan == NULL) {
		std::this_thread::sleep_for(std::chrono::seconds(2));
	}
	while (true) {
		spdlog::get("util_infer")->info("wait for new msg in sink channel");
		corex::msg_t_ msg;
		chan->get(msg);
		pool.push_back(msg.data);
		if (msg.data->seq < 0) { // last frame arrives
			spdlog::get("util_infer")->info("last frame result arrives");
			// merge the parallel results
			res = pool;
			pool.clear();
			break;
		}
	}
	return true;
}

corex::inference &get_infer() {
	return infer_;
}

};
