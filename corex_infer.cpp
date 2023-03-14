#include <iostream>
#include <fstream>

#include<pthread.h>

#include "corex_infer.h"
using namespace std;

#include "nlohmann/json.hpp"
using json = nlohmann::json;

#include "spdlog/spdlog.h"
//#include "spdlog/sinks/stdout_color_sinks.h"

#include "dummy_engine.h"

#include <chrono>
#include <thread>

namespace corex {
channel::channel(const char *name_) {
        name = name_;
}

bool channel::get(frame_t_ *&pp, bool blocking) {
        spdlog::get("corex_infer")->info("wait for message from channel {}", name);
        pp = queue.pop();
        return true;
}

bool channel::put(frame_t_ *p, bool blocking) {
        spdlog::get("corex_infer")->info("put message into channel {}", name);
        queue.push(p);
        return true;
}

void *daemon(void *p) {
        inference *that = (inference *)p;
        channel *chan = that->query_ochannel(0);
        while (true) {
                corex::frame_t_ *msg;
                chan->get(msg);
                // route msg
                int32_t src = msg->src;
                spdlog::get("corex_infer")->info("message arrives with src {}", src);

                that->query_ochannel(src)->put(msg);
        }
        pthread_exit(NULL);
}

inference::inference(const char* conf) {
        auto console1 = spdlog::stdout_color_mt("corex_infer");
        spdlog::get("corex_infer")->info("inference ctor!");

        channel *out = new channel("sink");
        cout_map[0] = out;

        pthread_t pt;
        pthread_create(&pt, NULL, daemon, this);

        spdlog::get("corex_infer")->info("start to parse conf!");

        ifstream ifs(conf);
        auto obj = json::parse(ifs);

	auto data = obj["models"];
	for (auto& ele : data) {
                auto name = ele["name"];
                // TODO: use factory design pattern later
                if (name == "dummy") {
                        spdlog::get("corex_infer")->info("register dummy engine");
                        channel *in = new channel("dummy");

                        dummy_engine *eng = new dummy_engine(in, out, ele);
                        e_map[name] = eng;

                        cin_map[name] = in;
                }
        }
        spdlog::get("corex_infer")->info("done with inference building!");
}

inference::~inference() {
        spdlog::get("corex_infer")->info("~inference");
}

bool inference::build(const char *conf) {
        return true;
}

engine *inference::query_engine(const char *name) {
        return e_map[std::string(name)];
}

channel *inference::query_ichannel(const char *name) {
        return cin_map[std::string(name)];
}

channel *inference::query_ochannel(int32_t th) {
        static std::mutex m_mutex;
        channel *chan = NULL;
        {
                std::unique_lock<std::mutex> lock(m_mutex);
                chan = cout_map[th];
                if (!chan) {
                        chan = new channel();
                        cout_map[th] = chan;
                }
        }
        return chan;
}
};
