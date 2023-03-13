#include <iostream>
#include <fstream>

#include<pthread.h>

#include "corex_infer.h"
using namespace std;

#include "json/json.h"

#include "spdlog/spdlog.h"
//#include "spdlog/sinks/stdout_color_sinks.h"

#include "dummy_engine.h"
#include "tdnn_engine.h"

#include <chrono>
#include <thread>

namespace corex {
channel::channel(const char *name_) {
        name = name_;
}

bool channel::get(msg_t_ &pp, bool blocking) {
        spdlog::get("corex_infer")->info("wait for message from channel {}", name);
        pp = queue.pop();
        return true;
}

bool channel::put(msg_t_ p, bool blocking) {
        spdlog::get("corex_infer")->info("put message into channel {}", name);
        queue.push(p);
        return true;
}

inference::inference(const char* conf) {
        auto console1 = spdlog::stdout_color_mt("corex_infer");
        spdlog::get("corex_infer")->info("inference ctor!");

        ifstream ifs(conf);
        Json::Reader reader;
        Json::Value obj;
        reader.parse(ifs, obj); // reader can also read strings

        channel *out = new channel("sink");
        cout_map["sink"] = out;

        spdlog::get("corex_infer")->info("start to parse conf!");
        const Json::Value& eles = obj["models"]; // array of models
        for (int i = 0; i < eles.size(); i++) {
                auto name = eles[i]["name"].asString();
                // TODO: use factory design pattern later
                if (name == "dummy") {
                        spdlog::get("corex_infer")->info("register dummy engine");
                        channel *in = new channel("dummy");

                        dummy_engine *eng = new dummy_engine(in, out, eles[i]);
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

channel *inference::query_channel(const char *name, int type) {
        if (type == 0) {
                return cin_map[std::string(name)];
        } else {
                return cout_map[std::string(name)];
        }
}

};
