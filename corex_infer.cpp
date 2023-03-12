#include <iostream>
#include <fstream>

#include<pthread.h>

#include "corex_infer.h"
using namespace std;

#include "json/json.h"

#include "spdlog/spdlog.h"
//#include "spdlog/sinks/stdout_color_sinks.h"

#include <chrono>
#include <thread>

namespace corex {
auto console2 = spdlog::stdout_color_mt("dummy_worker");

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

class dummy_engine: public engine {
private:
        bool running;
        std::vector<thread> agents;
        channel *in_ch, *out_ch;

public:
        dummy_engine(channel *in, channel *out, const int parallel = 1);
        ~dummy_engine();

public:
        channel &get_in();
        channel &get_out();
private:
        static void *run(void *);
};

dummy_engine::dummy_engine(channel *in, channel *out, const int parallel) {
        running = true;
        in_ch = in;
        out_ch = out;
        // dummy_engine::run(this);
        for (int i = 0; i < parallel; i++) {
                pthread_t pt;
                pthread_create(&pt, NULL, run, this);
                // agents[i] = thread(run, this);
        }
}

dummy_engine::~dummy_engine() {
        running = false;
}

channel &dummy_engine::get_in() {
        return *in_ch;
}

channel &dummy_engine::get_out() {
        return *out_ch;
}

void *dummy_engine::run(void *p) {

        dummy_engine *that = (dummy_engine *)p;
        // spdlog::get("dummy_worker")->info("dummy_worker is running!");
        while (that->running) {
                // wait for incomming items
                msg_t_ msg;
                that->get_in().get(msg);

                spdlog::get("dummy_worker")->info("dummy_worker has an item to process");
                spdlog::get("dummy_worker")->info("item info seq:{} val:[{},,,]", msg.data->seq, msg.data->data[0]);
                std::this_thread::sleep_for(std::chrono::seconds(3));
                spdlog::get("dummy_worker")->info("takes some time");

                // pass through
                // if (msg.data) delete []msg.data;
                msg_t_ reply = msg;
                that->get_out().put(reply);

                spdlog::get("dummy_worker")->info("dummy_worker finishes the computation");
        }
        pthread_exit(NULL);
}

class tdnn_engine: public engine {
private:
        std::vector<thread> agents;
        channel *in_ch, *out_ch;

};



class creator {
public:
        creator() {}
        virtual ~creator() {};
public:
        virtual engine* engineMethod() const = 0;
};

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
                auto num = eles[i]["tasks_num"].asInt();
                spdlog::get("corex_infer")->info("model name {}", name);
                spdlog::get("corex_infer")->info("model path {}", eles[i]["path"].asString());
                spdlog::get("corex_infer")->info("tasks num {}", num);
                spdlog::get("corex_infer")->info("batch size {}", eles[i]["batch_size"].asInt());

                // TODO: use factory design pattern later
                if (name == "dummy") {
                        spdlog::get("corex_infer")->info("register dummy engine");
                        channel *in = new channel("dummy");

                        dummy_engine *eng = new dummy_engine(in, out, num);
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
