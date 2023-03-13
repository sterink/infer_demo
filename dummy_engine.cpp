#include "dummy_engine.h"

namespace corex {
dummy_engine::dummy_engine(channel *in, channel *out, const Json::Value info) {
        auto console2 = spdlog::stdout_color_mt("dummy_worker");
        running = true;
        in_ch = in;
        out_ch = out;
        auto num = info["tasks_num"].asInt();
        auto path = info["path"].asString();
        auto bs = info["batch_size"].asInt();
        spdlog::get("dummy_worker")->info("      model path {}", path);
        spdlog::get("dummy_worker")->info("      tasks num {}", num);
        spdlog::get("dummy_worker")->info("      batch size {}", bs);

        // dummy_engine::run(this);
        for (int i = 0; i < num; i++) {
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

};
