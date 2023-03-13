#include "dummy_engine.h"

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

namespace corex {
dummy_engine::dummy_engine(channel *in, channel *out, const Json::Value info) {
        auto console2 = spdlog::stdout_color_mt("dummy_worker");
        running = true;
        in_ch = in;
        out_ch = out;
        ordial = info["ordial"].asInt();
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

bool dummy_engine::call(const float *input, int32_t len, int32_t th) {
        spdlog::get("dummy_worker")->info("prepare a batch of wave data");
        // prepare wave data batch by batch
        int32_t pos = 0;
        int32_t seq = 0;
        while (pos < len) {
                int32_t slice = 20;
                frame_t_ *buffer = (frame_t_ *)new float[sizeof(frame_t_)/sizeof(float) + slice];
                buffer->type = ordial;
                buffer->src = th;
                buffer->len = slice;
                memcpy(buffer->data, input + pos, slice * sizeof(float));
                pos += slice;
                seq++;
                buffer->seq = seq;
                if (pos >= len) buffer->seq = -seq; // negative means the last batch

                in_ch->put(buffer);
                spdlog::get("dummy_worker")->info("extrace a frame data");
                spdlog::get("dummy_worker")->info("frame info seq:{} src:{} type:{}", buffer->seq, buffer->src, buffer->type);

                std::this_thread::sleep_for(std::chrono::seconds(2));
        }
        return true;
}

void *dummy_engine::run(void *p) {

        dummy_engine *that = (dummy_engine *)p;
        // spdlog::get("dummy_worker")->info("dummy_worker is running!");
        while (that->running) {
                // wait for incomming items
                frame_t_ *msg;
                that->get_in().get(msg);

                spdlog::get("dummy_worker")->info("has an item to process");
                spdlog::get("dummy_worker")->info("item info seq:{} val:[{},,,]", msg->seq, msg->data[0]);
                std::this_thread::sleep_for(std::chrono::seconds(3));

                int len = msg->len;

                // simply augment the input
                frame_t_ *reply = (frame_t_ *)new float[sizeof(frame_t_)/sizeof(float) + len];
                for (int i = 0; i < len; i++) {
                        reply->data[i] = msg->data[i] * that->ordial;
                }
                reply->type = that->ordial;
                reply->src = msg->src;
                reply->seq = msg->seq;
                reply->len = len;

                // pass through
                if (msg) delete []msg;

                that->get_out().put(reply);

                spdlog::get("dummy_worker")->info("finishes the calculation");
        }
        pthread_exit(NULL);
}

};
