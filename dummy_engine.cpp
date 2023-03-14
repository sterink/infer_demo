#include "dummy_engine.h"

#include "spdlog/spdlog.h"
//#include "spdlog/sinks/stdout_color_sinks.h"

namespace corex {
dummy_engine::dummy_engine(channel *in, channel *out, const json &info) {
        auto console2 = spdlog::stdout_color_mt("dummy_worker");
        running = true;
        in_ch = in;
        out_ch = out;
        ordial = info["ordial"].get<int32_t>();
	std::cout << ordial << std::endl;
        auto num = info["tasks_num"].get<int32_t>();
        auto path = info["path"].get<std::string>();
        auto bs = info["batch_size"].get<int32_t>();
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
       	std::list<corex::frame_t_*> pool;
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

		if(msg->seq<0){ // last frame
			// combine all frames
			int32_t size = reply->len;
			for (auto it = pool.begin(); it != pool.end(); it++) {
				if((*it)->src == reply->src){
				       	size += (*it)->len;
				}
			}
		       	// simply augment the input
			frame_t_ *ans = (frame_t_ *)new float[sizeof(frame_t_)/sizeof(float) + size];
			ans->len = size;
		       	ans->type = that->ordial;
		       	ans->src = msg->src;
		       	ans->seq = -1;
			float *p = ans->data;
		       	std::list<corex::frame_t_*> bar;
			for (auto it = pool.begin(); it != pool.end(); it++) {
				if((*it)->src == reply->src){
				       	memcpy(p, (*it)->data, sizeof(float)*(*it)->len);
				       	delete [](*it);
				}
				else{ // keep other frames
					bar.push_back(*it);
				}
			}
			pool = bar;

			delete []reply;
		       
			that->get_out().put(ans);
		       	spdlog::get("dummy_worker")->info("finishes the calculation");
		}
	       	else{
		       	pool.push_back(reply);
		}

                if (msg) delete []msg;
        }
        pthread_exit(NULL);
}

};
