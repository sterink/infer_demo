#include <string>

#include "tdnn_engine.h"
#include "ixrt.h"
#include "util.h"
#include "data_utils.h"
#include "cuda_runtime_api.h"
#include <iostream>

#include "spdlog/spdlog.h"
//#include "spdlog/sinks/stdout_color_sinks.h"

using namespace std;

namespace corex {
tdnn_engine::tdnn_engine(channel *in, channel *out, const json &info) {
        auto console3 = spdlog::stdout_color_mt("tdnn_worker");
        running = true;
        in_ch = in;
        out_ch = out;
        ordial = info["ordial"].get<int32_t>();
	std::cout << ordial << std::endl;
        auto num = info["tasks_num"].get<int32_t>();
        auto path = info["path"].get<std::string>();
        auto bs = info["batch_size"].get<int32_t>();

        spdlog::get("tdnn_engine")->info("      model path {}", path);
        spdlog::get("tdnn_engine")->info("      tasks num {}", num);
        spdlog::get("tdnn_engine")->info("      batch size {}", bs);

	RuntimeConfig config;
       	string input_name = info["input_name"].get<std::string>();
       	config.device_idx = info["device_idx"].get<int32_t>();
       	config.graph_file = info["graph_file"].get<std::string>();
       	config.weights_file = info["weight_file"].get<std::string>();
       	config.input_shapes = {{input_name, {bs, 37, 400, 1}}};

       	runtime.Init(config);

        // dummy_engine::run(this);
        for (int i = 0; i < num; i++) {
                pthread_t pt;
                pthread_create(&pt, NULL, run, this);
                // agents[i] = thread(run, this);
        }
}

tdnn_engine::~tdnn_engine() {
        running = false;
}

channel &tdnn_engine::get_in() {
        return *in_ch;
}

channel &tdnn_engine::get_out() {
        return *out_ch;
}

bool tdnn_engine::call(const float *input, int32_t len, int32_t th) {
	// TODO: slice wave data into batches
        spdlog::get("tdnn_worker")->info("prepare a batch of wave data");
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
                spdlog::get("tdnn_worker")->info("extrace a frame data");
                spdlog::get("tdnn_worker")->info("frame info seq:{} src:{} type:{}", buffer->seq, buffer->src, buffer->type);

        }
        return true;
}

void *tdnn_engine::run(void *p) {
        tdnn_engine *that = (tdnn_engine *)p;
       	std::list<corex::frame_t_*> pool;

	auto input_map = that->runtime.GetInputShape();
       	auto output_map = that->runtime.GetOutputShape();
        IOBuffers input_io_buffers, output_io_buffers;
       	std::vector<HostBuffer::Ptr> input_buffer_owners;
       	std::vector<HostBuffer::Ptr> output_buffer_owners;
	for (const auto& [name, shape] : input_map) {
	       	input_buffer_owners.emplace_back(std::make_shared<HostBuffer>(shape));
	       	input_io_buffers.emplace_back(name, input_buffer_owners.back()->GetDataPtr(), shape);
       	}
       	for (const auto& [name, shape] : output_map) {
	       	output_buffer_owners.emplace_back(std::make_shared<HostBuffer>(shape));
	       	output_io_buffers.emplace_back(name, output_buffer_owners.back()->GetDataPtr(), shape);
	}

	auto input_dims = input_io_buffers.back().shape.dims;
	auto output_dims = output_io_buffers.back().shape.dims;
       	int64_t input_element = 1, output_element = 1;
       	for (auto d : input_dims) input_element *= d;
       	for (auto d : output_dims) output_element *= d;

        while (that->running) {
                // wait for incomming items
                frame_t_ *msg;
                that->get_in().get(msg);

                spdlog::get("tdnn_worker")->info("has an item to process");
                spdlog::get("tdnn_worker")->info("item info seq:{} val:[{},,,]", msg->seq, msg->data[0]);
                // std::this_thread::sleep_for(std::chrono::seconds(3));
                spdlog::get("tdnn_worker")->info("takes some time");

                // do inference computation

                that->runtime.LoadInput(&input_io_buffers);
                that->runtime.Execute();
                that->runtime.FetchOutput(&output_io_buffers);

                if (msg) delete []msg;

	       	frame_t_ *reply = (frame_t_ *)new float[sizeof(corex::frame_t_)/sizeof(float)+output_element];
                reply->type = that->ordial;
                reply->src = msg->src;
                reply->seq = msg->seq;
                reply->len = output_element;

		memcpy(msg->data, output_buffer_owners.back()->GetDataPtr(), sizeof(float) * output_element);

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
		       	spdlog::get("tdnn_worker")->info("finishes the calculation");
		}
	       	else{
		       	pool.push_back(reply);
		}

                if (msg) delete []msg;
        }
        pthread_exit(NULL);
}
};
