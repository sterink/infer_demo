#pragma once
#include "corex_infer.h"
#include "json/json.h"

#include "spdlog/spdlog.h"
//#include "spdlog/sinks/stdout_color_sinks.h"

namespace corex {
class dummy_engine: public engine {
public:
        dummy_engine(channel *in, channel *out, const Json::Value);
        ~dummy_engine();

public:
        channel &get_in();
        channel &get_out();
private:
        static void *run(void *);
};
};
