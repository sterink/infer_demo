#pragma once

#include "corex_infer.h"

#include "nlohmann/json.hpp"
using json = nlohmann::json;

#include "ixrt.h"

using namespace iluvatar::inferrt;

namespace corex {
class tdnn_engine: public engine {
public:
        tdnn_engine(channel *in, channel *out, const json &info);
        ~tdnn_engine();
public:
        channel &get_in();
        channel &get_out();
public:
        bool call(const float *, int32_t, int32_t);

private:
	IxRT runtime;
private:
        static void *run(void *);
};
}
