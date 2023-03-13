#pragma once
#include "corex_infer.h"
#include "json/json.h"

namespace corex {
class dummy_engine: public engine {
public:
        dummy_engine(channel *in, channel *out, const Json::Value);
        ~dummy_engine();

public:
        channel &get_in();
        channel &get_out();

public:
        bool call(const float *, int32_t, int32_t);

private:
        static void *run(void *);
};
};
