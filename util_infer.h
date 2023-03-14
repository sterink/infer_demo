#pragma once
#include <stdint.h>
#include <list>
namespace corex {
class inference;
struct frame_t_;
};

namespace util_infer {
bool init(const char *conf);
bool run_tdnn(const float *data, int32_t len, int32_t th);
bool run_dummy(const float *data, int32_t len, int32_t th);
// collect a collection of results from sink cannel for wave files.
bool get_rep(std::list<corex::frame_t_*> &res, int lanes);

bool get_rep(int th, int lanes, std::list<corex::frame_t_*> &res);
};
