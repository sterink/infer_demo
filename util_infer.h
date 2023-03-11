#pragma once
#include <list>
namespace corex {
class inference;
struct frame_t_;
};

namespace util_infer {
bool init(const char *conf);
bool run_tdnn(const corex::frame_t_ *in_data);
bool run_dummy(const corex::frame_t_ *in_data);
// collect a collection of results from sink cannel for wave files.
bool get_rep(std::list<corex::frame_t_*> &res, int lanes);

corex::inference &get_infer();

};