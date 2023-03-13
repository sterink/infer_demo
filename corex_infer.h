#pragma once

#include <iostream>
#include <string>
#include <list>
#include <map>

#include <thread>
#include "TSQueue.hpp"

using namespace std;

namespace corex {

struct frame_t_ {
        int16_t src; // source
        int16_t type; // wave feature type
        int16_t seq; // seq == 0. negative means the last bs
        int16_t len; // data size in float
        float data[0];
};

class channel {
public:
        channel(const char *name_ = "");
public:
        bool get(frame_t_ *&, bool blocking = true);
        bool put(frame_t_ *, bool blocking = true);

private:
        TSQueue<frame_t_ *> queue;
        std::string name;
};

class engine {
protected:
        bool running;
        int ordial;
        std::vector<thread> agents;
        channel *in_ch, *out_ch;
public:
        virtual bool call(const float *, int32_t, int32_t) = 0;

public:
        virtual channel &get_in() = 0;
        virtual channel &get_out() = 0;
};

class inference {
private:
        std::map<std::string, channel*> cin_map;
        std::map<int32_t, channel*> cout_map;
        std::map<std::string, engine*> e_map;

public:
        inference(const char *);
        ~inference();

public:
        channel *query_ichannel(const char *);
        channel *query_ochannel(int32_t);

        engine *query_engine(const char *);

private:
        bool build(const char*);
};

};
