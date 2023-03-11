#pragma once

#include <iostream>
#include <string>
#include <list>
#include <map>

#include "TSQueue.hpp"

using namespace std;

namespace corex {

struct frame_t_ {
        int16_t src; // source
        int16_t type; // wave feature type
        int16_t seq; // seq == 0. negative means the last bs
        int16_t len; // data size in bytes
        float data[0];
};

struct msg_t_ {
        msg_t_(frame_t_ *p=NULL) {data = p;}
        frame_t_ *data;
};

class channel {
public:
        channel(const char *name_ = "");
public:
        bool get(msg_t_ &, bool blocking = true);
        bool put(msg_t_ , bool blocking = true);

private:
        TSQueue<msg_t_> queue;
        std::string name;
};

class engine {
public:

public:
        virtual channel &get_in() = 0;
        virtual channel &get_out() = 0;
};

class inference {
private:
        std::map<std::string, channel*> cin_map, cout_map;
        std::map<std::string, engine*> e_map;

public:
        inference(const char *);
        ~inference();

public:
        channel *query_channel(const char *, int type=0);

private:
        bool build(const char*);
};

};