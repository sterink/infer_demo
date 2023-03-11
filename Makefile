all:
	g++ -g -std=c++11 corex_infer.cpp  demo.cpp  util_infer.cpp `pkg-config --libs --cflags spdlog jsoncpp` -o demo