CXX=clang++
CXXFLAGS=-std=c++11

all: neural_network neural_network_parallel

neural_network: main.cpp models/nnetwork.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

neural_network_parallel: main_parallel.cpp models/nnetwork_parallel.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ -lpthread
clean:
	rm -f neural_network
	rm -f neural_network_parallel
