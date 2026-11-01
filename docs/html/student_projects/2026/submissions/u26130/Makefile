CXX = g++
CXXFLAGS = -O2 -std=c++20 -Wall -Wextra
SRCS = main.cc graph.cpp scanner.cpp peeling.cpp fsm.cpp

qec_baseline: $(SRCS) $(wildcard *.hpp)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $@

clean:
	rm -f qec_baseline

.PHONY: clean
