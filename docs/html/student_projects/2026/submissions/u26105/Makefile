CXX := g++

CPPFLAGS := -Iinclude
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -Wpedantic -MMD -MP
LDFLAGS :=
LDLIBS :=

TARGET := build/wavelet_demo

LIB_SOURCES := src/bit_vector.cpp src/wavelet_matrix.cpp
LIB_OBJECTS := $(patsubst %.cpp,build/%.o,$(LIB_SOURCES))

MAIN_OBJECT := build/main.o

TEST_BV_OBJECT := build/tests/test_bit_vector.o
TEST_WM_OBJECT := build/tests/test_wavelet_matrix.o

ALL_OBJECTS := $(LIB_OBJECTS) $(MAIN_OBJECT) \
               $(TEST_BV_OBJECT) $(TEST_WM_OBJECT)

DEPS := $(ALL_OBJECTS:.o=.d)

.PHONY: all run clean test test-bit-vector test-matrix

all: $(TARGET)

$(TARGET): $(MAIN_OBJECT) $(LIB_OBJECTS)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

build/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

test: test-bit-vector test-matrix

build/test_bit_vector: $(TEST_BV_OBJECT) build/src/bit_vector.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

test-bit-vector: build/test_bit_vector
	./build/test_bit_vector

build/test_wavelet_matrix: $(TEST_WM_OBJECT) $(LIB_OBJECTS)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

test-matrix: build/test_wavelet_matrix
	./build/test_wavelet_matrix

clean:
	rm -rf build

-include $(DEPS)
