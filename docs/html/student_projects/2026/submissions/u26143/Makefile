CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O3 -Isrc
LDFLAGS :=

TARGET := build/raytracer

SRCS := \
    src/main.cpp \
    src/geometry/sphere.cpp \
    src/geometry/plane.cpp \
    src/shading/shading.cpp \
    src/renderer/framebuffer.cpp \
    src/renderer/renderer.cpp

OBJS := $(patsubst src/%.cpp,build/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

.PHONY: all clean rebuild

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p build
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

build/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

clean:
	rm -rf build

rebuild: clean all

-include $(DEPS)