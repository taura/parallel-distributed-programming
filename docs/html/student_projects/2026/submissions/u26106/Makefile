CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -pedantic -Iinclude -MMD -MP

BUILD_DIR := build
TARGET := $(BUILD_DIR)/cornell_path_tracer

SOURCES := $(wildcard src/*.cpp)
OBJECTS := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
DEPS := $(OBJECTS:.o=.d)

.PHONY: all clean rebuild

all: $(TARGET)

$(TARGET): $(OBJECTS) | $(BUILD_DIR)
	$(CXX) $(OBJECTS) -o $@

$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)

rebuild: clean all

-include $(DEPS)
