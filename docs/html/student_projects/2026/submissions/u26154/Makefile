CXX := g++
CXXFLAGS := -O2 -Wall -std=c++17

TARGET := app

SRCS := \
	app.cc \
	inference/attention.cc \
	inference/mlp_block.cc \
	inference/model_loader.cc \
	inference/ops.cc \
	inference/orchestrator.cc \
	inference/tokenizer.cc \
	utils/file_utils.cc \
	utils/json_parser.cc \
	utils/safetensors.cc

MODEL_DIR ?= models/EleutherAI_pythia_14m
PROMPT_FILE ?= prompt_text.txt

# Default inference settings
GEN_LEN ?= 100
LAYERS ?= -1
SEED ?= 1234

# Sampling settings
TEMPERATURE ?= 1.0
TOP_K ?= 1
REPETITION_PENALTY ?= 1.0

# Benchmark settings
WARMUP ?= 2
REPEAT ?= 5

.PHONY: all run run-check run-quick benchmark clean help

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

# Normal text generation with sampling
run: $(TARGET)
	./$(TARGET) \
		--model-dir $(MODEL_DIR) \
		--prompt-file $(PROMPT_FILE) \
		--gen-len $(GEN_LEN) \
		--layers $(LAYERS) \
		--warmup 1 \
		--repeat 1 \
		--seed $(SEED) \
		--temperature $(TEMPERATURE) \
		--top-k $(TOP_K) \
		--repetition-penalty $(REPETITION_PENALTY)


# Full-model performance benchmark
benchmark: $(TARGET)
	./$(TARGET) \
		--model-dir $(MODEL_DIR) \
		--prompt-file $(PROMPT_FILE) \
		--gen-len 32 \
		--layers -1 \
		--warmup $(WARMUP) \
		--repeat $(REPEAT) \
		--seed $(SEED) \
		--temperature 1.0 \
		--top-k 1 \
		--repetition-penalty 1.0

clean:
	rm -f $(TARGET)

help:
	@echo "Available commands:"
	@echo "  make              Build the application"
	@echo "  make run          Normal text generation"
	@echo "  make run-check    Deterministic correctness check"
	@echo "  make run-quick    Quick smoke test"
	@echo "  make benchmark    Performance benchmark"
	@echo "  make clean        Remove compiled executable"
