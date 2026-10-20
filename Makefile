.PHONY: run-benchmark run-test build-env clean


ROOT_DIR  := $(shell pwd)
IMAGE_NAME := cachecrunch-env

# Usage: make run-benchmark DIR=cpp-experiments/matrix-multiplications/transpose
run-benchmark:
	@if [ -z "$(DIR)" ]; then echo "ERROR: specify DIR=<path>"; exit 1; fi
	@echo "=== Running benchmarks in $(DIR) ==="
	docker run --rm \
		-v "$(ROOT_DIR)/$(DIR):/app" \
		-w /app $(IMAGE_NAME) bash -c "\
		echo '-> Compiling matrices.cpp...' && \
		g++ -O1 matrices.cpp -o matrix && \
		echo '-> Running Naive (i-j-k) Benchmark...' && \
		valgrind --tool=cachegrind --cache-sim=yes ./matrix 1 && \
		echo '-> Running Optimized Benchmark...' && \
		valgrind --tool=cachegrind --cache-sim=yes ./matrix 2"

# Runs test.cpp from root — always mounts full repo
run-test:
	@echo "=== Running tests ==="
	docker run --rm \
		-v "$(ROOT_DIR):/app" \
		-w /app $(IMAGE_NAME) bash -c "\
		echo '-> Compiling...' && \
		g++ -O1 -g -fsanitize=address,undefined test.cpp cpp-experiments/matrix-multiplications/transpose/matrices.cpp \
		cpp-experiments/softmax/simple/softmax.cpp \
		cpp-experiments/normalization/normalization.cpp \
		cpp-experiments/activation/relu.cpp \
		-o test_bin -lm && \
		echo '-> Running tests...' && \
		./test_bin"

# Command 1: Builds the reusable Docker image
build-env:
	@echo "=== Creating temporary Dockerfile ==="
	@echo "FROM ubuntu:latest" > Dockerfile.tmp
	@echo "RUN apt-get update -qq && apt-get install -y gcc g++ valgrind -qq" >> Dockerfile.tmp
	@echo "=== Building Docker image: $(IMAGE_NAME) ==="
	docker build --no-cache -t $(IMAGE_NAME) -f Dockerfile.tmp .
	@rm Dockerfile.tmp
	@echo "=== Image built successfully! ==="


# Run cachegrind across all three implementations and compare
benchmark-all:
	@echo "=== Benchmarking ALL implementations ==="
	@echo ""
	@echo "--- 1. Loop Reordering (naive i-j-k vs optimized i-k-j) ---"
	docker run --rm \
		-v "$(ROOT_DIR)/cpp-experiments/matrix-multiplications/loop-reordering:/app" \
		-w /app $(IMAGE_NAME) bash -c "\
		gcc -O1 matrices.c -o matrix && \
		echo '[naive i-j-k]' && \
		valgrind --tool=cachegrind --cache-sim=yes ./matrix 1 2>&1 | grep -E 'D1|LL|I1' && \
		echo '[optimized i-k-j]' && \
		valgrind --tool=cachegrind --cache-sim=yes ./matrix 2 2>&1 | grep -E 'D1|LL|I1'"
	@echo ""
	@echo "--- 2. Transpose ---"
	docker run --rm \
		-v "$(ROOT_DIR)/cpp-experiments/matrix-multiplications/transpose:/app" \
		-w /app $(IMAGE_NAME) bash -c "\
		g++ -O1 benchmark.cpp matrices.cpp -o matrix && \
		echo '[naive]' && \
		valgrind --tool=cachegrind --cache-sim=yes ./matrix 1 2>&1 | grep -E 'D1|LL|I1' && \
		echo '[transpose]' && \
		valgrind --tool=cachegrind --cache-sim=yes ./matrix 2 2>&1 | grep -E 'D1|LL|I1'"
	@echo ""
	@echo "--- 3. Tiling ---"
	docker run --rm \
		-v "$(ROOT_DIR)/cpp-experiments/matrix-multiplications/tiling:/app" \
		-w /app $(IMAGE_NAME) bash -c "\
		echo '-> Compiling...' && \
		g++ -O1 matrices.cpp -o matrix && \
		echo '[naive]' && \
		valgrind --tool=cachegrind --cache-sim=yes ./matrix 1 2>&1 | grep -E 'D1|LL|I1' && \
		echo '[tiled]' && \
		valgrind --tool=cachegrind --cache-sim=yes ./matrix 2 2>&1 | grep -E 'D1|LL|I1'"

clean:
	@echo "=== Cleaning up ==="
	rm -f $(ROOT_DIR)/test_bin
	find $(ROOT_DIR)/cpp-experiments -name "matrix" -delete
	find $(ROOT_DIR)/cpp-experiments -name "cachegrind.out.*" -delete
