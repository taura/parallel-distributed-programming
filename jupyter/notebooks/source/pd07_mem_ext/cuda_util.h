static void check_cuda_api_(cudaError_t e,
                     const char * msg, const char * file, int line) {
  if (e) {
    fprintf(stderr, "%s:%d:error: %s %s\n",
            file, line, msg, cudaGetErrorString(e));
    exit(1);
  }
}

#define check_cuda_api(e) check_cuda_api_(e, #e, __FILE__, __LINE__)

static void check_cuda_launch_(const char * msg, const char * file, int line) {
  cudaDeviceSynchronize();
  cudaError_t e = cudaGetLastError();
  if (e) {
    fprintf(stderr, "%s:%d:error: %s %s\n",
            file, line, msg, cudaGetErrorString(e));
    exit(1);
  }
}

#define check_cuda_launch(exp) do { exp; check_cuda_launch_(#exp, __FILE__, __LINE__); } while (0)

static __device__ int get_thread_index() {
  unsigned int thread_idx = threadIdx.x;
  unsigned int block_idx = blockIdx.x;
  unsigned int block_dim = gridDim.x;
  int global_idx = thread_idx + block_idx * block_dim;
  return global_idx;
}

static __device__ int get_n_threads() {
  unsigned int grid_dim = gridDim.x;
  unsigned int block_dim = blockDim.x;
  return grid_dim * block_dim;
}
