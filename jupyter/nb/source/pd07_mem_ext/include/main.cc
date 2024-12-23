#include <assert.h>
#include <err.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
/*** if "cuda" in VER */
#include "cuda_util.h"
/*** elif "omp" in VER */
#if __NVCOMPILER                // NVIDIA nvc++
#else  // Clang
#define __host__
#define __device__
#define __global__
#endif
/*** endif */

/* get current time in nanosecond */
double cur_time() {
  struct timespec ts[1];
  int ok = clock_gettime(CLOCK_REALTIME, ts);
  if (ok == -1) err(1, "clock_gettime");
  return ts->tv_sec + ts->tv_nsec * 1e-9;
}

/* random number generator */
struct prng {
  long sd;
  void seed(long sd) {
    this->sd = sd;
  }
  long gen_randint() {
    long a = 0x5DEECE66Dull;
    long c = 0xB;
    sd = (a * sd + c) & ((1L << 48) - 1);
    long y = sd >> 17;
    return y;
  }
};

/* allocate on the device the main thing will be run */
template<typename T>
T * alloc_dev(size_t n_elems) {
  size_t sz = sizeof(T) * n_elems;
/*** if "cuda" in VER */
  T * b;
  check_cuda_api(cudaMallocManaged((void **)&b, sz));
/*** else */
  T * b;
  if (-1 == posix_memalign((void **)&b, 4096, sz)) err(1, "posix_memalign");
/*** endif */
  return b;
}

template<typename T>
void dealloc_dev(T * b) {
/*** if "cuda" in VER */
  check_cuda_api(cudaFree((void *)b));
/*** else */
  free((void *)b);
/*** endif */
}

/* swap a[i] and a[j] */
void swap(long * a, long i, long j) {
  long ai = a[i];
  long aj = a[j];
  a[i] = aj;
  a[j] = ai;
}

/* shuffle seq = [0,1,2,...,n_cycles*len_cycle-1];
   make sure
   (1) coalese_size consecutive elements are sequential.
   (2) seq[0:n_cycles] = [0,1,...,n_cycles] */
void shuffle(long * seq, long coalese_size,
             long n_cycles, long len_cycle, long seed) {
  assert(n_cycles % coalese_size == 0);
  long m = n_cycles * len_cycle;
  for (long i = 0; i < m; i++) {
    seq[i] = i;
  }
  if (seed >= 0) {
    prng rg;
    rg.seed(seed);
    long n_blocks = m / coalese_size;
    for (long i = n_cycles / coalese_size; i < n_blocks; i++) {
      long j = rg.gen_randint() % (n_blocks - i);
      for (long k = 0; k < coalese_size; k++) {
        swap(seq, i * coalese_size + k, (i + j) * coalese_size + k);
      }
    }
  }
/*** if DBG >= 2 */
#if DBG >= 2
  for (long i = 0; i < m; i++) {
    printf("seq[%ld] = %ld\n", i, seq[i]);
  }
#endif
/*** endif */
}

/* set a[k] to the next element to access */
__host__ __device__
void make_cycle(long * a, long * seq,
                long start_idx, long n_cycles, long len_cycle) {
/*** if DBG >= 1 */
#if DBG >= 1
  printf("make_cycle : a = %p, seq = %p, n_cycles = %ld, len_cycle = %ld\n",
         a, seq, n_cycles, len_cycle);
#endif
/*** endif */
  // a cycle starting from seq[idx] :
  // a[seq[idx]] -> a[seq[idx+n_cycles]] -> a[seq[idx+2*n_cycles]] -> ..
  long m = n_cycles * len_cycle;
  for (long i = 0; i < len_cycle; i++) {
    long cur  = seq[ start_idx +      i  * n_cycles];
    long next = seq[(start_idx + (i + 1) * n_cycles) % m];
/*** if DBG >= 2 */
#if DBG >= 2
    printf("a[%ld] = %ld\n", cur, next);
#endif
/*** endif */
    a[cur] = next;
  }
}

/*** if "cuda" in VER */
__global__
void make_cycles_g(long * a, long * seq, long n_cycles, long len_cycle) {
  long nthreads = get_n_threads();
  for (long idx = get_thread_index(); idx < n_cycles; idx += nthreads) {
    make_cycle(a, seq, idx, n_cycles, len_cycle);
  }
}
/*** endif */

void make_cycles(long * a, long * seq, long m,
                 long n_cycles, long len_cycle, 
                 long n_teams, long n_threads_per_team) {
/*** if "cuda" in VER */
  check_cuda_launch((make_cycles_g<<<n_teams,n_threads_per_team>>>(a, seq, n_cycles, len_cycle)));
/*** else */
#pragma omp target teams distribute parallel for num_teams(n_teams) num_threads(n_threads_per_team) map(tofrom: a[0:m], seq[0:m])
  for (long idx = 0; idx < n_cycles; idx++) {
    make_cycle(a, seq, idx, n_cycles, len_cycle);
  }
/*** endif */
}

/* defined in a separate file (latency.cc or latency_ilp.cc) */
void cycles(long * a, long m, long n, long * end, long n_cycles,
            long n_conc_cycles,
            long n_teams, long n_threads_per_team, int * thread_idx);

struct opts {
  /* number of elements */
  long n_elements;
  /* minimum number of scans */
  double min_scans;
  /* minimum number of accesses */
  long min_accesses;
  /* number of consecutive elements guaranteed to be contiguous */
  long coalese_size;
  long n_cycles;
  long n_conc_cycles;
  long seed;
  opts() {
    n_elements = 1L << 24;
    min_scans = 5.3;
    min_accesses = (1 << 20);
    coalese_size = 1;
    n_cycles = 1;
    n_conc_cycles = 1;
    seed = 123456789012345L;
  }
};

void usage(char * prog) {
  opts o;
  fprintf(stderr, "usage:\n");
  fprintf(stderr, "  %s [options]\n", prog);
  fprintf(stderr, "options:\n");
  fprintf(stderr, "  -m,--n-elements (%ld)\n", o.n_elements);
  fprintf(stderr, "  --min-scans N (%.3f)\n", o.min_scans);
  fprintf(stderr, "  --min-accesses N (%ld)\n", o.min_accesses);
  fprintf(stderr, "  -c,--coalese-size N (%ld)\n", o.coalese_size);
  fprintf(stderr, "  --n-cycles N (%ld)\n", o.n_cycles);
  fprintf(stderr, "  --seed N (%ld)\n", o.seed);
}

opts parse_opts(int argc, char ** argv) {
  static struct option long_options[] = {
    {"n-elements",          required_argument, 0, 'm' },
    {"min-scans",           required_argument, 0, 0 },
    {"min-accesses",        required_argument, 0, 0 },
    {"coalese-size",        required_argument, 0, 0 },
    {"n-cycles",            required_argument, 0, 0 },
    {"n-conc-cycles",       required_argument, 0, 0 },
    {"seed",                required_argument, 0, 0 },
    {0,         0,                 0,  0 }
  };
  opts o;
  while (1) {
    int option_index = 0;
    int c = getopt_long(argc, argv, "m:c:",
			long_options, &option_index);
    if (c == -1) break;

    switch (c) {
    case 'm':
      o.n_elements = atol(optarg);
      break;
    case 0:
      {
        const char * opt_name = long_options[option_index].name;
        if (strcmp(opt_name, "seed") == 0) {
          o.seed = atol(optarg);
        } else if (strcmp(opt_name, "min-scans") == 0) {
          o.min_scans = atof(optarg);
        } else if (strcmp(opt_name, "min-accesses") == 0) {
          o.min_accesses = atol(optarg);
        } else if (strcmp(opt_name, "coalese-size") == 0) {
          o.coalese_size = atol(optarg);
        } else if (strcmp(opt_name, "n-cycles") == 0) {
          o.n_cycles = atol(optarg);
        } else if (strcmp(opt_name, "n-conc-cycles") == 0) {
          o.n_conc_cycles = atol(optarg);
        } else {
          usage(argv[0]);
          exit(1);
        }
        break;
      }
    default:
      usage(argv[0]);
      exit(1);
    }
  }
  return o;
}

long getenv_long(const char * s) {
  char * vs = getenv(s);
  if (!vs) {
    fprintf(stderr, "set environment variable %s\n", s);
    exit(0);
  }
  return atol(vs);
}

/*** if "simd" in VER */
typedef long longv __attribute__((vector_size(64),__may_alias__,aligned(sizeof(long))));
/*** else */
typedef long longv;
/*** endif */

int main(int argc, char ** argv) {
  const long L = sizeof(longv) / sizeof(long);
  long n_teams = getenv_long("OMP_NUM_TEAMS");
  long n_threads_per_team = getenv_long("OMP_NUM_THREADS");
  opts opt = parse_opts(argc, argv);
  long m = opt.n_elements;
  long n_cycles = opt.n_cycles;
  long coalese_size = opt.coalese_size;
  assert(n_cycles % coalese_size == 0);
  assert(coalese_size % L == 0);
  long len_cycle = (m + n_cycles - 1) / n_cycles;
  if (m % n_cycles) {
    fprintf(stderr,
            "WARNING : m (%ld) not divisible by n_cycles (%ld),"
            " rounded up to %ld\n",
            m, n_cycles, len_cycle * n_cycles);
    m = len_cycle * n_cycles;
  }
  printf("n_elements : %ld\n", m);
  size_t sz = sizeof(long) * m;
  printf("sz : %ld bytes\n", sz);
  printf("n_cycles : %ld\n", n_cycles);
  printf("len_cycle : %ld\n", len_cycle);
  double s = opt.min_scans;
  long n = len_cycle * s;
  if (n * n_cycles < opt.min_accesses) {
    n = (opt.min_accesses + n_cycles - 1) / n_cycles;
  }
  printf("n_accesses_per_cycle : %ld\n", n);
  printf("total_accesses : %ld\n", n * n_cycles);
  long n_conc_cycles = opt.n_conc_cycles;
  printf("n_conc_cycles : %ld\n", n_conc_cycles);
  assert(n_cycles % n_conc_cycles == 0);
  printf("coalese_size : %ld\n", coalese_size);
  printf("seed : %ld\n", opt.seed);

  long * seq = alloc_dev<long>(m);
  shuffle(seq, coalese_size, n_cycles, len_cycle, opt.seed);

  long * a = alloc_dev<long>(m);
  double t0 = cur_time();
  make_cycles(a, seq, m, n_cycles, len_cycle, 
              n_teams, n_threads_per_team);
  double t1 = cur_time();
  double dt0 = t1 - t0;
  printf("make_cycles_total : %f sec\n", dt0);
  printf("make_cycles_per_elem : %.1f nsec\n", 1.0e9 * dt0 / m);
  long * end = alloc_dev<long>(n_cycles);
  int * thread_idx = alloc_dev<int>(n_cycles);
  double t2 = cur_time();
  cycles(a, m, n, end, n_cycles, n_conc_cycles,
         n_teams, n_threads_per_team, thread_idx);
  double t3 = cur_time();
  double dt1 = t3 - t2;
  long bytes = sizeof(long) * n * n_cycles;
  double bw = bytes / dt1;
  printf("bytes accessed : %ld bytes\n", bytes);
  printf("time_total : %f sec\n", dt1);
  printf("time_per_access : %.1f nsec/access\n", 1.0e9 * dt1 / (n * n_cycles));
  printf("bw : %.3f GB/sec\n", bw * 1.e-9);
  printf("checking results ... "); fflush(stdout);
  for (long idx = 0; idx < n_cycles; idx++) {
    assert(end[idx] == seq[(idx + n * n_cycles) % m]);
  }
  printf("OK\n");
#if 0
  for (long idx = 0; idx < n_cycles; idx++) {
    printf("idx = %ld by thread %d\n", idx, thread_idx[idx]);
  }
#endif
  dealloc_dev(seq);
  dealloc_dev(end);
  dealloc_dev(a);
  return 0;
}

