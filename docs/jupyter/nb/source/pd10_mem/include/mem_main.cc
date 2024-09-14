#include <assert.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include "event.h"

#if __AVX512F__
enum { vwidth = 64 };
enum { L = vwidth / sizeof(long) };
typedef long longv __attribute__((vector_size(vwidth)));
#else
#error "__AVX512F__ not defined. forgot to pass -mavx512f?"
#endif

enum { line_size = 64 };

/* make a list of n cells, each of which points
   to s cells after */
long * mk_list(long n, long s) {
  long * a = (long *)aligned_alloc(vwidth, sizeof(long) * n);
  for (long i = 0; i < n; i++) {
    a[i] = -1;
  }
  long k = 0;
  for (long i = 0; i < n; i++) {
    if (a[k] != -1) {
      assert(k == 0);
      break;
    }
    long k_next = (k + s) % n;
    a[k] = k_next;
    k = k_next;
  }
  return a;
}

long count_distinct_lines(long * a, long n) {
  long d = 0;
  long elems_per_line = line_size / sizeof(long);
  for (long i = 0; i < n; i += elems_per_line) {
    for (long j = 0; j < elems_per_line; j++) {
      if (a[i + j] != -1) {
        d++;
        break;
      }
    }
  }
  return d;
}

longv& V(long& p) {
  return *((longv *)&p);
}

long scan(long * a, long n);

struct opts {
  long size;
  long stride;
  long n_accesses;
  const char * events;
  opts() {
    size       = (1 << 15);
    stride     = line_size;
    n_accesses = 100 * 1000 * 1000;
    //events     = strdup("instructions,cycles,L1-dcache-load-misses,l2_lines_in.all,offcore_requests.l3_miss_demand_data_rd");
    events     = strdup("instructions,cycles,L1-dcache-load-misses,cache-misses");
  }
  ~opts() {
    free((void *)events);
  }
};

void usage(char * prog) {
  opts o;
  fprintf(stderr, "usage:\n");
  fprintf(stderr, "  %s [options]\n", prog);
  fprintf(stderr, "options:\n");
  fprintf(stderr, "  -n,--size N (%ld)\n", o.size);
  fprintf(stderr, "  -s,--stride N (%ld)\n", o.stride);
  fprintf(stderr, "  -a,--n-accesses N (%ld)\n", o.n_accesses);
  fprintf(stderr, "  -e,--events ev,ev,ev,.. (%s)\n", o.events);
}

long round_up(long x, long a) {
  x += a - 1;
  x -= x % a;
  return x;
}

opts * parse_cmdline(int argc, char * const * argv, opts * o) {
  static struct option long_options[] = {
    {"size",       required_argument, 0, 'n' },
    {"stride",     required_argument, 0, 's' },
    {"n_accesses", required_argument, 0, 'a' },
    {"events",     required_argument, 0, 'e' },
    {0,            0,                 0,  0 }
  };

  while (1) {
    int option_index = 0;
    int c = getopt_long(argc, argv, "n:s:a:e:",
			long_options, &option_index);
    if (c == -1) break;

    switch (c) {
    case 'n':
      o->size = atol(optarg);
      break;
    case 's':
      o->stride = atol(optarg);
      break;
    case 'a':
      o->n_accesses = atol(optarg);
      break;
    case 'e':
      free((void *)o->events);
      o->events = strdup(optarg);
      break;
    default:
      usage(argv[0]);
      exit(1);
    }
  }
  o->size = round_up(o->size, sizeof(long));
  o->stride = round_up(o->stride, sizeof(long));
  return o;
}

int main(int argc, char ** argv) {
  opts o;
  parse_cmdline(argc, argv, &o);
  long n_elements = o.size / sizeof(long);
  long stride = o.stride / sizeof(long);
  long * a = mk_list(n_elements, stride);
  long d = count_distinct_lines(a, n_elements);
  printf("size            : %ld\n", o.size);
  printf("stride          : %ld\n", o.stride);
  printf("distinct_blocks : %ld\n", d);
  printf("accesses        : %ld\n", o.n_accesses);
  fflush(stdout);
  perf_event_counters_t pc = mk_perf_event_counters(o.events);
  perf_event_values_t v0 = perf_event_counters_get(pc);
  long l = scan(a, o.n_accesses);
  perf_event_values_t v1 = perf_event_counters_get(pc);
  printf("last element   : %ld\n", l);
  for (int i = 0; i < pc.n; i++) {
    printf("%s : %lld\n", pc.events[i], v1.values[i] - v0.values[i]);
  }
  perf_event_counters_destroy(pc);
  assert(l == (o.n_accesses * stride % n_elements));
  printf("OK\n");
  free(a);
  return 0;
}
