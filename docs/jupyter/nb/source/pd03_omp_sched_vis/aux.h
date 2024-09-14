/* 
 * aux.h --- auxiliary functions for omp_sched_rec.c
 */

static inline unsigned long long rdtscp() {
  unsigned long long u;
  asm volatile ("rdtscp;shlq $32,%%rdx;orq %%rdx,%%rax;movq %%rax,%0":"=q"(u)::"%rax", "%rdx", "%rcx");
  return u;
}

typedef struct {
  union {
    char pad[64];
    struct {
      unsigned long long load;
      unsigned long long begin;
      unsigned long long end;
      pthread_t worker_idx;
      int done;
    };
  };
} work;

static inline double my_rand(unsigned long i, unsigned long j, 
			     unsigned long n, unsigned long prime) {
  return (((i * n + j) * prime) % (n * n)) / (double)(n * n);
}

static inline unsigned long long 
waste_random_time(unsigned long i, unsigned long j, 
		  unsigned long n, 
		  unsigned long long min_cycle, 
		  unsigned long long max_cycle,
		  unsigned long prime) {
  unsigned long long dc 
    = (i > j ? 0 : min_cycle + (max_cycle - min_cycle) * my_rand(i, j, n, prime));
  unsigned long long c0 = rdtscp();
  while (rdtscp() < c0 + dc) ;
  return dc;
}

pthread_t get_thread_num() { 
  return pthread_self(); 
}

static inline void unit_work(unsigned long n, 
			     work * r,
			     unsigned long i, 
			     unsigned long j, 
			     unsigned long long min_cycle, 
			     unsigned long long max_cycle,
			     unsigned long prime) {
  int ok = __sync_bool_compare_and_swap(&r->done, 0, 1);
  assert(ok);
  r->begin = rdtscp();
  r->load = waste_random_time(i, j, n, min_cycle, max_cycle, prime);
  r->end = rdtscp();
  r->worker_idx = get_thread_num();
}


typedef struct {
  unsigned long x;
  unsigned long y;
} point;

typedef struct {
  point p[2];
} rectangle;

void split_hv(rectangle b, rectangle q[2][2]) {
  int i, j, k, d;
  /* for the 2x2 children */
  for (i = 0; i < 2; i++) {
    for (j = 0; j < 2; j++) {
      /* lower left and upper right points */
      for (k = 0; k < 2; k++) {
	/* each dimension (x and y coordinates)  */
	for (d = 0; d < 2; d++) {
	  q[i][j].p[k].x
	    = ((2 - i - k) * b.p[0].x + (i + k) * b.p[1].x) / 2;
	  q[i][j].p[k].y
	    = ((2 - j - k) * b.p[0].y + (j + k) * b.p[1].y) / 2;
	}
      }
    }
  }
}

static inline int 
prime(unsigned long x) {
  unsigned long i;
  for (i = 2; i < x; i++) {
    if (x % i == 0) return 0;
    else if (i * i > x) return 1;
  }
  return 1;
}

static unsigned long 
find_prime(unsigned long x) {
  while (1) {
    if (prime(x)) return x;
    x++;
  }
}

void dump_record(int repeat, unsigned long n, work R[repeat][n][n], const char * file) {
  unsigned long i, j;
  int t;
  FILE * fp = fopen(file, "wb");
  if (!fp) { perror("fopen"); exit(1); }
  for (t = 0; t < repeat; t++) {
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
	work * r = &R[t][i][j];
	fprintf(fp, "%d %lu %lu %llu %llu %llu %lu\n", 
		t, i, j, r->begin, r->end, r->load, r->worker_idx);
      }
    }
  }
  fclose(fp);
}

