/**
   @file event.h
   @brief a small procedure to get performance event counter
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <pthread.h>

#include <perfmon/pfmlib.h>
#include <perfmon/pfmlib_perf_event.h>
//#include <linux/perf_event.h>
#include <asm/unistd.h>

/**
   @brief a structure encapsulating a performance counter
 */
enum { max_perf_events = 8 };
typedef struct {
  pthread_t tid;                /**< thread ID this is valid for */
  int n;                        /**< number of events  */
  char * events[max_perf_events]; /**< event names  */
  int fds[max_perf_events];     /**< what perf_event_open returned  */
} perf_event_counters_t;

typedef struct {
  long long values[max_perf_events];
} perf_event_values_t;

struct perf_event_attr perf_event_encode(const char * ev) {
  struct perf_event_attr attr;
  pfm_perf_encode_arg_t perf;
  char *fstr = NULL;
  memset(&attr, 0, sizeof(struct perf_event_attr));
  memset(&perf, 0, sizeof(pfm_perf_encode_arg_t));
  attr.size = sizeof(attr);
  perf.attr = &attr;
  perf.fstr = &fstr;
  perf.size = sizeof(perf);
  int ret = pfm_initialize();
  if (ret != PFM_SUCCESS) {
    fprintf(stderr, "%s:%d:perf_event_encode: pfm_initialize failed %s\n",
            __FILE__, __LINE__, pfm_strerror(ret));
    attr.size = 0;
    return attr;
  }
  ret = pfm_get_os_event_encoding(ev, PFM_PLM3, PFM_OS_PERF_EVENT_EXT, &perf);
  if (ret != PFM_SUCCESS) {
    fprintf(stderr, "%s:%d:perf_event_encode: pfm_get_os_event_encoding(\"%s\") failed %s\n",
            __FILE__, __LINE__, ev, pfm_strerror(ret));
    attr.size = 0;
    return attr;
  }
  //printf("fstr = %s\n", fstr);
  assert(attr.size == sizeof(attr));
  return attr;
}

/**
   @brief make a perf_event_counters
   @details 
   perf_event_counters_t t = mk_perf_event_counters();
   long c0 = perf_event_counters_get(t);
      ... do something ...
   long c1 = perf_event_counters_get(t);
   long dc = c1 - c0; <- the number of CPU clocks between c0 and c1
  */
static int mk_perf_event_counter_1(const char * ev) {
  struct perf_event_attr attr = perf_event_encode(ev);
  if (attr.size == 0) {
    exit(EXIT_FAILURE);
  }
  attr.disabled = 1;
  attr.exclude_kernel = 1;
  attr.exclude_hv = 1;
  attr.exclude_idle = 1;
  int fd = perf_event_open(&attr, 0, -1, -1, 0);
  if (fd == -1) {
    perror("perf_event_open");
    exit(EXIT_FAILURE);
  }
  if (ioctl(fd, PERF_EVENT_IOC_RESET, 0) == -1) {
    perror("ioctl");
    exit(EXIT_FAILURE);
  }
  if (ioctl(fd, PERF_EVENT_IOC_ENABLE, 0) == -1) {
    perror("ioctl");
    exit(EXIT_FAILURE);
  }
  return fd;
}

/**
   @brief make a perf_event_counters
   @details 
   perf_event_counters_t t = mk_perf_event_counters();
   long c0 = perf_event_counters_get(t);
      ... do something ...
   long c1 = perf_event_counters_get(t);
   long dc = c1 - c0; <- the number of CPU clocks between c0 and c1
  */
static perf_event_counters_t mk_perf_event_counters(const char * events) {
  perf_event_counters_t ec;
  ec.tid = pthread_self();
  int n = 0;
  char * e = (events ? strdup(events) : 0);
  char * p = e;
  while (p) {
    char * q = strstr(p, ",");
    if (q) { *q = 0; q++; }
    int fd = mk_perf_event_counter_1(p);
    if (fd != -1) {
      ec.events[n] = strdup(p);
      ec.fds[n] = fd;
      n++;
    }
    p = q;
  }
  ec.n = n;
  free(e);
  return ec;
}

/**
   @brief destroy a cpu clock counter
  */
static void perf_event_counters_destroy(perf_event_counters_t ec) {
  for (int i = 0; i < ec.n; i++) {
    close(ec.fds[i]);
  }
}

/**
   @brief get a specified counters
  */
static long long perf_event_counters_get_i(perf_event_counters_t ec, int i) __attribute__((unused));
static long long perf_event_counters_get_i(perf_event_counters_t ec, int i) {
  pthread_t tid = pthread_self();
  if (tid != ec.tid) {
    fprintf(stderr,
            "%s:%d:perf_event_counters_get: the caller thread (%ld)"
            " is invalid (!= %ld)\n", 
            __FILE__, __LINE__, (long)tid, (long)ec.tid);
    return -1;
  } else if (i < ec.n) {
    long long c;
    ssize_t rd = read(ec.fds[i], &c, sizeof(long long));
    if (rd == -1) {
      perror("read"); 
      exit(EXIT_FAILURE);
    }
    assert(rd == sizeof(long long));
    return c;
  } else {
    return -1;
  }
}

/**
   @brief get all counters
  */
static perf_event_values_t perf_event_counters_get(perf_event_counters_t ec) {
  pthread_t tid = pthread_self();
  perf_event_values_t v;
  for (int i = 0; i < ec.n; i++) {
    v.values[i] = -1;
  }
  if (tid != ec.tid) {
    fprintf(stderr,
            "%s:%d:perf_event_counters_get: the caller thread (%ld)"
            " is invalid (!= %ld)\n", 
            __FILE__, __LINE__, (long)tid, (long)ec.tid);
    return v;
  } else {
    for (int i = 0; i < ec.n; i++) {
      ssize_t rd = read(ec.fds[i], &v.values[i], sizeof(long long));
      if (rd == -1) {
        perror("read"); 
        exit(EXIT_FAILURE);
      }
      assert(rd == sizeof(long long));
    }
    return v;
  }
}


#if 0
int main(int argc, char **argv) {
  float a = (argc > 1 ? atof(argv[1]) : 0.9);
  float b = (argc > 2 ? atof(argv[2]) : 1.0);
  float n = (argc > 3 ? atol(argv[3]) : 1000000);
  float x = 1.0;
  cycle_timer_t tm = mk_cycle_timer_for_thread();
  cycle_t t0 = cycle_timer_get(tm);
  for (long i = 0; i < n; i++) {
    x = a * x + b;
  }
  cycle_t t1 = cycle_timer_get(tm);
  printf("%lld (%lld) CPU (REF) cyles %.4f (%.4f) CPU (REF) cycles/iter\n",
         (t1.c - t0.c),
         (t1.r - t0.r),
         (t1.c - t0.c)/(double)n,
         (t1.r - t0.r)/(double)n);
  printf("x = %f\n", x);
  
  close(tm.fd);
}

#endif
