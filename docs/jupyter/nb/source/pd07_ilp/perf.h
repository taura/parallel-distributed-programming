/**
   @file clock.h
   @brief a small procedure to get CPU/reference cycle
 */

/* these two are Linux-specific.
   make them zero on other OSes */
#if __linux__
#define HAVE_PERF_EVENT 1
#define HAVE_CLOCK_GETTIME 1
#else
#define HAVE_PERF_EVENT 0
#define HAVE_CLOCK_GETTIME 0
#endif

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <pthread.h>

#include <linux/perf_event.h>
#include <asm/unistd.h>

/**
   this is a wrapper to Linux system call perf_event_open
 */
int perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                    int cpu, int group_fd, unsigned long flags) {
  int ret;
  ret = syscall(__NR_perf_event_open, hw_event, pid, cpu,
                group_fd, flags);
  return ret;
}

/**
   @brief a structure encapsulating a performance counter
 */
typedef struct {
  pthread_t tid;                /**< thread ID this is valid for */
  int fd;                       /**< what perf_event_open returned  */
} perf_event_counter_t;

/**
   @brief make a perf_event_counter
   @details 
   perf_event_counter_t t = mk_perf_event_counter();
   long c0 = perf_event_counter_get(t);
      ... do something ...
   long c1 = perf_event_counter_get(t);
   long dc = c1 - c0; <- the number of CPU clocks between c0 and c1
  */
perf_event_counter_t mk_perf_event_counter() {
  pthread_t tid = pthread_self();
  struct perf_event_attr pe;
  memset(&pe, 0, sizeof(struct perf_event_attr));
  pe.type = PERF_TYPE_HARDWARE;
  pe.size = sizeof(struct perf_event_attr);
  //pe.config = PERF_COUNT_HW_INSTRUCTIONS;
  pe.config = PERF_COUNT_HW_CPU_CYCLES;
  pe.disabled = 1;
  pe.exclude_kernel = 1;
  pe.exclude_hv = 1;
  
  int fd = perf_event_open(&pe, 0, -1, -1, 0);
  if (fd == -1) {
    perror("perf_event_open");
  }
  if (fd != -1 && ioctl(fd, PERF_EVENT_IOC_RESET, 0) == -1) {
    perror("ioctl");
    close(fd);
    fd = -1;
  }
  if (fd != -1 && ioctl(fd, PERF_EVENT_IOC_ENABLE, 0) == -1) {
    perror("ioctl");
    close(fd);
    fd = -1;
  }
  if (fd == -1) {
    fprintf(stderr,
            "%s:%d:warning: the environment does not support perf_event."
            " CPU clock cannot be obtained\n", __FILE__, __LINE__);
  }
  perf_event_counter_t cc = { tid, fd };
  return cc;
}

/**
   @brief destroy a cpu clock counter
  */
void perf_event_counter_destroy(perf_event_counter_t cc) {
  if (cc.fd != -1) {
    close(cc.fd);
  }
}

/**
   @brief get CPU clock
  */
long long perf_event_counter_get(perf_event_counter_t cc) {
  pthread_t tid = pthread_self();
  if (tid != cc.tid) {
    fprintf(stderr,
            "%s:%d:perf_event_counter_get: the caller thread (%ld)"
            " is invalid (!= %ld)\n", 
            __FILE__, __LINE__, (long)tid, (long)cc.tid);
    return -1;
  } else {
    long long c;
    if (cc.fd == -1) {
      c = 0; // rdtsc();
    } else {
      ssize_t rd = read(cc.fd, &c, sizeof(long long));
      if (rd == -1) {
        perror("read"); 
        exit(EXIT_FAILURE);
      }
      assert(rd == sizeof(long long));
    }
    return c;
  }
}

