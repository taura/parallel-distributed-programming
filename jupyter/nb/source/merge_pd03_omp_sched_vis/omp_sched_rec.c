/* 
 * omp_sched_rec.c --- OpenMP scheduling recorder
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <omp.h>

#include "aux.h"

/* for-loop version */
void omp_work_loop(unsigned long n, work R[n][n],
		   unsigned long long min_cycle, 
		   unsigned long long max_cycle,
		   unsigned long prime) {
#pragma omp for schedule(runtime) collapse(2)
  for (unsigned long i = 0; i < n; i++) {
    for (unsigned long j = 0; j < n; j++) {
      unit_work(n, &R[i][j], i, j, min_cycle, max_cycle, prime);
    }
  }
}

/* taskloop version */
void omp_work_taskloop(unsigned long n, work R[n][n],
                       unsigned long long min_cycle, 
                       unsigned long long max_cycle,
                       unsigned long prime,
                       unsigned long gran) {
#pragma omp taskloop collapse(2) grainsize(gran) untied
  for (unsigned long i = 0; i < n; i++) {
    for (unsigned long j = 0; j < n; j++) {
      unit_work(n, &R[i][j], i, j, min_cycle, max_cycle, prime);
    }
  }
}

/* task version (2D decomposition) */
void omp_work_task(unsigned long n, work R[n][n],
                   rectangle b,
                   unsigned long long min_cycle, 
                   unsigned long long max_cycle,
                   unsigned long prime, 
                   unsigned long gran) {
  point * p = b.p;
  unsigned long np = (p[1].x - p[0].x) * (p[1].y - p[0].y);
  if (np <= gran) {
    unsigned long i, j;
    for (i = p[0].x; i < p[1].x; i++) {
      for (j = p[0].y; j < p[1].y; j++) {
	unit_work(n, &R[i][j], i, j, min_cycle, max_cycle, prime);
      }
    }
  } else {
    rectangle q[2][2];
    split_hv(b, q);
    int i, j;
    for (i = 0; i < 2; i++) {
      for (j = 0; j < 2; j++) {
#pragma omp task firstprivate(i,j)
	omp_work_task(n, R, q[i][j], min_cycle, max_cycle, prime, gran);
      }
    }
#pragma omp taskwait
  }
}

/* choose the appropriate version based on environment variable LB

   LB=taskloop[,gran] --- taskloop (omp_work_taskloop)
   LB=task[,gran]     --- task (omp_work_task)
   LB=any other --- for loop (omp_work_loop)

   gran specifies the granularity, or the number of iterations 
   below which we don't create further tasks.
   the default is 1

   if LB=any other, OMP_SCHEDULE is used to determine 
   schedule clause of the for loop

   e.g.,

   LB=task        ./omp_schedule --- task
   LB=taskloop    ./omp_schedule --- taskloop
   LB=taskloop,10 ./omp_schedule --- taskloop, each task having at least 10 iterations
   LB=task,10     ./omp_schedule --- task, each task having at least 10 iterations
   OMP_SCHEDULE=static ./omp_schedule --- for-loop with schedule(static)
   OMP_SCHEDULE=dynamic ./omp_schedule --- for-loop with schedule(dynamic)
   OMP_SCHEDULE=dynamic,10 ./omp_schedule --- for-loop with schedule(dynamic), each worker fetching 10 iterations at a time
   OMP_SCHEDULE=guided ./omp_schedule --- for-loop with schedule(guided)

   combine it with OMP_NUM_THREADS

*/
void omp_work(unsigned long n, work R[n][n],
              unsigned long long min_cycle, 
              unsigned long long max_cycle,
              unsigned long prime) {
  char * sched = getenv("LB");
  const char * kw_rec = "task";
  size_t kw_rec_len = strlen(kw_rec);
  const char * kw_tl = "taskloop";
  size_t kw_tl_len = strlen(kw_tl);
  if (sched && strncmp(kw_tl, sched, kw_tl_len) == 0) {
#pragma omp master
    {
      /* LB=tl,g */
      unsigned long gran = 1;
      if (sched[kw_tl_len] == ',') {
        gran = atol(sched + kw_tl_len + 1);
        if (!gran) gran = 1;
      }
      printf("taskloop granularity = %lu\n", gran);
      omp_work_taskloop(n, R, min_cycle, max_cycle, prime, gran);
    }
  } else if (sched && strncmp(kw_rec, sched, kw_rec_len) == 0) {
#pragma omp master
    {
      /* LB=rec{,g} */
      rectangle b = { { { 0, 0 }, { n, n } } };
      unsigned long gran = 1;
      if (sched[kw_rec_len] == ',') {
        gran = atol(sched + kw_rec_len + 1);
        if (!gran) gran = 1;
      }
      printf("task granularity=%lu\n", gran);
      omp_work_task(n, R, b, min_cycle, max_cycle, prime, gran);
    }
  } else {
#pragma omp master
    printf("for loop\n");
    omp_work_loop(n, R, min_cycle, max_cycle, prime);
  }
}

int main(int argc, char ** argv) {
  unsigned long n = (argc > 1 ? atol(argv[1]) : 700);
  int repeat      = (argc > 2 ? atoi(argv[2]) : 2);
  char * file     = (argc > 3 ? strdup(argv[3]) : "log.txt");
  unsigned long long 
    min_cycle     = (argc > 4 ? atoll(argv[4]) : 100);
  unsigned long long 
    max_cycle     = (argc > 5 ? (unsigned long long)atoll(argv[5]) : min_cycle * 100L);
  long seed       = (argc > 6 ? atol(argv[6]) : 72743383232329L);

  unsigned short rg[3] = { seed >> 16, seed >> 8, seed };
  unsigned long prime = find_prime(nrand48(rg) / 2);

  work * R_ = calloc(sizeof(work), repeat * n * n);
  work (*R)[n][n] = (work (*)[n][n])R_;

  assert(min_cycle <= max_cycle);

#pragma omp parallel
  {
    int i;
    for (i = 0; i < repeat; i++) {
#pragma omp barrier
      unsigned long long c0 = rdtscp();
      omp_work(n, R[i], min_cycle, max_cycle, prime);
#pragma omp barrier
      unsigned long long c1 = rdtscp();
      unsigned long long dc = c1 - c0;
#pragma omp master
      {
	printf("%llu clocks / %lu pixels\n", dc, n * n);
	printf("%f clocks/pixel\n", dc / (double)(n * n));
      }
    }
  }
  dump_record(repeat, n, R, file); 
  return 0;
}

