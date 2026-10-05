<link rel="stylesheet" href="scripts/style.css">

# Parallel and Distributed Programming <br/> Kenjiro Taura {.unnumbered}

# What's new (in the newest-first order)

* <font color=blue>(Posted: Oct. 05, 2026)</font> Today's plan
  * See the page [Getting Started with the Course Environment](html/get_started.html) and submit "Assignment 0"
  * [Introduction](slides/intro.pdf)
  * Receive your account during the class
  * Follow [Getting Started with the Course Environment](html/get_started.html), until you successfully fetch the first notebook (`pd00_intro`)!
* <font color=blue>(Posted: Sep. 28, 2026)</font> 
  * This course uses Miyabi supercomputer and Jupyter server
  * To receive account and set up things, see the page [Getting Started with the Course Environment](html/get_started.html)
* <font color=blue>(Posted: Sep. 28, 2026)</font> The site is up!

# Slides

1. [Introduction](slides/intro.pdf)
1. [OpenMP](slides/openmp.pdf)
1. [CUDA](slides/cuda.pdf)
1. [OpenMP for GPU](slides/openmp_gpu.pdf)
1. [SIMD](slides/simd.pdf)
1. [How to get nearly peak FLOPS (with CPU)](slides/peak_cpu.pdf)
1. [What You Must Know about Memory, Caches, and Shared Memory](slides/memory.pdf)
1. [Analyzing Data Access of Algorithms and How to Make Them Cache-Friendly](slides/cache.pdf)
1. [Divide and Conquer](slides/divide_and_conquer.pdf)
1. [Neural Network Basics](slides/nn.pdf)
1. [Understanding Task Scheduling Algorithms](slides/worksteal.pdf)

# Languages

* All written materials (slides, home pages, etc.) will be in English
* Lectures will be in English

# Hands-on programming exercise

* You will have an access to latest CPU and GPU machines and hands-on experiences on parallel programming
* This year, I emphasize a programming model targetting both CPUs and GPUs (OpenMP + GPU offloading)

# How to get the credit

* <font color="blue">Jan. 6</font> Details released in [a separage page](html/credit.html)

# Topics covered

* Parallel Programming in Practice
  * It's easy! --- a quick and gentle introduction to parallel problem solving
  * Some examples of parallel problem solving and programming
* Taxonomy of parallel machines and programming models
  * What today's machines look like --- parallel computer architecture
    * Distributed memory machines
    * Multi-core / multi-socket nodes
    * SIMD instructions
  * Parallel programming models
    * Finding and expressing parallelism
    * Mapping computation onto compute resources
    * Coordination and communication
    * Examples of parallel programming languages/models
* Understanding performance of parallel programs (and achieving high performance)
  * The maximum performance of your CPU/GPU and why you don't get it for your program?
  * The maximum performance of memory and why you don't get it for your program?
  * How to reason about memory traffic of your programs
  * Provable bounds of greedy schedulers
  * Provable bounds of work-stealing schedulers
  * Cache miss bounds of work-stealing schedulers


