<link rel="stylesheet" href="scripts/style.css">

# Parallel and Distributed Programming <br/> Kenjiro Taura {.unnumbered}

# What's new (in the newest-first order)

* <font color=blue>(Posted: Nov. 09, 2026)</font> Today's plan
  * [SIMD](slides/simd_arm.pdf)
  * Jupyter `pd07_simd` and `pd09_arm_asm`
  * Tentative grouping out (check 教材 / Course Materials section of [UTOL course page](https://utol.ecc.u-tokyo.ac.jp/lms/course?idnumber=2026_4884_4840-1004_01)); the goal is to make teams
	* Indicate where you are attending now (room or zoom) in the spreadsheet
	* Greet other members and briefly explain what your project is (2-3 min / member)
	* Exchange contact (email etc.) to discuss matters off class time
    * Each team should have 2 or 3 members (3 is preferred for redundancy) working on the same project
	* If your group has $>3$ participants today, you should split it into multiple teams each having 2 or 3 students (ignore those not present today)
	* Discuss how to form teams based on interests
	* If you make your team, write your team number (any unique ID) in the spreadsheet in the "team ID" column, so I can identify who are in the same team
	* Do it no later than this weekend (Sat. Nov 14), but try to do it today
	* If you make teams quickly (or your group is already the right size), start discussing the project 
  * Jupyter `pd08_ilp` ?
  
* <font color=blue>(Posted: Nov. 02, 2026)</font> Today's plan
  * [CUDA](slides/cuda.pdf)
  * Share [submissions to `pd02_app` (Making YOUR\_APP from scratch)](https://taura.github.io/parallel-distributed-programming/html/student_projects/2026/submissions/)
	* answer to the survey on this page by 2026/11/07 (Sat) -> Assignment 3
  * [OpenMP for GPU](slides/openmp_gpu.pdf)
  * Jupyter `pd05_omp_gpu`
  * Jupyter `pd06_visualize`

* <font color=blue>(Posted: Oct. 26, 2026)</font> Today's plan
  * [CUDA](slides/cuda.pdf)
  * Jupyter `pd04_cuda`

* <font color=blue>(Posted: Oct. 26, 2026)</font> 
  * Released "C++ primer" notebook `pd03_cpp` for those of you who are new to C++

* <font color=blue>(Posted: Oct. 19, 2026)</font> Today's plan
  * [OpenMP](slides/openmp.pdf)
  * Jupyter `pd01_omp`
  * Jupyter `pd02_app` (Making YOUR\_APP from scratch)
* <font color=blue>(Posted: Oct. 19, 2026)</font> Assignments
  * Assignment 1: `pd01_omp`, due Oct 31
  * Assignment 2: `pd02_app`, due Oct 31

* <font color=blue>(Posted: Oct. 05, 2026)</font> Today's plan
  * See the page [Getting Started with the Course Environment](html/get_started.html) and submit "Assignment 0"
  * [Introduction](slides/intro.pdf)
  * Receive your account during the class
  * Follow [Getting Started with the Course Environment](html/get_started.html), until you successfully fetch the first notebook (`pd00_intro`)!
  * Do not forget to submit a short reflection every week (posted on UTOL every week)
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


