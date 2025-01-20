<link rel="stylesheet" href="../scripts/style.css">

# How to Get Credit {.unnumbered}

1. participate in in-class exercises (evaluated based on submissions to reflections)
1. submit programming exercise assignments (`pd01_omp` and `pd30_mlp`)
1. write and submit a final report (term paper).
    * Abstract deadline: <font color="red">Jan 18th (Sat) 23:59</font>
    * Final report deadline: <font color="red">February 4th (Tue) 23:59</font>

# Spec

* Define a problem you want to get high performance for and apply what you have learned in the class to achieve high performance. The problem may be one that arises in your research or one that interests you. It must be one for which you have a good prospect for applying parallelization or vectorization. Think of good parallel algorithms to get good performance. Apply multicore parallelization, GPU parallelization and/or vectorization, understand what is the maximum performance achievable, and investigate how close your implementation is.

# How to Choose the Problem

* You are encouraged to seek a problem that is relevant to your research or that you are highly interested in
* That said, if you have no idea what problem you want to work on, here are some suggestions
* Simulation
  * Solve a Poisson equation with Finite Diffence Method
  * Solve a Poisson equation with Finite Element Method
  * Solve a Schrödinger equation with Density Functional Theory
* Big Data Processing
  * Build Suffix Array for large text or Gnome sequence
* Machine Learning
  * Train MLP
  * Train a Convolutional Neural Network
  * Train a Transformer
* Either way, you are encouraged to use this opportunity to _understand_ the algorithms/computational methods you have been treating as blackbox

# Rules about the Use of GenAI (e.g., ChatGPT)

* You can ask AI to get suggestions 
  * e.g., "can you suggest me a computational method for solving XYZ?"
* You can also ask AI to generate a baseline code for the problem you chose
  * e.g., "can you show me a sequential Python (C++) code for constructing Suffix Array?"
  * e.g., "can you show me a sequential Python (C++) code for GPT"
* <font color="blue">_As a matter of fact, it is instructive to use AI to obtain the simple baseline code you can easily understand and get started with_</font>
* Clearly indicate in your report which part is helped by AI
  
# Rules about the Use of (High-Level) Libraries and Languages

* General rule is that you are not allowed to use libraries that solve a significant part of the problem on your behalf
* For example, you are not allowed to use `torch` to train MLP, as it does most important components (e.g., matrix multiplication, backpropagation, and parameter updates) on your behalf
* Along the same line, you cannot use high performance library for matrix-matrix multipliocation (e.g., BLAS) or a linear equation solver (PETSc)
* Comparing your implementation with one of those libraries is encouraged
* Using languages other than C/C++ is discouraged but allowed, as long as you can analyze the performance in the machine code level, which is difficut in scripting languages such as Python
* You are generally advised to work on a simple baseline code written in low-level languages, for which performance analysis at machine code level is relatively simple

# What You Have to Do Yourself

* You have to apply parallelization, vectorization, and other code-level optimizations by yourself
* You have to find the most time-consuming part(s) of the code through actual measurements by yourself (you can ask ChatGPT which part is likely to be most time-consuming, but you must witness it by yourself, through real experiments)
* You have to analyze the machine code of the time-consuming part(s) by actually generating assembly code and seeing them by yourself
* You have to do other necessary analysis to reason about the performance of the code by yourself (ChatGPT won't be able to do it anyways, but just in case you might consider asking it something like "Please reason about the performance of this code", it would respond to you with full of hullcinations)

# What Must Be in the Abstract

* A brief description of the problem you are going to work on
* A prospect of performance optimizations (vectorization, parallelizations, ILP, etc.) you are going to apply

# How to Submit the Abstract

* Just send the brief descriptions through UTOL assignment (<font color="blue">"Final Report Abstract"</font>)

* Thank you for sending your abstracts!
  * Broyden–Fletcher–Goldfarb–Shanno (BFGS) Quasi-Newton method
  * Convolutional Neural Network
  * Convolutional Neural Network
  * Convolutional Neural Network
  * Convolutional Neural Network
  * Convolutional Neural Network
  * Convolution Neural Network
  * Convolution Neural Network
  * Fast Fourier Transform
  * Fast Fourier Transform
  * Fast Fourier Transform
  * Floyd Warshall algorithm
  * Fluid dynamics using Position-Based Fluids (PBF)
  * Inverse Kinematics Problems
  * jbonsai (speech synthesis)
  * lattice-based cryptography
  * Multilayer Perceptron
  * Multilayer Perceptron
  * Multilayer Perceptron
  * Multilayer Perceptron
  * Neural network to classify handwritten digits from the MNIST dataset
  * Optimizing a neural network for the MNIST dataset
  * Optimizing Training of YOLO-Nano Object Detector
  * Parallel Procedural Terrain Generation
  * Path tracing
  * Poisson equation with Finite Difference Method
  * Ray tracing
  * Ray tracing
  * Ray tracing
  * Schrödinger equation with Density Functional Theory
  * Simulating air traffic control
  * Solid mechanics problem using the finite element method
  * Solve a Poisson equation with Finite Element Method
  * Stable Fluids (fluid simulation)
  * Stroke optimization algorithm ?
  * Suffix Array
  * Suffix array construction
  * Suffix array construction
  * Transformer
  * UNet
  * Vision Transformer
  * Vision Transformer


# What Must Be in the Report

* Description of the problem
* Description of the basic algorithm (why it solves the problem it is supposed to solve)
* The code that solves the problem
* Reproducing instructions (instructions for compiling and running your code so that I can reproduce it)
* Description of performance optimization you applied (parallelization, vectorization, etc.)
* Experimental results that show your implementation is correct
* Experimental results that show the performance of your code
* Analysis of performance (in the manner you did it for `pd30_mlp` exercise)
* A brief description about the part(s) of your work significantly helped by generative AIs including the name of the service (e.g., ChatGPT 4o, Github Copilot, etc.) and model

# How to Submit the Report

* As in other assignments, the body of the work should be sent through Jupyter and a note should be dropped in UTOL

Details:

* I provided you with an almost empty Jupyter notebook, `pd31_final`, just as a uniform means to submit your work
* You can write your report in the Jupyter notebook just as you have done for `pd30_mlp`, or put a separate PDF file under the `pd31_final` directory; you can mix the two
* You should put all your code under the same `pd31_final` directory
* It is recommended to write the reproducing instructions directly in the Jupyter notebook as a bash code cell, so that I can easily compile and run your code, but if you have an issue with composing Jupyter notebook, you can write it in the PDF file
* To submit your work, press the `submit` button in the Jupyter, which sends your report (notebook or PDF) and code (everything under `pd3_final`, for that matter)
  * the directory should not contain a big file, which may cause a submission error
  * if the big file is an essential part of your work you cannot get rid of, put it elsewhere (e.g., Google drive or OneDrive) and make a link to it
* After submitting the work in Jupyter, drop a note for the UTOL assignment (<font color="blue">"Final Report"</font>) saying you have submitted your work (as you have done for earlier assignments)


