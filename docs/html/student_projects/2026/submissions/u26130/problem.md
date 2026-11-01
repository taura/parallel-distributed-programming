# Simulating error correction for a quantum surface code

# Problem

A quantum computer must detect and correct errors without directly reading the
information it is trying to protect. A *surface code* does this by arranging
checks on a grid. The pattern of checks that signal a possible error is called
a *syndrome*. Errors in the stored data, or in the checks themselves, can
change this pattern. Given the syndrome over several rounds of checks, a
*decoder* guesses which errors happened and how to correct them.

This program asks: **How often does the decoder's correction leave a logical
error?** It estimates that rate by simulating many independent trials (called
*shots*). It is a simulation of the error-correction algorithm, not a program
that runs on a quantum computer.

# How the algorithm/baseline code works

The code first builds a grid of possible errors and checks. It includes errors
in the data and in measurements made during successive rounds of checks. In each shot:

1. It simulates errors in both the stored data and the check measurements.
   Each possible error occurs independently with probability `p`; a measurement
   error flips a check result. The code records the resulting syndrome and
   whether the simulated errors change the protected logical information.
2. The decoder sees only the syndrome. It grows clusters from the checks that
   signaled an error by extending along grid edges. When growing edges connect
   clusters, it joins them and traces paths to produce a correction.
3. The code compares the logical effect of that correction with the logical
   effect of the original errors. A mismatch counts as a logical error.

The reported `logical_error_rate` is the number of mismatches divided by the
number of shots. The program also reports time spent sampling errors and
decoding. `--d` sets the grid's code distance, `--rounds` the number of check
rounds, `--p` the error probability, `--shots` the number of trials, and
`--seed` the random seed. For example, `./qec_baseline --d 5 --shots 100000`.
To check the result, `./qec_baseline --selftest` tries individual errors and
reports how many are decoded incorrectly. With `--p 0`, there should be no
logical errors.

# Where is parallelism

- **Across shots:** Each trial has its own random-number stream and does not
  depend on other trials. CPU cores or GPU workers could run different shots
  simultaneously, then add up their error counts. They would need separate
  decoder state for each shot. Some shots have more errors and take longer to
  decode, so workers that finish early may have to wait for slower ones.
- **Across edges during growth:** Within one shot, the decoder checks many grid
  edges to decide whether their regions should grow or meet. In a growth step,
  these checks read the current state independently; the new edge states are
  applied afterward. CPU threads or GPU workers could check different edges
  in parallel. Each step would still have to wait before applying the new
  states, and joining regions can depend on earlier joins. Those dependencies
  and synchronization can limit speedup.
