// mppi.cpp  --  usage: ./mppi [K] [T] [steps] [lambda] [sigma]
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

static const double PI = 3.14159265358979323846; 
static const double G = 9.81,  DAMP = 0.1, DT = 0.02, UMAX = 2.0; // gravity, friction coefficient, simulation timestep, motor torque limit

struct State { double th, thd; };   // th = angle, thd = angular velocity, th = 0 when upright, and pi when hanging down
// We are assuming mass = length = 1

static inline State step(State s, double u) { // takes current state and torque as params
    s.thd += (G * std::sin(s.th) - DAMP * s.thd + u) * DT;  // m = l = 1, acceleration = g*sin(theta), minus friction, plus motor input
    s.th  += s.thd * DT; 
    return s;
}

// stage cost is the cost per step. 1 - cos(th) penalizes being away from upright, thd^2 penalizes spinning, and u^2 penalizes control effort.
static inline double stage_cost(const State& s, double u) {
    return (1.0 - std::cos(s.th)) + 0.01 * s.thd * s.thd + 0.001 * u * u;
} 

// high terminal cost so that planner cares most about where each trajectory ends up
static inline double terminal_cost(const State& s) {
    return 10.0 * (1.0 - std::cos(s.th)) + 0.1 * s.thd * s.thd;
}

// counter-based RNG (splitmix64): same result regardless of thread order
struct Rng {
    uint64_t s;
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    double uniform() { return ((next() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
    double normal() {
        double u1 = uniform();
        double u2 = uniform();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * PI * u2);
    }
};

int main(int argc, char** argv) {
    const int K     = argc > 1 ? std::atoi(argv[1]) : 4096;  // number of sampled trajectories per control step
    const int T     = argc > 2 ? std::atoi(argv[2]) : 100;   // horizon length in steps (100 × 0.02 s (DT) = 2 s lookahead)
    const int steps = argc > 3 ? std::atoi(argv[3]) : 600;   // how many times the controller runs (600 × 0.02 s (DT) = 12 s of simulated time).
    const double lambda = argc > 4 ? std::atof(argv[4]) : 1.0; // 'temperature' for the weighting
    const double sigma  = argc > 5 ? std::atof(argv[5]) : 1.0; // standard deviation of the noise added to the torque
    const uint64_t seed = 42; // random seed

    State x{PI, 0.0};                       // start hanging down, initial state
    std::vector<double> U(T, 0.0);          // nominal control sequence
    std::vector<double> eps((size_t)K * T); // Storage for every sample’s noise. Sample k, time t lives at index k*T + t.
    std::vector<double> cost(K), w(K); // One total cost and one weight per sample.
    auto start = std::chrono::steady_clock::now();
    for (int n = 0; n < steps; ++n) { // for every simulation step
        // 1. rollouts (independent over k -> parallel loop later)
        for (int k = 0; k < K; ++k) { // simulate the k-th model out of K models.
            Rng r{seed ^ ((uint64_t)n << 32) ^ (uint64_t)k};
            State s = x; // initial state
            double c = 0.0; // initial cost
            for (int t = 0; t < T; ++t) { // loop over T horizons
                double e = sigma * r.normal(); // Draw Gaussian noise
                eps[(size_t)k * T + t] = e; // save the noise
                double u = std::clamp(U[t] + e, -UMAX, UMAX); // Perturbed torque = current torque + noise
                s = step(s, u); // next step using new torque
                c += stage_cost(s, u); // add the stage cost measured on the new state
            }
            cost[k] = c + terminal_cost(s); // Add the terminal cost and store the sample’s total.
        }
        // 2. weights (subtract min for numerical stability)
        double cmin = *std::min_element(cost.begin(), cost.end()); // smallest total cost across all samples/models
        double wsum = 0.0, w2sum = 0.0;
        for (int k = 0; k < K; ++k) {
            w[k] = std::exp(-(cost[k] - cmin) / lambda); // weight for each sample
            wsum += w[k];  // total weight across all samples
        }
        for (int k = 0; k < K; ++k) {
            w[k] = w[k] / wsum; // Normalize weights over the total weight, sum = 1
        }
        // 3. compute the weighted average of all samples’ noise at that time, and add it to the plan
        for (int t = 0; t < T; ++t) { // for every horizon step
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                acc += w[k] * eps[(size_t)k * T + t];
            }
            U[t] = std::clamp(U[t] + acc, -UMAX, UMAX); // Motor inputs decided
        }
        for (int k = 0; k < K; ++k) w2sum += w[k] * w[k]; // Compute the sum of squared normalized weights

        // 4. apply first control to the real system, then shift the plan
        x = step(x, std::clamp(U[0], -UMAX, UMAX)); // Apply only the first torque of the updated plan to the real system.
        for (int t = 0; t + 1 < T; ++t) U[t] = U[t + 1]; // Shift the plan one slot left
        U[T - 1] = 0.0; // last value is arbitrary after shift, set to 0

        if (n % 20 == 0 || n == steps - 1)
            std::printf("step %3d  theta=%7.3f  omega=%7.3f  min_cost=%8.2f  ESS=%7.1f\n",
                        n, std::remainder(x.th, 2 * PI), x.thd, cmin, 1.0 / w2sum);
    }
    auto end = std::chrono::steady_clock::now();
    double elapsed_secs = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    double err = std::fabs(std::remainder(x.th, 2 * PI));
    std::printf("final |theta|=%.4f rad, |omega|=%.4f  -> %s\n", err, std::fabs(x.thd),
                (err < 0.1 && std::fabs(x.thd) < 0.5) ? "SWING-UP OK" : "FAILED");
    std::printf("Elapsed computation time: %f seconds\n", elapsed_secs);
    return 0;
}