# European Call Option Pricing by Monte Carlo Simulation

## Problem and model

A **European call option** gives its owner the right to buy a stock on a fixed future date for a fixed **strike price** $K$. If the stock price on that date is $S_T$, the option pays $\max(S_T-K,0)$. We want its value today under a simple mathematical model, not a prediction of the actual stock price.

We assume no dividends, a constant interest rate $r$, and constant **volatility** $\sigma$ (how much the stock price can vary). Starting from today's price $S_0$, we draw a standard normal random number $Z$ (from a bell-shaped distribution) and generate only the price at expiration, $T$ years from now:

$$S_T=S_0\exp\left((r-\sigma^2/2)T+\sigma\sqrt{T}\,Z\right).$$

This is a **risk-neutral** model: it uses the interest rate for valuation rather than a forecast of the stock's growth. Each trial gives a *discounted payoff* $Y=e^{-rT}\max(S_T-K,0)$; discounting converts a future payment into its value today. The option price is the expected (average) value of $Y$ in this model. Intermediate stock prices are unnecessary.

## Serial algorithm

`main.cpp` makes $N$ independent **Monte Carlo trials** (random samples of possible outcomes). It generates $Z$, calculates $S_T$ and $Y$, and updates the mean and sample variance. **Welford's method** performs these updates without storing every trial. The standard-library `std::mt19937_64` and `std::normal_distribution` generate the random values from a chosen seed. Results can differ between C++ standard-library implementations.

The mean estimates the option price. The **standard error** estimates the uncertainty of this mean from a finite number of trials. The program reports an approximate 95% confidence interval, estimate $\pm 1.96$ standard errors; a single interval need not contain the true model price. It also computes the **Black–Scholes price**, an analytic value for the same model, using standard C++ math functions. At $T=0$ or $\sigma=0$, the payoff is known without random sampling and is computed directly.

## Build and run on Miyabi

Arguments are `N S0 K T r sigma seed`, in that order; omit trailing arguments to use defaults `1000000 100 100 1 0.05 0.2 42`. The program requires $N\ge2$, $S_0,K>0$, and $T,\sigma\ge0$; $r$ may be negative. Non-finite values and calculations that overflow are rejected.

In the assignment notebook's Miyabi G kernel, build and check the program with:

```bash
%%bash
cd code
module purge && module load nvidia/25.9 cuda/12.9 && PATH=/work/gt81/share/env/llvm/bin:$PATH
clang++ -O3 -Wall -std=c++17 main.cpp -o app
./app --self-test
```

Run `%load_ext miyabi` in a separate cell first. Then use a compute-node job for the main benchmark:

```bash
%%sub
cd code
./app 500000000 100 100 1 0.05 0.2 42
```

The program prints a short result and timing summary. It appends a full row to `code/results.csv` when run from `code`, including inputs, seed, estimate, standard error, interval, analytic price, loop time, throughput, and a UTC timestamp. Use `--log FILE` before the arguments to choose another CSV file. The measured time covers the Monte Carlo loop, not startup, writing the log, or job waiting time. Run jobs that share the log sequentially; record compiler and node settings separately.

## Checks and measured results

The built-in self-test checks simple payoffs, the $T=0$ and $\sigma=0$ cases, a known mean and sample variance, and a Black–Scholes value of about **10.4506**. The notebook records `self-test: PASS` on Miyabi.

The following Miyabi measurements use $S_0=K=100$, $T=1$, $r=0.05$, $\sigma=0.2$, and seed 42. The analytic price is **10.45058357**.

| Trials $N$ | Estimated price | Standard error | Computation time |
| ---: | ---: | ---: | ---: |
| 10,000,000 | 10.45010109 | 0.00465545 | 2.384 s and 2.373 s |
| 100,000,000 | 10.44941074 | 0.00147180 | 23.742 s |
| 500,000,000 | 10.45008717 | 0.00065823 | 117.670 s |

Five times as many trials (100 million to 500 million) took about **4.96 times** as long; the standard error fell by about $\sqrt5$, as expected. Throughput was around **4.2 million trials/s**. This makes 500 million trials a useful, roughly two-minute serial baseline and 100 million a shorter comparison case. All three intervals contained the analytic price, but that alone does not prove correctness. Runs with the same seed share initial random samples, so their estimates are not independent; the absolute error need not decrease for every larger $N$.

## Future parallel work

Trials can be divided among CPU cores or GPU workers. Each worker could collect its own statistics and combine them at the end, provided the random-number streams do not repeat the same samples. Compare serial and parallel times using **the same inputs and trial count**, and repeat runs to assess timing variation. Later, independent seeds can test statistical variation; different strikes such as $K=80$ and $K=120$ can test other payoff patterns. Those cases have **not** been measured here. Neither parallel execution nor variance-reduction methods are implemented in this baseline.
