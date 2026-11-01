
* _this is an example file you should finish_
* write problem description, how the algorithm/baseline code works, and where you think you can apply parallel computing
* if you are looking at this file from Jupyter, you can see the preview by right-clicking on the cell and "Show Markdown Preview"

# Making a Historical Portfolio Search Tool from scratch

# Problem

How can we compare different ways of distributing a fixed investment budget among stocks? This program searches many candidate portfolios and reports several configurations that perform well on a historical training period. A portfolio lists stock names and the fraction of the budget assigned to each stock.

The application is an educational tool for studying the tradeoff between historical return and variability, and whether performance carries over to a later period. It does not predict prices or guarantee future returns. The search is approximate: the reported portfolios are the best distinct candidates encountered, not proven global optima.

All weights are nonnegative, sum to 100%, and have a resolution of 0.1%. The model assumes fractional shares and daily rebalancing to fixed target weights, with no transaction costs, taxes, borrowing, or short selling. Real input should contain consistently adjusted closing prices on common trading dates. Historical stock selection can introduce survivorship bias, which this program does not correct.

# How the algorithm/baseline code works

`portfolio.cc` is a complete serial C++17 implementation using only standard library headers. It has no external dependencies and includes a synthetic dataset so it can run immediately without downloading market data. Synthetic results are explicitly labeled and are not evidence of investment performance.

1. Read a price table, or generate 757 synthetic price observations for six assets using a fixed data-generation seed. Synthetic assets share a random market component and have individual random components.
2. Compute simple daily returns, `r[t][i] = price[t+1][i] / price[t][i] - 1`. Split these returns chronologically into training and test periods (70% / 30% by default). The first test return starts at the last training price; no daily return appears in both periods.
3. For each candidate, randomly choose the number of holdings, uniformly between one and the configured maximum, then shuffle the asset indices to choose a subset. Randomly choose distinct cuts among integers 1 through 999. The gaps between sorted cuts and endpoints 0 and 1000 give positive integer allocations totaling 1000. Thus weights have exactly 0.1% resolution. Conditional on the chosen number of holdings and subset, positive integer allocations are sampled uniformly; portfolios across all possible holding counts are not equally likely.
4. For each training day, compute the daily-rebalanced portfolio return `R[t] = sum_i weight[i] * r[t][i]`. Compute the daily sample mean and sample variance using an online update. Annualized arithmetic mean return is `252 * mean`; annualized volatility is `sqrt(252 * sample_variance)`. The ranking score is `(annualized_mean - annual_risk_free_rate) / annualized_volatility`, a conventional annualized Sharpe estimate assuming 252 trading days. Effectively zero-volatility candidates are assigned a lowest-priority internal score and displayed as `NA`.
5. Keep up to K distinct candidates ordered by training score. Ties use the candidate generation index. Evaluate those same weights on the untouched test returns after search completes; test results never influence ranking. Report compounded total return as `product_t(1 + R[t]) - 1`, along with the other metrics. The CSV has separate training and test rows and decimal weights; console weights are percentages.

The baseline evaluates returns directly rather than precomputing a covariance matrix. This keeps the daily-rebalancing interpretation explicit and provides substantial, scalable numerical work. For M candidates, T training days, and at most H holdings among N assets, the main cost is approximately `O(M * (N + H log H + T*H + 1000))`, plus maintenance of the small top-K list. Price/return storage is `O(TotalDays*N)` and candidate storage is `O(K*N)`.

## Build and run

From the directory containing the source:

```bash
c++ -O3 -std=c++17 -Wall -Wextra -pedantic portfolio.cc -o portfolio
./portfolio --demo --samples 20000 --top 10 --output portfolios.csv
./portfolio --input prices.csv --samples 100000 --max-holdings 20 --seed 12345 --output portfolios.csv
./portfolio --help
```

Use the course's compiler/module setup on Miyabi, then run the same build and execution commands in the required notebook cells. Compilation and execution on Miyabi must be confirmed separately.

The default run uses synthetic data. Options are `--input`, `--demo`, `--samples`, `--top`, `--max-holdings`, `--seed`, `--train-fraction`, `--risk-free`, and `--output`. The risk-free rate is an annual decimal (for example, `0.02`). Increase `--samples` to increase runtime; more assets, more training days, and more holdings also increase the work. Measure runtime on the target machine instead of assuming a particular duration. The printed search time excludes input preparation, final test evaluation, and output.

Input CSV format:

```csv
Date,STOCK_A,STOCK_B
2023-01-03,100.0,80.0
2023-01-04,101.0,79.5
```

This short snippet only illustrates the format; a run requires at least 21 price rows, and at least two returns per split. Supply one row per common trading day, with strictly increasing ISO `YYYY-MM-DD` dates, unique stock names, and positive finite prices. Missing values and quoted CSV input fields are unsupported and must be cleaned before use. Stock names must not contain commas. The program validates column counts, date ordering, numeric values, and split sizes.

## Checking the results

- Every reported allocation must be nonnegative, use integer multiples of 0.001, and sum to one.
- With the same executable, input, options, and seed, the result CSV is reproducible. Standard library random distributions can differ across compiler/library implementations, so cross-platform bitwise identity is not promised.
- A one-stock input must produce a 100% allocation to that stock, with daily returns equal to the stock's returns. Fewer than K results are valid if fewer distinct portfolios were encountered.
- Identical asset price series must produce identical portfolio performance regardless of allocation, apart from floating-point roundoff.
- Constant prices must produce zero total return and volatility, with Sharpe displayed as `NA`.
- Training scores in the output must be nonincreasing. Test results can be worse than training results; this is an informative outcome rather than a program error.

# Where is parallelism

Candidate evaluations are independent once the shared return table is prepared. A future multicore implementation can distribute candidate IDs among workers. Each worker can generate and evaluate candidates, maintain a local top-K list, and then merge those lists with duplicate removal. Random-number generation must be private to workers; seeding by candidate ID is one way to make the candidate set independent of scheduling. The current serial version uses one random-number stream, so adopting such a scheme would also require updating the serial reference for a fair comparison.

Daily portfolio dot products offer SIMD opportunities. A GPU implementation can evaluate many candidates or candidate/day pairs concurrently, followed by reductions for statistics and selection. The final test evaluations can also run independently. Random generation, reductions, memory layout, and top-K merging require explicit work; adding a pragma alone is insufficient.

Useful performance experiments include varying candidate count, asset count, holding count, and history length; measuring time spent in generation versus evaluation; and studying return-table layout and cache reuse. Any optimized version should preserve the portfolio constraints and compare numerical results against the serial reference with a floating-point tolerance.
