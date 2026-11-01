# Sample input data

`prices_sample.csv` is a ready-to-run historical sample containing 504 price observations from 2007-01-03 through 2008-12-31 for MSFT (Microsoft), IBM, SBUX (Starbucks), and AAPL (Apple).

Source: [Plotly stockdata.csv](https://github.com/plotly/datasets/blob/master/stockdata.csv).

Raw source: https://raw.githubusercontent.com/plotly/datasets/master/stockdata.csv

Retrieved on 2026-10-31. The sample preserves the source's numeric values, selects the 2007–2008 rows, moves Date to the first column, removes CSV quoting, and omits GSPC because it is an index rather than an individual stock. No prices were simulated or missing values filled. The source file does not explicitly document its price adjustment methodology; treat this as an executable example, not a verified total-return dataset. This small stock selection is not the Nasdaq-100 universe.

Run from `code/` after compiling:

```bash
./portfolio --input prices_sample.csv --samples 20000 --top 10 --output portfolios.csv
```

The default chronological split gives 352 training daily returns and 151 test daily returns. The later period includes the 2008 financial crisis, so poor test performance is possible. Increase `--samples` to enlarge the computation for performance experiments.
