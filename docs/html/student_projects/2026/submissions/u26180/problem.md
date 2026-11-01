
* _this is an example file you should finish_
* write problem description, how the algorithm/baseline code works, and where you think you can apply parallel computing
* if you are looking at this file from Jupyter, you can see the preview by right-clicking on the cell and "Show Markdown Preview"

# Making XXXXXX from scratch

# Problem

[remove this] 

$$ s = \sum_{n=0}^{\infty} \frac{1}{n^2} $$

# How the algorithm/baseline code works

```
double s = 0.0;
for (n = 1000000; n >= 1; n--) {
  s += 1 / ((double)n * (double)n);
}
```

# Where is parallelism

- all $1/n^2$ tems can be done in parallel
