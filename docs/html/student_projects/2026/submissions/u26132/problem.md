
* _this is an example file you should finish_
* write problem description, how the algorithm/baseline code works, and where you think you can apply parallel computing
* if you are looking at this file from Jupyter, you can see the preview by right-clicking on the cell and "Show Markdown Preview"

# Making Scaled Dot-Product Attention from scratch

# Problem

Calculate Scaled Dot-Product Attention
$$ A = \mathrm{softmax}\Big(\frac{Q\cdot K^T}{\sqrt{d_k}}\Big)V $$
* Queries `Q` (shape `N × D`)  
* Keys  `K` (shape `N × D`)  
* Values `V` (shape `N × D`)

# How the algorithm/baseline code works

```
// 1) Q * K / sqrt(d_k)
    vector<vector<float>> scores(N, vector<float>(N, 0));
    const float inv_sqrt_d = 1.f / sqrt(D);
    for (long long i = 0; i < N; ++i)
        for (long long j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < D; ++k) sum += Q[i][k] * K[j][k];
            scores[i][j] = sum * inv_sqrt_d;
        }

    // 2) Row‑wise softmax
    vector<vector<float>> attn(N, vector<float>(N, 0));
    for (long long i = 0; i < N; ++i) {
        float maxv = *max_element(scores[i].begin(), scores[i].end());
        vector<float> expv(N);
        for (long long j = 0; j < N; ++j)
            expv[j] = exp(scores[i][j] - maxv);
        float sum = 0;
        for (long long j = 0; j < N; ++j) sum += expv[j];
        for (long long j = 0; j < N; ++j) attn[i][j] = expv[j] / sum;
    }

    // 3) attn · V
    vector<vector<float>> out(N, vector<float>(D, 0));
    for (long long i = 0; i < N; ++i)
        for (long long j = 0; j < N; ++j)
            for (int k = 0; k < D; ++k)
                out[i][k] += attn[i][j] * V[j][k];
```

# Where is parallelism

* Score matrix (Q * K^T), : matrix-matrix multiplication (GEMM operation)
    * Each row can be calculated in parallel (OpenMP, SIMD, GPU kernel)
* Row-wise softmax
    * Each row can be processed, and reduction tree or OpenMP `reduction(+:sum)` are available
* Some famous methodologies such as FlashAttention might be available


I don't have enough time, so this problem is straightforward. I want to try entire LLM inference.
