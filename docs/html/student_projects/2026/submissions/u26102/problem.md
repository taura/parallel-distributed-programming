
* _this is an example file you should finish_
* write problem description, how the algorithm/baseline code works, and where you think you can apply parallel computing
* if you are looking at this file from Jupyter, you can see the preview by right-clicking on the cell and "Show Markdown Preview"

# Making XXXXXX from scratch

# Problem

サイン波を小規模なMLPで回帰する．

# How the algorithm/baseline code works

```
// ミニバッチ SGD
for(size_t pos=0;pos<idx.size();pos+=batch){
    std::vector<double> dw1,db1,dw2,db2; dw1.assign(w1.size(),0);
    // 先頭バッチ分だけ初期化
    // 勾配集計
    //   ŷ = forward(...);
    //   error = ŷ - y;
    //   dL/dw2 = error * h;
    //   ... (全勾配計算)
    //   w2 -= lr * dw2; b2 -= lr * db2;
    //   w1 += lr * dw1; b1 += lr * db1;
}
```

# Where is parallelism

- ミニバッチごとの勾配集計を `#pragma omp prallel for`でデータ並列化
- ベクトル積を`#pragma omp simd`で
