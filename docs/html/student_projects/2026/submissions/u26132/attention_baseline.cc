// compilation example: clang++ -O3 -std=c++17 scaled_dot_product.cc -o scaled_dot_product
// execution example: ./scaled_dot_product 128 256

#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv){
    int N = argc>1 ? atoi(argv[1]) : 64;         // sequence length
    int d = argc>2 ? atoi(argv[2]) : 128;        // model dimension
    mt19937 rng(42);
    uniform_real_distribution<float> dist(-1.f,1.f);
    vector<vector<float>> Q(N, vector<float>(d)),
                           K(N, vector<float>(d)),
                           V(N, vector<float>(d));
    for (auto &row : Q) for (float &x : row) x = dist(rng);
    for (auto &row : K) for (float &x : row) x = dist(rng);
    for (auto &row : V) for (float &x : row) x = dist(rng);

    auto t0 = chrono::high_resolution_clock::now();

    // Q * K^T / sqrt(d)
    vector<vector<float>> scores(N, vector<float>(N,0));
    float inv_sqrt_d = 1.f / sqrt(d);
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++){
            float sum=0;
            for (int k=0;k<d;k++) sum += Q[i][k]*K[j][k];
            scores[i][j] = sum * inv_sqrt_d;
        }

    // row-wise softmax
    vector<vector<float>> attn(N, vector<float>(N));
    for (int i=0;i<N;i++){
        float mx = *max_element(scores[i].begin(), scores[i].end());
        vector<float> exps(N);
        for (int j=0;j<N;j++) exps[j] = exp(scores[i][j] - mx);
        float sum = 0;
        for (int j=0;j<N;j++) sum += exps[j];
        for (int j=0;j<N;j++) attn[i][j] = exps[j] / sum;
    }

    // output = attn · V
    vector<vector<float>> out(N, vector<float>(d));
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            for (int k=0;k<d;k++) out[i][k] += attn[i][j] * V[j][k];

    auto t1 = chrono::high_resolution_clock::now();
    cerr << "Time (ms): "
         << chrono::duration<double, milli>(t1-t0).count() << '\n';
    // simple verification: Frobenius norm of output
    double norm = 0;
    for (auto &row : out)
        for (float x : row) norm += x * x;
    cout << fixed << setprecision(6) << norm << '\n';
    return 0;
}