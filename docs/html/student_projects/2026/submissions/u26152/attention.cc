
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <algorithm>
using namespace std;

int main(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 1024;
    int D = argc > 2 ? atoi(argv[2]) : 64;
    if (N <= 0 || D <= 0) return 1;

    vector<float> Q(size_t(N)*D), K(size_t(N)*D), V(size_t(N)*D);
    vector<float> S(size_t(N)*N), O(size_t(N)*D, 0);

    for (size_t i = 0; i < Q.size(); i++) {
        Q[i] = sin(i * 0.01);
        K[i] = cos(i * 0.01);
        V[i] = sin(i * 0.02);
    }

    using Clock = chrono::steady_clock;
    auto t0 = Clock::now();

    // Step 1: QK^T / sqrt(D)
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < D; k++)
                sum += Q[i*D+k] * K[j*D+k];
            S[i*N+j] = sum / sqrt(float(D));
        }

    auto t1 = Clock::now();

    // Step 2: Softmax
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < N; i++) {
        float mx = S[i*N];
        for (int j = 0; j < N; j++)
            mx = max(mx, S[i*N+j]);

        float sum = 0;
        for (int j = 0; j < N; j++) {
            S[i*N+j] = exp(S[i*N+j] - mx);
            sum += S[i*N+j];
        }
        for (int j = 0; j < N; j++)
            S[i*N+j] /= sum;
    }

    auto t2 = Clock::now();

    // Step 3: Multiply by V
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < N; i++)
        for (int k = 0; k < D; k++) {
            float sum = 0;
            for (int j = 0; j < N; j++)
                sum += S[i*N+j] * V[j*D+k];
            O[i*D+k] = sum;
        }

    auto t3 = Clock::now();

    auto seconds = [](auto a, auto b) {
        return chrono::duration<double>(b-a).count();
    };

    double max_error = 0;
    for (int i = 0; i < N; i++) {
        double row_sum = 0;
        for (int j = 0; j < N; j++) row_sum += S[i*N+j];
        max_error = max(max_error, abs(row_sum - 1.0));
    }

    double checksum = 0;
    for (float x : O) checksum += x;

    cout << "N=" << N << " D=" << D << "\n";
    cout << "Scores:  " << seconds(t0,t1) << " s\n";
    cout << "Softmax: " << seconds(t1,t2) << " s\n";
    cout << "Output:  " << seconds(t2,t3) << " s\n";
    cout << "Total:   " << seconds(t0,t3) << " s\n";
    cout << "Checksum: " << checksum << "\n";
    cout << "Softmax max error: " << max_error << "\n";

    // Generate attention heatmap (PPM)
    string name = argc > 3 ? argv[3] : "attention.ppm";
    ofstream img(name);
    img << "P3\n" << N << " " << N << "\n255\n";

    float max_p = *max_element(S.begin(), S.end());
    for (float p : S) {
        int c = int(255 * p / max_p);
        img << c << " " << c << " " << c << "\n";
    }

    return 0;
}
