
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <algorithm>
using namespace std;

int main(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 1024;
    int D = argc > 2 ? atoi(argv[2]) : 64;

    if (N <= 0 || D <= 0) return 1;

    vector<float> Q(size_t(N)*D), K(size_t(N)*D), V(size_t(N)*D);
    vector<float> S(size_t(N)*N), O(size_t(N)*D, 0);

    // Initialize Q, K, V
    for (size_t i = 0; i < Q.size(); i++) {
        Q[i] = sin(i * 0.01);
        K[i] = cos(i * 0.01);
        V[i] = sin(i * 0.02);
    }

    auto start = chrono::steady_clock::now();

    // Step 1: Attention scores QK^T / sqrt(D)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < D; k++)
                sum += Q[i*D+k] * K[j*D+k];

            S[i*N+j] = sum / sqrt(float(D));
        }
    }

    // Step 2: Softmax
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

    // Step 3: Multiply attention weights by V
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < D; k++) {
            float sum = 0;
            for (int j = 0; j < N; j++)
                sum += S[i*N+j] * V[j*D+k];

            O[i*D+k] = sum;
        }
    }

    auto end = chrono::steady_clock::now();

    // Verify softmax rows sum to 1
    double max_error = 0;
    for (int i = 0; i < N; i++) {
        double sum = 0;
        for (int j = 0; j < N; j++)
            sum += S[i*N+j];

        max_error = max(max_error, abs(sum - 1.0));
    }

    // Calculate output checksum
    double checksum = 0;
    for (float x : O)
        checksum += x;

    // Generate grayscale attention heatmap
    ofstream img("attention.ppm");
    img << "P3\n" << N << " " << N << "\n255\n";

    float max_p = *max_element(S.begin(), S.end());

    for (float p : S) {
        int c = int(255 * p / max_p);
        img << c << " " << c << " " << c << "\n";
    }

    // Print results
    cout << "N = " << N << ", D = " << D << "\n";
    cout << "Checksum: " << checksum << "\n";
    cout << "Softmax max error: " << max_error << "\n";
    cout << "Time: "
         << chrono::duration<double>(end-start).count()
         << " seconds\n";

    return 0;
}
