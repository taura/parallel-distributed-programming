#include <iostream>
#include <cstdlib>
#include <cmath>
#include <fstream>

using namespace std;

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        cerr << "Usage: ./wave2d N Nt" << endl;
        cerr << "Example: ./wave2d 256 300" << endl;
        return 1;
    }

    int N  = atoi(argv[1]);
    int Nt = atoi(argv[2]);

    const double dx = 1.0;
    const double dy = 1.0;
    const double c  = 200.0;
    const double dt = 0.002;

    double* u_prev = new double[N*N];
    double* u_curr = new double[N*N];
    double* u_next = new double[N*N];

    // initialize zero
    for (int j = 0; j < N; j++)
    {
        for (int i = 0; i < N; i++)
        {
            u_prev[j*N + i] = 0.0;
            u_curr[j*N + i] = 0.0;
            u_next[j*N + i] = 0.0;
        }
    }
    int cx = N / 2;
    int cy = N / 2;

    // Ricker wavelet parameters
    const double f0 = 8.0;
    const double t0 = 1.2 / f0;

    // time loop
    for (int t = 0; t < Nt; t++)
    {
        // Ricker source added at center each time step
        double t_src = t * dt;
        double tau = t_src - t0;
        double ricker = (1.0 - 2.0 * M_PI*M_PI * f0*f0 * tau*tau) * exp(-M_PI*M_PI * f0*f0 * tau*tau);
        u_curr[cy * N + cx] += ricker;

        for (int j = 1; j < N-1; j++)
        {
            for (int i = 1; i < N-1; i++)
            {
                double lap = u_curr[j*N + i + 1]
                           + u_curr[j*N + i - 1]
                           + u_curr[(j+1)*N + i]
                           + u_curr[(j-1)*N + i]
                           - 4.0 * u_curr[j*N + i];

                u_next[j*N + i] = 2.0 * u_curr[j*N + i]
                                - u_prev[j*N + i]
                                + c*c*dt*dt/(dx*dy) * lap;
            }
        }
        // swap three wavefield arrays
        double* tmp = u_prev;
        u_prev = u_curr;
        u_curr = u_next;
        u_next = tmp;
    }

    // PPM output: gray background(127,127,127), red positive, blue negative
    const double gain = 4.0;
    const double clip = 1.0;
    ofstream fout("wave.ppm");
    fout << "P3\n" << N << " " << N << "\n255\n";
    for (int j = 0; j < N; j++)
    {
        for (int i = 0; i < N; i++)
        {
            double val = u_curr[j*N + i] * gain;
            if(val > clip) val = clip;
            if(val < -clip) val = -clip;

            int r = 127;
            int g = 127;
            int b = 127;

            if(val > 0)
            {
                r = 127 + (int)(128 * val);
                g = 127 - (int)(127 * val);
                b = 127 - (int)(127 * val);
            }
            else
            {
                double v = -val;
                r = 127 - (int)(127 * v);
                g = 127 - (int)(127 * v);
                b = 127 + (int)(128 * v);
            }
            fout << r << " " << g << " " << b << "\n";
        }
    }
    fout.close();

    cout << "Done. N=" << N << ", Nt=" << Nt << ". Saved wave.ppm" << endl;

    delete[] u_prev;
    delete[] u_curr;
    delete[] u_next;
    return 0;
}
