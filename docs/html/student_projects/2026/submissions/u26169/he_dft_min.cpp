#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    const double pi = 3.14159265358979323846;
    const double hartree_to_ev = 27.211386245988;
    const double Z = 2.0;
    const int Ne = 2;

    int nr = 500;
    double rmax = 20.0;
    int scf_max = 120;
    double mix = 0.25;
    double scf_tol = 1e-6;
    if (argc > 1) nr = std::max(100, std::stoi(argv[1]));
    if (argc > 2) rmax = std::max(5.0, std::stod(argv[2]));

    const double dr = rmax / (nr + 1.0);
    const double kin_d = 1.0 / (dr * dr);
    const double kin_o = -0.5 / (dr * dr);
    const double Cx = std::cbrt(3.0 / pi);

    std::vector<double> r(nr), u(nr), rho(nr), rho_new(nr), vH(nr), vx(nr), veff(nr), Hu(nr);
    for (int i = 0; i < nr; ++i) {
        r[i] = (i + 1) * dr;
        u[i] = r[i] * std::exp(-Z * r[i]);
    }

    auto normalize = [&](std::vector<double>& x) {
        double n = 0.0;
        for (double v : x) n += v * v * dr;
        n = std::sqrt(std::max(n, 1e-30));
        for (double& v : x) v /= n;
    };
    normalize(u);

    auto build_density = [&](const std::vector<double>& ur, std::vector<double>& out) {
        for (int i = 0; i < nr; ++i) {
            double rr = r[i];
            out[i] = Ne * ur[i] * ur[i] / (4.0 * pi * rr * rr);
        }
    };
    build_density(u, rho);

    auto build_hartree = [&](const std::vector<double>& den, std::vector<double>& out) {
        double q = 0.0;
        for (int i = 0; i < nr; ++i) {
            q += 4.0 * pi * r[i] * r[i] * den[i] * dr;
            out[i] = q / r[i];
        }
        double tail = 0.0;
        for (int i = nr - 1; i >= 0; --i) {
            out[i] += tail;
            tail += 4.0 * pi * r[i] * den[i] * dr;
        }
    };

    auto apply_h = [&](const std::vector<double>& ur, std::vector<double>& out) {
        for (int i = 0; i < nr; ++i) {
            double left = (i == 0) ? 0.0 : ur[i - 1];
            double right = (i + 1 == nr) ? 0.0 : ur[i + 1];
            out[i] = kin_o * left + (kin_d + veff[i]) * ur[i] + kin_o * right;
        }
    };

    auto rayleigh = [&](const std::vector<double>& ur) {
        apply_h(ur, Hu);
        double num = 0.0, den = 0.0;
        for (int i = 0; i < nr; ++i) {
            num += ur[i] * Hu[i] * dr;
            den += ur[i] * ur[i] * dr;
        }
        return num / std::max(den, 1e-30);
    };

    double eps = 0.0;
    for (int scf = 1; scf <= scf_max; ++scf) {
        build_hartree(rho, vH);
        for (int i = 0; i < nr; ++i) {
            vx[i] = -Cx * std::cbrt(std::max(rho[i], 1e-18));
            veff[i] = -Z / r[i] + vH[i] + vx[i];
        }

        std::vector<double> up = u;
        const double tau = 0.05 * dr * dr;
        for (int it = 0; it < 400; ++it) {
            apply_h(up, Hu);
            for (int i = 0; i < nr; ++i) up[i] -= tau * Hu[i];
            normalize(up);
        }
        u.swap(up);
        eps = rayleigh(u);

        build_density(u, rho_new);
        double err = 0.0;
        for (int i = 0; i < nr; ++i) {
            err += 4.0 * pi * r[i] * r[i] * std::abs(rho_new[i] - rho[i]) * dr;
            rho[i] = (1.0 - mix) * rho[i] + mix * rho_new[i];
        }
        std::cout << "SCF " << std::setw(3) << scf << "  err=" << std::scientific << err / Ne
                  << "  eps1s=" << std::fixed << std::setprecision(8) << eps << " Ha\n";
        if (err / Ne < scf_tol) break;
    }

    build_hartree(rho, vH);
    double EH = 0.0, I43 = 0.0;
    for (int i = 0; i < nr; ++i) {
        EH += 0.5 * 4.0 * pi * r[i] * r[i] * rho[i] * vH[i] * dr;
        I43 += 4.0 * pi * r[i] * r[i] * std::pow(std::max(rho[i], 1e-18), 4.0 / 3.0) * dr;
    }
    double E = 2.0 * eps - EH + 0.25 * Cx * I43;
    const double E_ref = -2.903724377;  // He non-relativistic reference (Ha)

    std::cout << "\nHe (simple LDA-x SCF)\n";
    std::cout << "nr   : " << nr << "\n";
    std::cout << "rmax : " << std::fixed << std::setprecision(1) << rmax << "\n";
    std::cout << "Total energy    : " << std::fixed << std::setprecision(9) << E << " Ha ("
              << E * hartree_to_ev << " eV)\n";
    std::cout << "Reference (exact): " << E_ref << " Ha\n";
    std::cout << "Absolute error  : " << std::abs(E - E_ref) << " Ha\n";

    std::ofstream f("he_dft_min.csv");
    f << "r_bohr,rho,v_eff,v_h,v_x\n";
    for (int i = 0; i < nr; ++i) {
        f << std::setprecision(12) << r[i] << ',' << rho[i] << ','
          << (-Z / r[i] + vH[i] - Cx * std::cbrt(std::max(rho[i], 1e-18))) << ','
          << vH[i] << ',' << (-Cx * std::cbrt(std::max(rho[i], 1e-18))) << '\n';
    }
    std::cout << "Saved: he_dft_min.csv\n";
    return 0;
}
