#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

//TODO: Change name of variables

typedef struct {
  int nx;
  int ny;
  double lx_um;
  double tox_nm;
  double tsi_um;
  double vg;
  double vsub;
  double na_cm3;
  double nd_cm3;
  double ni_cm3;
  double temp_k;
  int max_iters;
  int inner_jacobi;
  double tol_update;
  double tol_residual;
  double omega;
  int has_out_phi;
  char out_phi[512];
} Params;

static const double KQ = 1.602176634e-19;
static const double KKB = 1.380649e-23;
static const double KEPS0 = 8.8541878128e-12;

//TODO: Move not important helper functions after main()

static void init_params(Params *p) {
  p->nx = 257;
  p->ny = 257;
  p->lx_um = 1.0;
  p->tox_nm = 10.0;
  p->tsi_um = 0.5;
  p->vg = 1.0;
  p->vsub = 0.0;
  p->na_cm3 = 1.0e16;
  p->nd_cm3 = 0.0;
  p->ni_cm3 = 1.0e10;
  p->temp_k = 300.0;
  p->max_iters = 1500;
  p->inner_jacobi = 10;
  p->tol_update = 1.0e-7;
  p->tol_residual = 1.0e-2;
  p->omega = 0.8;
  p->has_out_phi = 0;
  p->out_phi[0] = '\0';
}

static void usage(const char *prog) {
  fprintf(stderr,
          "Usage: %s [options]\n"
          "Options:\n"
          "  --nx INT              Grid size in x (default 257)\n"
          "  --ny INT              Grid size in y (default 257)\n"
          "  --lx-um FLOAT         Domain width in um (default 1.0)\n"
          "  --tox-nm FLOAT        Oxide thickness in nm (default 10.0)\n"
          "  --tsi-um FLOAT        Silicon thickness in um (default 0.5)\n"
          "  --vg FLOAT            Gate voltage in V (default 1.0)\n"
          "  --vsub FLOAT          Substrate voltage in V (default 0.0)\n"
          "  --na-cm3 FLOAT        Acceptor density in cm^-3 (default 1e16)\n"
          "  --nd-cm3 FLOAT        Donor density in cm^-3 (default 0)\n"
          "  --ni-cm3 FLOAT        Intrinsic density in cm^-3 (default 1e10)\n"
          "  --temp-k FLOAT        Temperature in K (default 300)\n"
          "  --max-iters INT       Max outer nonlinear iterations (default 1500)\n"
          "  --inner-jacobi INT    Jacobi sweeps per outer iter (default 10)\n"
          "  --tol-update FLOAT    Convergence for max update in V (default 1e-7)\n"
          "  --tol-residual FLOAT  Convergence for max residual in C/m^3 (default 1e-2)\n"
          "  --omega FLOAT         Jacobi damping factor in (0,1] (default 0.8)\n"
          "  --out-phi FILE        Optional output of phi grid as CSV\n"
          "  --help                Show this message\n",
          prog);
}

static int parse_int(const char *s, int *v) {
  char *end = NULL;
  long x;
  errno = 0;
  x = strtol(s, &end, 10);
  if (errno != 0 || end == s || *end != '\0') return 0;
  if (x < INT_MIN || x > INT_MAX) return 0;
  *v = (int)x;
  return 1;
}

static int parse_double(const char *s, double *v) {
  char *end = NULL;
  double x;
  errno = 0;
  x = strtod(s, &end);
  if (errno != 0 || end == s || *end != '\0' || !isfinite(x)) return 0;
  *v = x;
  return 1;
}

static double harmonic_mean(double a, double b) {
  double s = a + b;
  if (s == 0.0) return 0.0;
  return 2.0 * a * b / s;
}

static double clamp_exp_arg(double x) {
  if (x > 80.0) return 80.0;
  if (x < -80.0) return -80.0;
  return x;
}

static int parse_args(int argc, char **argv, Params *p) {
  int i;
  for (i = 1; i < argc; ++i) {
    const char *a = argv[i];
    const char *v = NULL;

    if (strcmp(a, "--help") == 0) {
      usage(argv[0]);
      return 2;
    }

    if (strcmp(a, "--nx") == 0 || strcmp(a, "--ny") == 0 || strcmp(a, "--lx-um") == 0 ||
        strcmp(a, "--tox-nm") == 0 || strcmp(a, "--tsi-um") == 0 || strcmp(a, "--vg") == 0 ||
        strcmp(a, "--vsub") == 0 || strcmp(a, "--na-cm3") == 0 || strcmp(a, "--nd-cm3") == 0 ||
        strcmp(a, "--ni-cm3") == 0 || strcmp(a, "--temp-k") == 0 ||
        strcmp(a, "--max-iters") == 0 || strcmp(a, "--inner-jacobi") == 0 ||
        strcmp(a, "--tol-update") == 0 || strcmp(a, "--tol-residual") == 0 ||
        strcmp(a, "--omega") == 0 || strcmp(a, "--out-phi") == 0) {
      if (i + 1 >= argc) {
        fprintf(stderr, "Missing value for %s\n", a);
        return 0;
      }
      v = argv[++i];
    }

    if (strcmp(a, "--nx") == 0) {
      if (!parse_int(v, &p->nx)) return 0;
    } else if (strcmp(a, "--ny") == 0) {
      if (!parse_int(v, &p->ny)) return 0;
    } else if (strcmp(a, "--lx-um") == 0) {
      if (!parse_double(v, &p->lx_um)) return 0;
    } else if (strcmp(a, "--tox-nm") == 0) {
      if (!parse_double(v, &p->tox_nm)) return 0;
    } else if (strcmp(a, "--tsi-um") == 0) {
      if (!parse_double(v, &p->tsi_um)) return 0;
    } else if (strcmp(a, "--vg") == 0) {
      if (!parse_double(v, &p->vg)) return 0;
    } else if (strcmp(a, "--vsub") == 0) {
      if (!parse_double(v, &p->vsub)) return 0;
    } else if (strcmp(a, "--na-cm3") == 0) {
      if (!parse_double(v, &p->na_cm3)) return 0;
    } else if (strcmp(a, "--nd-cm3") == 0) {
      if (!parse_double(v, &p->nd_cm3)) return 0;
    } else if (strcmp(a, "--ni-cm3") == 0) {
      if (!parse_double(v, &p->ni_cm3)) return 0;
    } else if (strcmp(a, "--temp-k") == 0) {
      if (!parse_double(v, &p->temp_k)) return 0;
    } else if (strcmp(a, "--max-iters") == 0) {
      if (!parse_int(v, &p->max_iters)) return 0;
    } else if (strcmp(a, "--inner-jacobi") == 0) {
      if (!parse_int(v, &p->inner_jacobi)) return 0;
    } else if (strcmp(a, "--tol-update") == 0) {
      if (!parse_double(v, &p->tol_update)) return 0;
    } else if (strcmp(a, "--tol-residual") == 0) {
      if (!parse_double(v, &p->tol_residual)) return 0;
    } else if (strcmp(a, "--omega") == 0) {
      if (!parse_double(v, &p->omega)) return 0;
    } else if (strcmp(a, "--out-phi") == 0) {
      snprintf(p->out_phi, sizeof(p->out_phi), "%s", v);
      p->has_out_phi = 1;
    } else {
      fprintf(stderr, "Unknown option: %s\n", a);
      return 0;
    }
  }

  if (p->nx < 3 || p->ny < 3 || p->lx_um <= 0.0 || p->tox_nm <= 0.0 || p->tsi_um <= 0.0 ||
      p->temp_k <= 0.0 || p->ni_cm3 <= 0.0 || p->max_iters <= 0 || p->inner_jacobi <= 0 ||
      p->omega <= 0.0 || p->omega > 1.0 || p->na_cm3 < 0.0 || p->nd_cm3 < 0.0) {
    fprintf(stderr, "Invalid parameter values. Run with --help for details.\n");
    return 0;
  }
  return 1;
}

static void apply_boundaries(int nx, int ny, double vg, double vsub, double** phi) {
  int i, j;
  for (i = 0; i < nx; ++i) {
    phi[0][i] = vg;
    phi[ny - 1][i] = vsub;
  }
  for (j = 1; j < ny - 1; ++j) {
    phi[j][0] = phi[j][1];
    phi[j][nx - 1] = phi[j][nx - 2];
  }
}

int main(int argc, char **argv) {
  Params p;
  int parse_status;

  int i, j, iter, sweep;
  int j_interface;
  double lx, ly, tox, dx, dy;
  double eps_ox, eps_si, vt;
  double na, nd, ni;
  double dop, p0, n0;
  int iters_done = 0;
  double final_max_update = 0.0;
  double final_max_residual = 0.0;
  clock_t t0, t1;
  double seconds;

  init_params(&p);
  parse_status = parse_args(argc, argv, &p);
  if (parse_status == 2) return 0;
  if (parse_status == 0) return 1;

  lx = p.lx_um * 1.0e-6;
  ly = p.tox_nm * 1.0e-9 + p.tsi_um * 1.0e-6;
  tox = p.tox_nm * 1.0e-9;
  dx = lx / (double)(p.nx - 1);
  dy = ly / (double)(p.ny - 1);

//TODO: Add variables (relative permittivity)
      
  eps_ox = 3.9 * KEPS0;
  eps_si = 11.7 * KEPS0;
  vt = KKB * p.temp_k / KQ;

  na = p.na_cm3 * 1.0e6;
  nd = p.nd_cm3 * 1.0e6;
  ni = p.ni_cm3 * 1.0e6;
  dop = nd - na;
  p0 = 0.5 * (sqrt(dop * dop + 4.0 * ni * ni) - dop);
  n0 = ni * ni / p0;

  j_interface = 0;
  for (j = 0; j < p.ny; ++j) {
    if ((double)j * dy >= tox) {
      j_interface = j;
      break;
    }
  }
  if (j_interface <= 0) j_interface = 1;
  if (j_interface >= p.ny - 1) j_interface = p.ny - 2;

  {
    double** phi = (double**)malloc(sizeof(double*) * p.ny);
    double** phi_new = (double**)malloc(sizeof(double*) * p.ny);
    double** eps = (double**)malloc(sizeof(double*) * p.ny);
    unsigned char** is_si = (unsigned char**)malloc(sizeof(unsigned char*) * p.ny);

    for (j = 0; j < p.ny; ++j) {
      phi[j] = (double*)malloc(sizeof(double) * p.nx);
      phi_new[j] = (double*)malloc(sizeof(double) * p.nx);
      eps[j] = (double*)malloc(sizeof(double) * p.nx);
      is_si[j] = (unsigned char*)malloc(sizeof(unsigned char) * p.nx);
    }

    if (!phi || !phi_new || !eps || !is_si) {
      fprintf(stderr, "Memory allocation failed.\n");
      free(phi);
      free(phi_new);
      free(eps);
      free(is_si);
      return 1;
    }

    for (j = 0; j < p.ny; ++j) {
      int in_ox = ((double)j * dy < tox) ? 1 : 0;
      for (i = 0; i < p.nx; ++i) {
        eps[j][i] = in_ox ? eps_ox : eps_si;
        is_si[j][i] = (unsigned char)(in_ox ? 0 : 1);
        phi[j][i] = 0.0;
        phi_new[j][i] = 0.0;
      }
    }

    apply_boundaries(p.nx, p.ny, p.vg, p.vsub, phi);
    apply_boundaries(p.nx, p.ny, p.vg, p.vsub, phi_new);

    t0 = clock();

//###########################
//Start of the main iteration
//###########################
    
    for (iter = 1; iter <= p.max_iters; ++iter) {
      double max_update_outer = 0.0;
      double max_residual_outer = 0.0;
      
//Why there are double loops?
      
      for (sweep = 0; sweep < p.inner_jacobi; ++sweep) {
        double max_update_sweep = 0.0;
        double max_residual_sweep = 0.0;

        apply_boundaries(p.nx, p.ny, p.vg, p.vsub, phi);
//For every grid points:
        for (j = 1; j < p.ny - 1; ++j) {
          for (i = 1; i < p.nx - 1; ++i) {
            double ee, ew, en, es;
            double a_e, a_w, a_n, a_s, a_p;
            double sum_nb;
            double rho = 0.0;
            double drho = 0.0;
            double g, gp, phi_lin, updated, delta;

            ee = harmonic_mean(eps[j][i], eps[j][i + 1]);
            ew = harmonic_mean(eps[j][i], eps[j][i - 1]);
            en = harmonic_mean(eps[j][i], eps[j + 1][i]);
            es = harmonic_mean(eps[j][i], eps[j - 1][i]);

            a_e = ee / (dx * dx);
            a_w = ew / (dx * dx);
            a_n = en / (dy * dy);
            a_s = es / (dy * dy);
            a_p = a_e + a_w + a_n + a_s;

            sum_nb = a_e * phi[j][i + 1] + a_w * phi[j][i - 1] + a_n * phi[j + 1][i] +
                     a_s * phi[j - 1][i];
//calculate rho from phi
            if (is_si[j][i]) {
              double psi = phi[j][i] - p.vsub;
              double ea = exp(clamp_exp_arg(psi / vt));
              double eb = exp(clamp_exp_arg(-psi / vt));
              rho = KQ * (p0 * eb - n0 * ea + dop);
              drho = -(KQ / vt) * (p0 * eb + n0 * ea);
            }
//calculate residual
            g = a_p * phi[j][i] - sum_nb - rho;
            gp = a_p - drho;
//calculate amout of update
            phi_lin = (sum_nb + rho - drho * phi[j][i]) / gp; // Use -g/gp instead
            updated = (1.0 - p.omega) * phi[j][i] + p.omega * phi_lin;
//update
            phi_new[j][i] = updated;

            delta = fabs(updated - phi[j][i]);
            if (delta > max_update_sweep) max_update_sweep = delta;
            if (fabs(g) > max_residual_sweep) max_residual_sweep = fabs(g);
          }
        }

        apply_boundaries(p.nx, p.ny, p.vg, p.vsub, phi_new);
//Copy phi_new to phi
//TODO: just swap phi and phi_new, no need to copy entire thing
        for (j = 1; j < p.ny - 1; ++j) {
          for (i = 1; i < p.nx - 1; ++i) {
            phi[j][i] = phi_new[j][i];
          }
        }

        if (max_update_sweep > max_update_outer) max_update_outer = max_update_sweep;
        if (max_residual_sweep > max_residual_outer) max_residual_outer = max_residual_sweep;
      }

      iters_done = iter;
      final_max_update = max_update_outer;
      final_max_residual = max_residual_outer;
//Convergence check
      if (final_max_update < p.tol_update && final_max_residual < p.tol_residual) {
        break;
      }
    }

//###########################
//End of the main iteration
//###########################

    t1 = clock();
    seconds = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    {
      int jc = j_interface;
      int jsi = j_interface + 1;
      double avg_surface_phi = 0.0;
      double avg_oxide_field = 0.0;
      double avg_si_charge = 0.0;
      int si_count = 0;

      if (jsi > p.ny - 2) jsi = p.ny - 2;

      for (i = 0; i < p.nx; ++i) {
        avg_surface_phi += phi[jsi][i];
        avg_oxide_field += -(phi[jc][i] - phi[jc - 1][i]) / dy;
      }
      avg_surface_phi /= (double)p.nx;
      avg_oxide_field /= (double)p.nx;

      for (j = j_interface; j < p.ny; ++j) {
        for (i = 0; i < p.nx; ++i) {
          double psi = phi[j][i] - p.vsub;
          double ea = exp(clamp_exp_arg(psi / vt));
          double eb = exp(clamp_exp_arg(-psi / vt));
          double rho = KQ * (p0 * eb - n0 * ea + dop);
          avg_si_charge += rho;
          si_count += 1;
        }
      }
      if (si_count > 0) avg_si_charge /= (double)si_count;

      printf("model=2d_mos_nonlinear_poisson_jacobi\n");
      printf("grid=%dx%d\n", p.nx, p.ny);
      printf("geometry_lx_um=%.10g tox_nm=%.10g tsi_um=%.10g\n", p.lx_um, p.tox_nm, p.tsi_um);
      printf("bias_vg=%.10g vsub=%.10g\n", p.vg, p.vsub);
      printf("doping_na_cm3=%.10g nd_cm3=%.10g ni_cm3=%.10g\n", p.na_cm3, p.nd_cm3, p.ni_cm3);
      printf("outer_iterations=%d\n", iters_done);
      printf("inner_jacobi=%d\n", p.inner_jacobi);
      printf("time_main_seconds=%.10g\n", seconds);
      printf("max_update_V=%.10g\n", final_max_update);
      printf("max_residual_C_per_m3=%.10g\n", final_max_residual);
      printf("avg_surface_phi_V=%.10g\n", avg_surface_phi);
      printf("avg_oxide_Ey_V_per_m=%.10g\n", avg_oxide_field);
      printf("avg_silicon_rho_C_per_m3=%.10g\n", avg_si_charge);
      if (iters_done == p.max_iters &&
          !(final_max_update < p.tol_update && final_max_residual < p.tol_residual)) {
        printf("converged=0\n");
      } else {
        printf("converged=1\n");
      }
    }

    if (p.has_out_phi) {
      FILE *fp = fopen(p.out_phi, "w");
      if (fp == NULL) {
        fprintf(stderr, "Failed to open output file: %s\n", p.out_phi);
        free(phi);
        free(phi_new);
        free(eps);
        free(is_si);
        return 1;
      }
      fprintf(fp, "# x_m,y_m,phi_V,material\n");
      for (j = 0; j < p.ny; ++j) {
        double y = (double)j * dy;
        for (i = 0; i < p.nx; ++i) {
          double x = (double)i * dx;
          fprintf(fp, "%.17g,%.17g,%.17g,%s\n", x, y, phi[j][i], is_si[j][i] ? "si" : "ox");
        }
      }
      fclose(fp);
    }

    free(phi);
    free(phi_new);
    free(eps);
    free(is_si);
  }

  return 0;
}
