// sumfact_harness.cpp
//
// Sum-factorized 3D FEM mass-operator kernel, parallelized over elements
// with OpenMP schedule(runtime).  Measures per-time-step wall time and
// per-thread active time to study scheduling overhead on a zero-imbalance
// workload.
//
// Every element performs identical work (same polynomial degree, same
// data size, same FLOPs).  Load imbalance is zero by construction — any
// overhead beyond schedule(static) is pure scheduling cost.
//
// Build:  cmake -S . -B build && cmake --build build
// Run:    OMP_SCHEDULE=static ./build/sumfact_harness [options]
//
// With LB4OMP:
//   LD_LIBRARY_PATH=/path/to/lb4omp/build/runtime/src \
//   OMP_SCHEDULE=6 KMP_CPU_SPEED=2000 \
//   ./build/sumfact_harness [options]

#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// ================================================================
// 1D tensor contraction: V = S * U  (row-major storage)
// S: (n_q × n_d),  U: (n_d × n_cols),  V: (n_q × n_cols)
// Cost: 2 · n_q · n_d · n_cols FLOPs.
// ================================================================
static inline void apply_1d(const double* __restrict__ S,
                            const double* __restrict__ U,
                            double* __restrict__ V,
                            int n_q, int n_d, int n_cols)
{
    for (int q = 0; q < n_q; ++q)
        for (int c = 0; c < n_cols; ++c) {
            double acc = 0.0;
            for (int d = 0; d < n_d; ++d)
                acc += S[q * n_d + d] * U[d * n_cols + c];
            V[q * n_cols + c] = acc;
        }
}

// Transpose contraction: V = S^T * U
static inline void apply_1d_T(const double* __restrict__ S,
                              const double* __restrict__ U,
                              double* __restrict__ V,
                              int n_q, int n_d, int n_cols)
{
    for (int d = 0; d < n_d; ++d)
        for (int c = 0; c < n_cols; ++c) {
            double acc = 0.0;
            for (int q = 0; q < n_q; ++q)
                acc += S[q * n_d + d] * U[q * n_cols + c];
            V[d * n_cols + c] = acc;
        }
}

// ================================================================
// Sum-factorized 3D mass-operator apply on one element.
//
// Evaluate:  u_q = (S ⊗ S ⊗ S) u_h   via 3 sequential 1D contractions
// Scale:     u_q *= JxW               (constant for Cartesian mesh)
// Integrate: v_h = (S^T ⊗ S^T ⊗ S^T) u_q
//
// Field: row-major u[i·m² + j·m + k],  i slowest, k fastest.
// m = degree + 1 (nodes = quadrature points per direction).
//
// FLOP count (model): 12·m⁴ + m³  per element.
// ================================================================
static void element_apply(const double* __restrict__ S,
                          double jxw,
                          const double* __restrict__ u_in,
                          double* __restrict__ u_out,
                          double* __restrict__ t1,
                          double* __restrict__ t2,
                          int m)
{
    const int m2 = m * m;
    const int m3 = m2 * m;

    // --- Evaluate: interpolate to quadrature points ---
    // Direction 0: S applied to first index, (m × m²) batch
    apply_1d(S, u_in, t1, m, m, m2);
    // Direction 1: per first-index slab, (m × m) batch
    for (int i = 0; i < m; ++i)
        apply_1d(S, t1 + i * m2, t2 + i * m2, m, m, m);
    // Direction 2: per (q1,q2) fiber, (m × 1) batch
    for (int i = 0; i < m2; ++i)
        apply_1d(S, t2 + i * m, t1 + i * m, m, m, 1);

    // --- Quadrature-point scaling (mass operator: multiply by JxW) ---
    for (int q = 0; q < m3; ++q)
        t1[q] *= jxw;

    // --- Integrate: project back to nodal values ---
    for (int i = 0; i < m2; ++i)
        apply_1d_T(S, t1 + i * m, t2 + i * m, m, m, 1);
    for (int i = 0; i < m; ++i)
        apply_1d_T(S, t2 + i * m2, t1 + i * m2, m, m, m);
    apply_1d_T(S, t1, u_out, m, m, m2);
}

// ================================================================
// Per-thread accumulator — cache-line padded to avoid false sharing.
// ================================================================
struct alignas(64) ThreadStats {
    double active_us;
    int    n_elems;
};

int main(int argc, char** argv)
{
    int  degree     = 4;
    int  n_elements = 20000;
    int  n_steps    = 200;
    int  warmup     = 3;
    bool verbose    = false;
    std::string csv_path;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--degree"    && i + 1 < argc) degree     = std::atoi(argv[++i]);
        else if (a == "--elements"  && i + 1 < argc) n_elements = std::atoi(argv[++i]);
        else if (a == "--timesteps" && i + 1 < argc) n_steps    = std::atoi(argv[++i]);
        else if (a == "--warmup"    && i + 1 < argc) warmup     = std::atoi(argv[++i]);
        else if (a == "--csv"       && i + 1 < argc) csv_path   = argv[++i];
        else if (a == "--verbose") verbose = true;
        else {
            std::fprintf(stderr,
                "Usage: %s [--degree P] [--elements N] [--timesteps T]\n"
                "          [--warmup W] [--csv PATH] [--verbose]\n", argv[0]);
            return 1;
        }
    }

    const int m       = degree + 1;
    const int m3      = m * m * m;
    const int n_thr   = omp_get_max_threads();
    const double flops_per_elem = 12.0 * std::pow(m, 4) + std::pow(m, 3);
    const double total_flops    = flops_per_elem * n_elements;

    std::printf("Sum-factorized 3D FEM kernel — scheduling experiment\n");
    std::printf("  degree=%d  m=%d  elements=%d  timesteps=%d  threads=%d\n",
                degree, m, n_elements, n_steps, n_thr);
    std::printf("  FLOPs/element=%.0f  total/step=%.3e\n", flops_per_elem, total_flops);
    const char* sched = std::getenv("OMP_SCHEDULE");
    std::printf("  OMP_SCHEDULE=%s\n\n", sched ? sched : "(default)");

    // --- Shape matrix (deterministic, fixed seed) ---
    std::vector<double> S(m * m);
    {
        std::mt19937_64 rng(42);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (auto& v : S) v = dist(rng);
    }

    // --- Element data (contiguous, ping-ponged between steps) ---
    const auto buf_sz = static_cast<size_t>(n_elements) * m3;
    std::vector<double> buf_a(buf_sz), buf_b(buf_sz, 0.0);
    {
        std::mt19937_64 rng(12345);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (auto& v : buf_a) v = dist(rng);
    }

    // --- Per-thread scratch (2 × m³ doubles per thread) ---
    std::vector<double> scratch(static_cast<size_t>(n_thr) * 2 * m3);

    // --- Thread stats ---
    std::vector<ThreadStats> tstats(n_thr);

    const double jxw = 1.0 / n_elements;
    double* in_ptr  = buf_a.data();
    double* out_ptr = buf_b.data();

    // --- Warmup: schedule(static), cache-warming only ---
    for (int w = 0; w < warmup; ++w) {
        #pragma omp parallel for schedule(static)
        for (int e = 0; e < n_elements; ++e) {
            int    tid = omp_get_thread_num();
            double* s1 = scratch.data() + static_cast<size_t>(tid) * 2 * m3;
            double* s2 = s1 + m3;
            element_apply(S.data(), jxw,
                          in_ptr  + static_cast<size_t>(e) * m3,
                          out_ptr + static_cast<size_t>(e) * m3,
                          s1, s2, m);
        }
        std::swap(in_ptr, out_ptr);
    }

    // --- CSV output ---
    std::ofstream csv;
    if (!csv_path.empty()) {
        csv.open(csv_path);
        csv << "step,wall_us,t_min_us,t_max_us,t_mean_us,"
               "imbalance_pct,elems_min,elems_max,gflops\n";
    }

    if (verbose)
        std::printf("%6s %10s %10s %10s %10s %8s %6s %6s %8s\n",
                    "step", "wall_us", "t_min", "t_max", "t_mean",
                    "imb%", "emin", "emax", "GF/s");

    // --- Timed loop ---
    std::vector<double> wall_samples;
    wall_samples.reserve(n_steps);

    for (int step = 0; step < n_steps; ++step) {

        double wall_start = omp_get_wtime();

        #pragma omp parallel
        {
            int tid      = omp_get_thread_num();
            int my_count = 0;
            double my_t0 = omp_get_wtime();

            #pragma omp for schedule(runtime) nowait
            for (int e = 0; e < n_elements; ++e) {
                double* s1 = scratch.data()
                             + static_cast<size_t>(tid) * 2 * m3;
                double* s2 = s1 + m3;
                element_apply(S.data(), jxw,
                              in_ptr  + static_cast<size_t>(e) * m3,
                              out_ptr + static_cast<size_t>(e) * m3,
                              s1, s2, m);
                ++my_count;
            }

            double my_t1 = omp_get_wtime();
            tstats[tid].active_us = (my_t1 - my_t0) * 1e6;
            tstats[tid].n_elems   = my_count;
        }

        double wall_us = (omp_get_wtime() - wall_start) * 1e6;
        wall_samples.push_back(wall_us);

        // Collect per-thread stats
        double t_min = 1e18, t_max = 0.0, t_sum = 0.0;
        int    e_min = n_elements, e_max = 0;
        for (int t = 0; t < n_thr; ++t) {
            if (tstats[t].n_elems == 0) continue;
            t_min = std::min(t_min, tstats[t].active_us);
            t_max = std::max(t_max, tstats[t].active_us);
            t_sum += tstats[t].active_us;
            e_min = std::min(e_min, tstats[t].n_elems);
            e_max = std::max(e_max, tstats[t].n_elems);
        }
        double t_mean = t_sum / n_thr;
        double imb    = (t_mean > 0) ? (t_max - t_min) / t_mean * 100.0 : 0.0;
        double gflops = total_flops / (wall_us * 1e-6) / 1e9;

        if (verbose)
            std::printf("%6d %10.1f %10.1f %10.1f %10.1f %8.2f %6d %6d %8.2f\n",
                        step, wall_us, t_min, t_max, t_mean, imb, e_min, e_max, gflops);

        if (csv.is_open())
            csv << step << ',' << wall_us << ',' << t_min << ',' << t_max
                << ',' << t_mean << ',' << imb << ',' << e_min << ','
                << e_max << ',' << gflops << '\n';

        std::swap(in_ptr, out_ptr);
    }

    // --- Summary ---
    std::sort(wall_samples.begin(), wall_samples.end());
    auto pct = [&](double p) {
        double idx = p * (wall_samples.size() - 1);
        size_t lo  = static_cast<size_t>(std::floor(idx));
        size_t hi  = static_cast<size_t>(std::ceil(idx));
        double f   = idx - lo;
        return wall_samples[lo] * (1.0 - f) + wall_samples[hi] * f;
    };

    std::printf("\nSummary (%d steps):\n", n_steps);
    std::printf("  wall time:  median=%.1f us   p5=%.1f   p95=%.1f\n",
                pct(0.5), pct(0.05), pct(0.95));
    double median_gflops = total_flops / (pct(0.5) * 1e-6) / 1e9;
    std::printf("  throughput: %.2f GFLOP/s (at median)\n", median_gflops);

    // Prevent dead-code elimination
    double sink = 0.0;
    for (size_t i = 0; i < buf_sz; i += m3)
        sink += in_ptr[i];
    std::printf("  sink=%.6e\n", sink);

    if (csv.is_open())
        std::printf("  results: %s\n", csv_path.c_str());

    return 0;
}
