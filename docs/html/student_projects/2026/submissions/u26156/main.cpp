#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

// Welford's method keeps the mean and variance without storing samples.
struct Statistics {
    std::uint64_t count = 0;
    double mean = 0.0;
    double m2 = 0.0;

    void add(double value) {
        ++count;
        const double delta = value - mean;
        mean += delta / static_cast<double>(count);
        m2 += delta * (value - mean);
        if (!std::isfinite(mean) || !std::isfinite(m2)) {
            throw std::runtime_error("statistics overflowed; choose smaller inputs");
        }
    }

    double standard_error() const {
        return std::sqrt((m2 / static_cast<double>(count - 1)) /
                         static_cast<double>(count));
    }
};

double payoff(double terminal_price, double strike, double discount) {
    return discount * std::max(terminal_price - strike, 0.0);
}

double normal_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double black_scholes(double spot, double strike, double maturity,
                     double rate, double volatility) {
    if (maturity == 0.0) return payoff(spot, strike, 1.0);
    const double discount = std::exp(-rate * maturity);
    if (volatility == 0.0) {
        return std::max(spot - strike * discount, 0.0);
    }
    const double width = volatility * std::sqrt(maturity);
    const double d1 = ((std::log(spot) - std::log(strike)) +
                       (rate + 0.5 * volatility * volatility) * maturity) / width;
    const double d2 = d1 - width;
    return spot * normal_cdf(d1) - strike * discount * normal_cdf(d2);
}

std::uint64_t parse_uint(const char* text) {
    const std::string value(text);
    if (value.empty()) throw std::invalid_argument("empty unsigned input");
    for (char digit : value) {
        if (digit < '0' || digit > '9') {
            throw std::invalid_argument("invalid unsigned input: " + value);
        }
    }
    std::size_t end = 0;
    std::uint64_t result = 0;
    try {
        result = std::stoull(value, &end);
    } catch (const std::out_of_range&) {
        throw std::invalid_argument("unsigned input out of range: " + value);
    }
    if (end != value.size()) throw std::invalid_argument("invalid unsigned input: " + value);
    return result;
}

double parse_double(const char* text) {
    const std::string value(text);
    std::size_t end = 0;
    double result = 0.0;
    try {
        result = std::stod(value, &end);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument("invalid number: " + value);
    } catch (const std::out_of_range&) {
        throw std::invalid_argument("number out of range: " + value);
    }
    if (end != value.size() || !std::isfinite(result)) {
        throw std::invalid_argument("invalid finite number: " + value);
    }
    return result;
}

void self_test() {
    const auto close = [](double actual, double expected, double tolerance) {
        if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance)
            throw std::runtime_error("self-test failed");
    };
    close(payoff(90.0, 100.0, 1.0), 0.0, 0.0);
    close(payoff(110.0, 100.0, 1.0), 10.0, 0.0);
    close(black_scholes(110.0, 100.0, 0.0, 0.05, 0.2), 10.0, 0.0);
    close(black_scholes(110.0, 100.0, 0.0, 0.05, 1e200), 10.0, 0.0);
    close(black_scholes(100.0, 90.0, 1.0, 0.0, 0.0), 10.0, 0.0);
    close(black_scholes(100.0, 100.0, 1.0, 1000.0, 0.0), 100.0, 0.0);
    close(black_scholes(100.0, 100.0, 1.0, 0.05, 0.2),
          10.4505835721856, 1e-10);
    Statistics stats;
    for (double x : {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0}) stats.add(x);
    close(stats.mean, 5.0, 1e-14);
    close(stats.m2 / 7.0, 32.0 / 7.0, 1e-14);
    std::cout << "self-test: PASS\n";
}

int main(int argc, char* argv[]) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            self_test();
            return 0;
        }
        const bool custom_log = argc > 1 && std::string(argv[1]) == "--log";
        const int first = custom_log ? 3 : 1;
        const std::string log_path = custom_log && argc > 2 ? argv[2] : "results.csv";
        if (custom_log && (argc < 3 || log_path.empty())) {
            throw std::invalid_argument("--log requires a nonempty file path");
        }
        if (argc > first + 7) {
            throw std::invalid_argument("usage: ./app [--log FILE] [N [S0 [K [T [r [sigma [seed]]]]]]]");
        }
        const std::uint64_t n = argc > first ? parse_uint(argv[first]) : 1000000;
        const double spot = argc > first + 1 ? parse_double(argv[first + 1]) : 100.0;
        const double strike = argc > first + 2 ? parse_double(argv[first + 2]) : 100.0;
        const double maturity = argc > first + 3 ? parse_double(argv[first + 3]) : 1.0;
        const double rate = argc > first + 4 ? parse_double(argv[first + 4]) : 0.05;
        const double volatility = argc > first + 5 ? parse_double(argv[first + 5]) : 0.20;
        const std::uint64_t seed = argc > first + 6 ? parse_uint(argv[first + 6]) : 42;
        if (n < 2 || spot <= 0.0 || strike <= 0.0 || maturity < 0.0 ||
            volatility < 0.0) {
            throw std::invalid_argument("require N >= 2, S0 > 0, K > 0, T >= 0, sigma >= 0");
        }

        // These limiting cases have a known payoff and need no random draws.
        const bool deterministic = maturity == 0.0 || volatility == 0.0;
        const double drift = deterministic ? 0.0 : (rate - 0.5 * volatility * volatility) * maturity;
        const double width = deterministic ? 0.0 : volatility * std::sqrt(maturity);
        const double discount = std::exp(-rate * maturity);
        const double analytic = black_scholes(spot, strike, maturity, rate, volatility);
        if (!std::isfinite(drift) || !std::isfinite(width) ||
            !std::isfinite(discount) || !std::isfinite(analytic)) {
            throw std::invalid_argument("derived values are not finite; choose smaller inputs");
        }

        std::mt19937_64 generator(seed);
        std::normal_distribution<double> normal(0.0, 1.0);
        double fixed_payoff = 0.0;
        if (deterministic) {
            fixed_payoff = maturity == 0.0 ? payoff(spot, strike, 1.0)
                                            : std::max(spot - strike * discount, 0.0);
        }
        Statistics stats;
        // Measure the trials only; setup and reporting are outside the timer.
        const auto start = std::chrono::steady_clock::now();
        for (std::uint64_t i = 0; i < n; ++i) {
            const double value = deterministic ? fixed_payoff
                : payoff(spot * std::exp(drift + width * normal(generator)), strike, discount);
            if (!std::isfinite(value)) {
                throw std::runtime_error("payoff overflowed; choose smaller inputs");
            }
            stats.add(value);
        }
        const auto stop = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(stop - start).count();
        const double se = stats.standard_error();
        const double lower = stats.mean - 1.96 * se;
        const double upper = stats.mean + 1.96 * se;
        const double ns_per_trial = seconds * 1e9 / static_cast<double>(n);
        const double precision_cost = seconds * se * se;
        if (!std::isfinite(se) || !std::isfinite(lower) || !std::isfinite(upper)) {
            throw std::runtime_error("confidence interval overflowed");
        }
        // Check the existing log format before appending another measurement.
        const std::string header =
            "utc_time,method,workers,N,S0,K,T,r,sigma,seed,mc_price,standard_error,"
            "ci_low,ci_high,bs_price,absolute_error,mc_seconds,trials_per_second,"
            "ns_per_trial,seconds_x_se_squared";
        std::ifstream previous(log_path);
        std::string previous_header;
        const bool has_header = static_cast<bool>(std::getline(previous, previous_header));
        if (has_header && previous_header != header) {
            throw std::runtime_error("CSV header does not match: " + log_path);
        }
        if (has_header) {
            previous.seekg(-1, std::ios::end);
            if (previous.get() != '\n') {
                throw std::runtime_error("CSV log does not end with a newline: " + log_path);
            }
        }
        const std::time_t now = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now());
        const std::tm* utc = std::gmtime(&now);
        if (!utc) throw std::runtime_error("could not get UTC time");
        std::ofstream log(log_path, std::ios::app);
        if (!log) throw std::runtime_error("could not open CSV log: " + log_path);
        if (!has_header) log << header << '\n';
        log << std::setprecision(17) << std::put_time(utc, "%Y-%m-%dT%H:%M:%SZ")
            << ",plain_mc,1," << n << ',' << spot << ',' << strike << ','
            << maturity << ',' << rate << ',' << volatility << ',' << seed << ','
            << stats.mean << ',' << se << ',' << lower << ',' << upper << ','
            << analytic << ',' << std::abs(stats.mean - analytic) << ','
            << seconds << ','
            << (seconds > 0.0 ? static_cast<double>(n) / seconds : 0.0) << ','
            << ns_per_trial << ',' << precision_cost << '\n';
        log.close();
        if (!log) throw std::runtime_error("could not write CSV log: " + log_path);

        std::cout << std::setprecision(12)
                  << "Serial Monte Carlo: N=" << n << " seed=" << seed << '\n'
                  << "Price=" << stats.mean << " +/- " << 1.96 * se
                  << " (approx. 95%); Black-Scholes=" << analytic
                  << "; absolute error=" << std::abs(stats.mean - analytic) << '\n'
                  << "MC time=" << seconds << " s; trials/second="
                  << (seconds > 0.0 ? static_cast<double>(n) / seconds : 0.0) << '\n'
                  << "Detailed CSV log: " << log_path << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
