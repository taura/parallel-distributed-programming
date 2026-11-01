// Serial PPO baseline. C++17, standard library only.
// All maps, trajectories, and evaluation episodes are generated locally.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr int MAP_START = 4;
constexpr int ACTIONS = 4;
constexpr int DX[ACTIONS] = {0, 0, -1, 1};
constexpr int DY[ACTIONS] = {-1, 1, 0, 0};
constexpr double GAMMA = 0.95;
constexpr double GAE_LAMBDA = 0.95;
constexpr double CLIP_EPS = 0.2;

struct Options {
    int updates = 2000, envs = 16, rollout = 16, epochs = 3, minibatch = 128;
    int hidden = 64, num_layers = 1, log_every = 100, eval_episodes = 100;
    int size = 5, obstacles = -1, horizon = 0;
    // 0 / -1 select size-dependent defaults.
    unsigned seed = 12345;
};

int positive(const std::string& s) {
    size_t used = 0;
    const long long v = std::stoll(s, &used);
    if (used != s.size() || v < 1 || v > 100000000)
        throw std::invalid_argument("expected a positive integer: " + s);
    return static_cast<int>(v);
}

int nonnegative(const std::string& s) {
    size_t used = 0;
    const long long v = std::stoll(s, &used);
    if (used != s.size() || v < 0 || v > 100000000)
        throw std::invalid_argument("expected a nonnegative integer: " + s);
    return static_cast<int>(v);
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) throw std::invalid_argument("missing value after " + std::string(argv[i]));
        const std::string k = argv[i], v = argv[i + 1];
        if (k == "--updates") o.updates = positive(v);
        else if (k == "--envs") o.envs = positive(v);
        else if (k == "--rollout") o.rollout = positive(v);
        else if (k == "--epochs") o.epochs = positive(v);
        else if (k == "--minibatch") o.minibatch = positive(v);
        else if (k == "--hidden") o.hidden = positive(v);
        else if (k == "--num-layers") o.num_layers = positive(v);
        else if (k == "--log-every") o.log_every = positive(v);
        else if (k == "--eval-episodes") o.eval_episodes = positive(v);
        else if (k == "--size") o.size = positive(v);
        else if (k == "--obstacles") o.obstacles = nonnegative(v);
        else if (k == "--horizon") o.horizon = positive(v);
        else if (k == "--seed") o.seed = static_cast<unsigned>(positive(v));
        else throw std::invalid_argument("unknown option: " + k);
    }
    if (o.size < 3 || o.size > 64) throw std::invalid_argument("--size must be in [3,64]");
    if (o.num_layers > 8) throw std::invalid_argument("--num-layers must be in [1,8]");
    if (o.obstacles < 0) o.obstacles = (3 * o.size * o.size + 5) / 10;
    if (o.obstacles > o.size * o.size - 2)
        throw std::invalid_argument("--obstacles must leave at least two free squares");
    if (o.horizon == 0) o.horizon = o.size * o.size;
    if (o.horizon < 2) throw std::invalid_argument("--horizon must be >= 2");
    return o;
}

using Observation = std::vector<double>;

// One robot in a new randomly generated, reachable obstacle map per episode.
struct Environment {
    std::mt19937 rng;
    int side, horizon, obstacles;
    std::vector<unsigned char> wall;
    int pos = 0, goal = 0, steps = 0;
    double episode_return = 0;

    Environment(unsigned seed, int size, int walls, int limit) : rng(seed), side(size),
        horizon(limit), obstacles(walls), wall(size * size) { reset(); }

    bool reachable(int from, int to) const {
        std::vector<int> queue(side * side);
        std::vector<int> depth(side * side, -1);
        int front = 0, back = 0;
        queue[back++] = from;
        depth[from] = 0;
        while (front != back) {
            const int p = queue[front++];
            if (p == to) return true;
            if (depth[p] >= horizon) continue;
            for (int a = 0; a < ACTIONS; ++a) {
                const int x = p % side + DX[a], y = p / side + DY[a];
                if (x < 0 || x >= side || y < 0 || y >= side) continue;
                const int q = y * side + x;
                if (depth[q] == -1 && !wall[q]) {
                    depth[q] = depth[p] + 1;
                    queue[back++] = q;
                }
            }
        }
        return false;
    }

    void reset() {
        std::uniform_int_distribution<int> square(0, side * side - 1);
        for (int maps = 0; maps < 1000; ++maps) {
            std::fill(wall.begin(), wall.end(), 0);
            for (int i = 0; i < obstacles;) {
                const int cell = square(rng);
                if (wall[cell]) continue;
                wall[cell] = 1;
                ++i;
            }
            std::vector<int> free_cells;
            for (int cell = 0; cell < side * side; ++cell)
                if (!wall[cell]) free_cells.push_back(cell);
            std::uniform_int_distribution<int> free_square(0, static_cast<int>(free_cells.size()) - 1);
            for (int pairs = 0; pairs < 100; ++pairs) {
                pos = free_cells[free_square(rng)];
                goal = free_cells[free_square(rng)];
                if (pos == goal || !reachable(pos, goal)) continue;
                steps = 0;
                episode_return = 0;
                return;
            }
        }
        throw std::runtime_error("could not generate a reachable map; lower --obstacles or increase --horizon");
    }

    Observation observe() const {
        Observation o(MAP_START + side * side);
        const double scale = static_cast<double>(side - 1);
        o[0] = (goal % side) / scale;
        o[1] = (goal / side) / scale;
        o[2] = (pos % side) / static_cast<double>(side - 1);
        o[3] = (pos / side) / static_cast<double>(side - 1);
        for (int i = 0; i < side * side; ++i) o[MAP_START + i] = wall[i];
        return o;
    }

    struct Result { double reward; bool done, success; double total_return; };

    Result step(int action) {
        int x = pos % side + DX[action], y = pos / side + DY[action];
        double reward = -0.01; // No distance-to-goal or shortest-path reward.
        if (x < 0 || x >= side || y < 0 || y >= side || wall[y * side + x]) {
            reward -= 0.02;
        } else {
            pos = y * side + x;
        }
        ++steps;
        const bool success = pos == goal;
        if (success) reward += 1.0;
        episode_return += reward;
        const bool done = success || steps >= horizon;
        return {reward, done, success, episode_return};
    }
};

struct Network {
    int hidden, input, layers;
    std::vector<std::vector<double>> w, b;
    std::vector<double> wp, bp, wv;
    double bv = 0;

    Network(int h, int n, int depth) : hidden(h), input(n), layers(depth),
        w(depth), b(depth), wp(ACTIONS * h), bp(ACTIONS), wv(h) {
        for (int layer = 0; layer < layers; ++layer) {
            w[layer].resize(hidden * (layer == 0 ? input : hidden));
            b[layer].resize(hidden);
        }
    }

    void initialize(std::mt19937& rng) {
        std::normal_distribution<double> normal(0.0, 1.0);
        for (int layer = 0; layer < layers; ++layer)
            for (double& x : w[layer])
                x = normal(rng) * std::sqrt(2.0 / (layer == 0 ? input : hidden));
        for (double& x : wp) x = normal(rng) * 0.02;
        for (double& x : wv) x = normal(rng) * 0.02;
    }

    struct Output {
        std::vector<std::vector<double>> hidden;
        std::array<double, ACTIONS> prob{};
        double value = 0;
    };

    Output forward(const Observation& obs) const {
        Output out;
        out.hidden.resize(layers);
        for (int layer = 0; layer < layers; ++layer) {
            auto& activation = out.hidden[layer];
            const auto& source = layer == 0 ? obs : out.hidden[layer - 1];
            const int width = layer == 0 ? input : hidden;
            activation.resize(hidden);
            for (int h = 0; h < hidden; ++h) {
                double z = b[layer][h];
                for (int j = 0; j < width; ++j) z += w[layer][h * width + j] * source[j];
                activation[h] = std::max(0.0, z);
            }
        }
        const auto& last = out.hidden.back();
        for (int h = 0; h < hidden; ++h) out.value += wv[h] * last[h];
        out.value += bv;
        double maximum = -std::numeric_limits<double>::infinity();
        for (int a = 0; a < ACTIONS; ++a) {
            double z = bp[a];
            for (int h = 0; h < hidden; ++h) z += wp[a * hidden + h] * last[h];
            out.prob[a] = z;
            maximum = std::max(maximum, z);
        }
        double sum = 0;
        for (double& p : out.prob) { p = std::exp(p - maximum); sum += p; }
        for (double& p : out.prob) p /= sum;
        return out;
    }
};

struct Transition {
    Observation obs;
    int action;
    double reward, old_prob, old_value;
    bool done;
    double advantage = 0, target = 0;
};

struct Trajectory {
    std::vector<Transition> data;
    Observation last_obs{};
    bool last_done = false;
};

struct Worker {
    Environment env;
    std::mt19937 rng;

    Worker(unsigned seed, int side, int obstacles, int horizon)
        : env(seed, side, obstacles, horizon), rng(seed ^ 0x9e3779b9u) {}

    Trajectory collect(const Network& policy, int length, int& completed, double& returns) {
        Trajectory trajectory;
        trajectory.data.reserve(length);
        for (int t = 0; t < length; ++t) {
            const Observation obs = env.observe();
            const auto prediction = policy.forward(obs);
            std::discrete_distribution<int> choice(prediction.prob.begin(), prediction.prob.end());
            const int a = choice(rng);
            const auto result = env.step(a);
            trajectory.data.push_back({obs, a, result.reward, prediction.prob[a], prediction.value,
                                       result.done});
            if (result.done) {
                ++completed;
                returns += result.total_return;
                env.reset();
            }
            trajectory.last_done = result.done;
        }
        trajectory.last_obs = env.observe();
        return trajectory;
    }
};

struct Gradient {
    Network g;
    Gradient(int hidden, int input, int layers) : g(hidden, input, layers) {}

    void add(const Network& net, const Observation& obs, const Network::Output& out,
             int action, double policy_gradient, double target) {
        std::vector<double> dh(net.hidden, 0.0);
        const auto& last = out.hidden.back();
        for (int a = 0; a < ACTIONS; ++a) {
            const double dlogit = policy_gradient * (out.prob[a] - (a == action ? 1.0 : 0.0));
            g.bp[a] += dlogit;
            for (int h = 0; h < net.hidden; ++h) {
                g.wp[a * net.hidden + h] += dlogit * last[h];
                dh[h] += dlogit * net.wp[a * net.hidden + h];
            }
        }
        const double dv = 0.5 * (out.value - target); // 0.5 * 0.5 * (V-target)^2
        g.bv += dv;
        for (int h = 0; h < net.hidden; ++h) {
            g.wv[h] += dv * last[h];
            dh[h] += dv * net.wv[h];
        }
        for (int layer = net.layers - 1; layer >= 0; --layer) {
            const auto& source = layer == 0 ? obs : out.hidden[layer - 1];
            const int width = layer == 0 ? net.input : net.hidden;
            std::vector<double> previous(layer == 0 ? 0 : width, 0.0);
            for (int h = 0; h < net.hidden; ++h) {
                if (out.hidden[layer][h] <= 0) continue;
                g.b[layer][h] += dh[h];
                for (int j = 0; j < width; ++j) {
                    g.w[layer][h * width + j] += dh[h] * source[j];
                    if (layer > 0) previous[j] += net.w[layer][h * width + j] * dh[h];
                }
            }
            if (layer > 0) dh = std::move(previous);
        }
    }

    void scale(double factor) {
        for (int layer = 0; layer < g.layers; ++layer) {
            for (double& x : g.w[layer]) x *= factor;
            for (double& x : g.b[layer]) x *= factor;
        }
        for (double& x : g.wp) x *= factor;
        for (double& x : g.bp) x *= factor;
        for (double& x : g.wv) x *= factor;
        g.bv *= factor;
    }
};

// Adam keeps running moments of the gradients for each parameter.
struct Adam {
    Network m, v;
    int step = 0;
    Adam(int h, int n, int layers) : m(h, n, layers), v(h, n, layers) {}

    void update_vector(std::vector<double>& p, const std::vector<double>& g,
                       std::vector<double>& mm, std::vector<double>& vv, double rate) {
        for (size_t i = 0; i < p.size(); ++i) {
            mm[i] = 0.9 * mm[i] + 0.1 * g[i];
            vv[i] = 0.999 * vv[i] + 0.001 * g[i] * g[i];
            p[i] -= rate * (mm[i] / (1.0 - std::pow(0.9, step))) /
                    (std::sqrt(vv[i] / (1.0 - std::pow(0.999, step))) + 1e-8);
        }
    }

    void apply(Network& net, const Gradient& grad) {
        ++step;
        const double rate = 0.0003;
        for (int layer = 0; layer < net.layers; ++layer) {
            update_vector(net.w[layer], grad.g.w[layer], m.w[layer], v.w[layer], rate);
            update_vector(net.b[layer], grad.g.b[layer], m.b[layer], v.b[layer], rate);
        }
        update_vector(net.wp, grad.g.wp, m.wp, v.wp, rate);
        update_vector(net.bp, grad.g.bp, m.bp, v.bp, rate);
        update_vector(net.wv, grad.g.wv, m.wv, v.wv, rate);
        std::vector<double> p{net.bv}, g{grad.g.bv}, mm{m.bv}, vv{v.bv};
        update_vector(p, g, mm, vv, rate);
        net.bv = p[0]; m.bv = mm[0]; v.bv = vv[0];
    }
};

// Estimate advantages under the policy that generated the trajectories.
// An episode ending cuts the GAE trace; an unfinished rollout is bootstrapped.
std::vector<Transition> prepare_batch(const Network& policy, std::vector<Trajectory>& trajectories) {
    std::vector<Transition> samples;
    for (auto& traj : trajectories) {
        const int n = static_cast<int>(traj.data.size());
        double next_value = traj.last_done ? 0.0 : policy.forward(traj.last_obs).value;
        double gae = 0.0;
        for (int t = n - 1; t >= 0; --t) {
            auto& tr = traj.data[t];
            const double live = tr.done ? 0.0 : 1.0;
            const double delta = tr.reward + GAMMA * live * next_value - tr.old_value;
            gae = delta + GAMMA * GAE_LAMBDA * live * gae;
            tr.advantage = gae;
            tr.target = tr.old_value + gae;
            next_value = tr.old_value;
        }
        for (auto& tr : traj.data) samples.push_back(std::move(tr));
    }
    // Keep the value targets unnormalized; normalize only the policy advantages.
    double mean = 0;
    for (const auto& tr : samples) mean += tr.advantage;
    mean /= samples.size();
    double variance = 0;
    for (const auto& tr : samples) variance += (tr.advantage - mean) * (tr.advantage - mean);
    const double scale = 1.0 / std::sqrt(variance / samples.size() + 1e-8);
    for (auto& tr : samples) tr.advantage = (tr.advantage - mean) * scale;
    return samples;
}

// Reuse each on-policy batch for several epochs of shuffled minibatch updates.
// On the clipped side of PPO's surrogate objective, its policy gradient is zero.
void learn_minibatch(Network& net, Adam& optimizer, const std::vector<Transition>& samples,
                     const std::vector<size_t>& order, size_t begin, size_t end) {
    Gradient grad(net.hidden, net.input, net.layers);
    for (size_t i = begin; i < end; ++i) {
        const auto& tr = samples[order[i]];
        const auto prediction = net.forward(tr.obs);
        const double ratio = prediction.prob[tr.action] / tr.old_prob;
        const bool clipped = (tr.advantage > 0 && ratio > 1.0 + CLIP_EPS) ||
                             (tr.advantage < 0 && ratio < 1.0 - CLIP_EPS);
        grad.add(net, tr.obs, prediction, tr.action,
                 clipped ? 0.0 : ratio * tr.advantage, tr.target);
    }
    grad.scale(1.0 / (end - begin));
    optimizer.apply(net, grad);
}

struct Evaluation { double mean_return, success_rate, mean_steps; };

Evaluation evaluate(const Network& net, unsigned seed, int episodes, const Options& opt) {
    Environment env(seed, opt.size, opt.obstacles, opt.horizon);
    double returns = 0, steps = 0;
    int wins = 0;
    for (int e = 0; e < episodes; ++e) {
        for (;;) {
            const auto prediction = net.forward(env.observe());
            const int action = static_cast<int>(std::max_element(
                prediction.prob.begin(), prediction.prob.end()) - prediction.prob.begin());
            const auto result = env.step(action);
            if (result.done) {
                returns += result.total_return;
                wins += result.success;
                steps += env.steps;
                env.reset();
                break;
            }
        }
    }
    return {returns / episodes, static_cast<double>(wins) / episodes, steps / episodes};
}

double seconds(std::chrono::steady_clock::duration duration) {
    return std::chrono::duration<double>(duration).count();
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options opt = parse(argc, argv);
        std::mt19937 initializer(opt.seed);
        Network policy(opt.hidden, MAP_START + opt.size * opt.size, opt.num_layers);
        policy.initialize(initializer);
        Adam optimizer(opt.hidden, policy.input, opt.num_layers);
        std::vector<Worker> workers;
        workers.reserve(opt.envs);
        for (int i = 0; i < opt.envs; ++i)
            workers.emplace_back(opt.seed + 1000u + static_cast<unsigned>(i),
                                 opt.size, opt.obstacles, opt.horizon);

        std::printf("%dx%d navigation, serial PPO, envs=%d rollout=%d epochs=%d minibatch=%d "
                    "hidden=%d num_layers=%d obstacles=%d horizon=%d seed=%u\n",
                    opt.size, opt.size, opt.envs, opt.rollout, opt.epochs, opt.minibatch,
                    opt.hidden, opt.num_layers, opt.obstacles, opt.horizon, opt.seed);
        std::printf("update training_s eval_return eval_success train_return\n");
        std::fflush(stdout);

        double rollout_s = 0, learner_s = 0, interval_returns = 0;
        int interval_episodes = 0;
        const unsigned eval_seed = opt.seed ^ 0x6a09e667u;
        auto log = [&](int update) {
            const Evaluation ev = evaluate(policy, eval_seed, opt.eval_episodes, opt);
            const double train_mean = interval_episodes ? interval_returns / interval_episodes : 0.0;
            std::printf("%d %.3f %.3f %.3f %.3f\n", update, rollout_s + learner_s,
                        ev.mean_return, ev.success_rate, train_mean);
            std::fflush(stdout);
            interval_returns = 0;
            interval_episodes = 0;
        };
        log(0);
        for (int u = 1; u <= opt.updates; ++u) {
            // Freeze the policy while all environments generate this on-policy batch.
            const auto rollout_start = std::chrono::steady_clock::now();
            std::vector<Trajectory> batch;
            batch.reserve(opt.envs);
            for (auto& worker : workers)
                batch.push_back(worker.collect(policy, opt.rollout, interval_episodes, interval_returns));
            rollout_s += seconds(std::chrono::steady_clock::now() - rollout_start);

            const auto learner_start = std::chrono::steady_clock::now();
            const auto samples = prepare_batch(policy, batch);
            std::vector<size_t> order(samples.size());
            std::iota(order.begin(), order.end(), 0);
            for (int epoch = 0; epoch < opt.epochs; ++epoch) {
                std::shuffle(order.begin(), order.end(), initializer);
                for (size_t begin = 0; begin < order.size(); begin += opt.minibatch) {
                    const size_t end = std::min(order.size(), begin + static_cast<size_t>(opt.minibatch));
                    learn_minibatch(policy, optimizer, samples, order, begin, end);
                }
            }
            learner_s += seconds(std::chrono::steady_clock::now() - learner_start);
            if (u % opt.log_every == 0 || u == opt.updates) log(u);
        }
        std::printf("training_seconds=%.3f rollout_seconds=%.3f learner_seconds=%.3f\n",
                    rollout_s + learner_s, rollout_s, learner_s);
        std::fflush(stdout);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        std::cerr << "usage: ./ppo [--updates N] [--envs N] [--rollout N]"
                     " [--epochs N] [--minibatch N] [--hidden N] [--num-layers N] [--log-every N]"
                     " [--eval-episodes N] [--size N]"
                     " [--obstacles N] [--horizon N] [--seed N]\n";
        return 1;
    }
}
