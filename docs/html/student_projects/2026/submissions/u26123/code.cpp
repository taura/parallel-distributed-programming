// minGPT の実行用ラッパー: 引数処理、データ生成、評価、テスト、結果表示。
// 移植元: https://github.com/karpathy/minGPT
// 対象コミット: 37baab71b9abea1b76ab957409a1cc2fbfba8a26
// モデルは model.cpp、学習は trainer.cpp、テンソル演算は tensor.cpp。
// このファイルの実行例は demo.ipynb（SortDataset）に対応する。
// 原実装との対応・相違点・使い方は mingpt_port.md を参照。
// 原実装の MIT ライセンスは LICENSE.minGPT を参照。
// Copyright (c) 2020 Andrej Karpathy
//
// コンパイル: clang++ -O3 -Wall -Wextra -std=c++17 code.cpp model.cpp trainer.cpp tensor.cpp -o mingpt
// 実行: ./mingpt --model-type gpt-nano --max-iters 300 --batch-size 8
// 検証: ./mingpt --self-test

#include "trainer.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using namespace mingpt;

namespace {

// ラベルと数値配列を 1 行に表示する。整数列と実数列で共用する。
template<class T>
void print_values(const string& label, const vector<T>& values) {
    cout << label;
    for (auto v : values) cout << ' ' << v;
    cout << '\n';
}

// ソート課題の問題列と教師データを生成する。原版の demo.ipynb に対応する。
// Python の hash(pickle.dumps(...)) はプロセス依存なので、ここでは固定のハッシュを使う。
struct SortDataset {
    int length, num_digits;
    bool test;
    mt19937 rng;
    // 問題の長さ・語彙サイズ・データ分割・乱数シードを保存する。
    SortDataset(int length, int num_digits, bool test, uint32_t seed);
    // 問題と答えを連結し、次トークン予測用にずらした系列長を返す。
    int get_block_size() const;
    // 固定ハッシュで、入力列をテスト側に割り当てるか判定する。
    static bool is_test(const Tokens& input);
    // 重複数とデータ分割の条件に合うまで候補を作り、問題列を 1 個返す。
    Tokens sample();
    // 問題とソート済みの答えから、入力・教師ラベル・任意の生成用入力を作る。
    void batch(int B, Tokens& x, Tokens& y, Tokens* prompts = nullptr);
};

// データ生成条件と、専用の乱数生成器を初期化する。
SortDataset::SortDataset(int length, int num_digits, bool test, uint32_t seed)
    : length(length), num_digits(num_digits), test(test), rng(seed) {}

// 学習に使う「問題＋答え」の系列長から、次トークン予測用の 1 要素を引く。
int SortDataset::get_block_size() const { return 2*length - 1; }

// 入力列の固定ハッシュにより、学習用とテスト用を分ける。
bool SortDataset::is_test(const Tokens& input) {
    uint64_t hash = 14695981039346656037ULL;
    for (int token : input) {
        hash ^= uint64_t(token);
        hash *= 1099511628211ULL;
    }
    hash ^= hash >> 33;
    hash *= 0xff51afd7ed558ccdULL;
    hash ^= hash >> 33;
    return hash % 4 == 0;
}

// 重複が多い例を増やしながら、指定した分割に属する問題列を生成する。
Tokens SortDataset::sample() {
    uniform_int_distribution<int> digit(0, num_digits - 1);
    bernoulli_distribution boost_repeats(0.5);
    for (int attempt = 0; attempt < 100000; ++attempt) {
        Tokens input(length);
        for (int& id : input) id = digit(rng);
        if (boost_repeats(rng)) {
            Tokens distinct = input;
            sort(distinct.begin(), distinct.end());
            auto count = unique(distinct.begin(), distinct.end()) - distinct.begin();
            if (count > length/2) continue;
        }
        if (is_test(input) == test) return input;
    }
    throw runtime_error("指定したデータ分割から入力を生成できません。sort-length または num-digits を増やしてください");
}

// 問題と正解を連結・シフトし、答えの位置だけを学習するバッチを作る。
void SortDataset::batch(int B, Tokens& x, Tokens& y, Tokens* prompts) {
    x.clear();
    y.clear();
    if (prompts) prompts->clear();
    int T = get_block_size();
    for (int b = 0; b < B; ++b) {
        Tokens input = sample();
        Tokens solution = input;
        sort(solution.begin(), solution.end());
        if (prompts) prompts->insert(prompts->end(), input.begin(), input.end());
        input.insert(input.end(), solution.begin(), solution.end());
        x.insert(x.end(), input.begin(), input.end() - 1);
        for (int t = 0; t < T; ++t) y.push_back(t < length - 1 ? -1 : input[t + 1]);
    }
}

// テストデータの平均損失と、ソート結果が完全一致した例数をまとめる。
struct Evaluation { double loss; int correct; };

// 固定テスト入力で、教師強制の損失と自己回帰生成の完全一致数を求める。
Evaluation evaluate(GPT& model, int length, int digits, int examples, uint32_t seed) {
    NoGrad no_grad;
    model.eval();
    SortDataset test(length, digits, true, seed);
    int T = test.get_block_size();
    double total_loss = 0.0;
    int correct = 0;
    for (int offset = 0; offset < examples; offset += 8) {
        const int B = min(8, examples - offset);
        Tokens x, y, prompt;
        test.batch(B, x, y, &prompt);
        total_loss += B * model(x, B, T, y).loss.item();
        auto prediction = model.generate(prompt, B, length, length);
        for (int b = 0; b < B; ++b) {
            auto expected = y.begin() + b*T + length - 1;
            auto actual = prediction.begin() + b*2*length + length;
            correct += equal(expected, expected + length, actual);
        }
    }
    return {total_loss / examples, correct};
}

// 数値比較用に、2 層・2 ヘッド・4 次元の小モデルの設定を返す。
GPTConfig fixture_config() {
    GPTConfig config;
    config.n_layer = 2;
    config.n_head = 2;
    config.n_embd = 4;
    config.vocab_size = 5;
    config.block_size = 3;
    return config;
}

// 全パラメータを決まった数列で上書きし、初期化乱数に依存しない重みを用意する。
void fixture_weights(GPT& model) {
    size_t index = 0;
    for (auto& [name, p] : model.parameters.named)
        for (double& v : p.value.data()) {
            v = 0.07 * sin(double(++index));
            if (name.find(".ln_") != string::npos && name.find(".weight") != string::npos) v += 1.0;
        }
}
// 重複 ID と系列ごとに異なる無視マスクを含む、2 系列の検証データ。
const Tokens fixture_x{0,1,0, 3,2,1};
const Tokens fixture_y{-1,0,2, 2,-1,-1};

// 計算済みの勾配を中心差分と照合し、全要素の最大絶対誤差を返す。
double check_gradients(const vector<Tensor>& parameters, const function<double()>& measure,
                       double tolerance = 2e-7) {
    NoGrad no_grad;
    constexpr double eps = 1e-6;
    double worst = 0;
    for (auto parameter : parameters) {
        Values gradient = parameter.grad();
        auto& values = parameter.data();
        for (size_t i = 0; i < values.size(); ++i) {
            double saved = values[i];
            values[i] = saved + eps;
            double plus = measure();
            values[i] = saved - eps;
            double minus = measure();
            values[i] = saved;

            double error = abs(gradient[i] - (plus - minus)/(2*eps));
            assert(error < tolerance);
            worst = max(worst, error);
        }
    }
    return worst;
}

// 固定条件で 2 回更新し、比較用の損失・ロジット・勾配・更新後の重みを出力する。
void reference_dump() {
    // LOSS/LOGITS/GRAD/WEIGHT は比較プログラムが読むデータ形式の識別子。
    GPT model(fixture_config(), 123);
    fixture_weights(model);
    model.eval();
    auto config = Trainer::get_default_config();
    config.grad_norm_clip = 0.05;
    auto optimizer = model.configure_optimizers(config);
    cout << setprecision(17);
    for (int step = 0; step < 2; ++step) {
        auto result = model(fixture_x, 2, 3, fixture_y);
        model.zero_grad();
        result.loss.backward();
        cout << "LOSS " << result.loss.item() << '\n';
        print_values("LOGITS", result.logits.data());
        for (const auto& [name, p] : model.parameters.named) print_values("GRAD " + name, p.value.grad());
        clip_grad_norm_(model.parameters, config.grad_norm_clip);
        optimizer->step();
        for (const auto& [name, p] : model.parameters.named) print_values("WEIGHT " + name, p.value.data());
    }
}

// 四則演算・スカラーの左右・比較マスク・非線形演算の値と勾配を検証する。
void operator_test() {
    Tensor a({2,3},0,true);
    Tensor b({1,3},0,true);
    a.data() = {-0.6,-0.2,0.0,0.4,0.8,1.2};
    b.data() = {0.7,1.1,1.6};
    // 同じ入力を複数の経路で使い、各演算の勾配が正しく合流するか確認する。
    auto expression = [&] {
        auto ratio = (2+a)*(a-b)/(b+1);
        auto z = ratio + (3-a) + 2/(a+4) - a/2 + (a-0.5) - (-a);
        return z.tanh() + a.pow(3)*0.1;
    };
    auto z = expression();
    for (size_t i = 0; i < a.numel(); ++i) {
        double x = a.data()[i];
        double y = b.data()[i%3];
        double ratio = (2+x)*(x-y)/(y+1);
        double expected = tanh(ratio + (3-x) + 2/(x+4) - x/2 + (x-0.5) - (-x)) + pow(x,3)*0.1;
        assert(abs(z.data()[i]-expected) < 1e-12);
    }
    Tokens targets{0,2};
    auto loss = F::cross_entropy(z,targets);
    loss.backward();
    // 共通の数値微分処理に、損失だけを返す関数を渡す。
    check_gradients({a,b}, [&] { return F::cross_entropy(expression(),targets).item(); });

    Tensor threshold({2,1});
    threshold.data() = {0.0,0.8};
    auto mask = a < threshold;
    assert(!mask.requires_grad() && mask.data() == Values({1,1,0,1,0,0}));
    auto masked = a.masked_fill(mask,-10);
    assert(masked.data() == Values({-10,-10,0,-10,0.8,1.2}));
    assert((a < 0.0).data() == Values({1,1,0,0,0,0}));
    assert((a == 0.0).data() == Values({0,0,1,0,0,0}));
    // スカラー側のゼロ次元 Tensor と、指数ゼロの局所微分も確認する。
    Tensor zero({},0,true);
    auto one = zero.pow(0);
    one.backward();
    assert(one.item() == 1 && zero.grad()[0] == 0);
    Tensor scalar({},2,true);
    auto scalar_loss = F::cross_entropy(a/scalar,targets);
    scalar_loss.backward();
    // ゼロ次元の分母に戻る勾配も、同じ数値微分処理で照合する。
    check_gradients({scalar}, [&] { return F::cross_entropy(a/scalar,targets).item(); });
    assert(a.pow(0).data() == Values(a.numel(),1));
    cout << "Tensor の演算子・比較マスク・累乗・tanh: 検証成功\n";
}

// 形状操作・ブロードキャスト・微分・勾配加算・NoGrad の入れ子を検証する。
void tensor_test() {
    Tensor a({2,1,2,3},0,true);
    Tensor b({1,3,3,2},0,true);
    for (size_t i = 0; i < a.numel(); ++i) a.data()[i] = 0.1*sin(double(i+1));
    for (size_t i = 0; i < b.numel(); ++i) b.data()[i] = 0.2*cos(double(i+1));
    auto view = a.view({-1});
    assert(&view.data() == &a.data());
    auto logits = (a+a).matmul(b);
    assert(logits.shape() == Shape({2,3,2,2}));
    Tokens targets(12,1);
    auto loss = F::cross_entropy(logits,targets);
    loss.backward();
    // 数値微分用の損失を計算し、入れ子の NoGrad が記録を停止し続けるか確認する。
    auto measure = [&] {
        NoGrad outer;
        { NoGrad inner; }
        auto result = F::cross_entropy((a+a).matmul(b),targets);
        assert(!result.requires_grad());
        return result.item();
    };
    check_gradients({a,b}, measure);
    Values once = a.grad();
    loss.backward();
    for (size_t i = 0; i < a.numel(); ++i)
        assert(abs(a.grad()[i]-2*once[i]) < 1e-12);
    cout << "テンソルの形状操作・ブロードキャスト・勾配加算・NoGrad: 検証成功\n";
}

// マスク、Dropout のモード、コールバック、学習器の再初期化を検証する。
void model_trainer_test() {
    auto model_config = fixture_config();
    model_config.n_layer = 1;
    GPT model(model_config,123);
    const auto& mask = model.transformer.h[0].attn.bias;
    assert(mask.shape() == Shape({1,1,3,3}) && !mask.requires_grad());
    for (int t = 0; t < 3; ++t)
        for (int s = 0; s < 3; ++s) assert(mask.data()[t*3+s] == double(s <= t));
    {
        NoGrad no_grad;
        Tensor x({1,3,4},1);
        model.eval();
        assert(model.transformer.drop(x).data() == x.data());
        model.train();
        assert(model.transformer.drop(x).data() != x.data());
    }
    auto train_config = Trainer::get_default_config();
    train_config.max_iters = 2;
    train_config.batch_size = 2;
    // データ供給関数として、毎回同じ検証用の入力と教師を返す。
    auto next_batch = [](int, Tokens& x, Tokens& y) {
        x = fixture_x;
        y = fixture_y;
    };
    Trainer trainer(model,3,next_batch,train_config);
    assert(!trainer.optimizer);
    Tokens events;
    int replaced = 0;
    // 後で置き換える処理を登録し、置換後には呼ばれないことを確認する。
    trainer.add_callback("on_batch_end",[&](const Trainer&) { ++replaced; });
    // 更新済みの状態を確認し、最初のコールバックの実行順を記録する。
    trainer.set_callback("on_batch_end",[&](const Trainer& t) {
        assert(t.optimizer->steps == t.iter_num+1 && isfinite(t.loss.item()));
        events.push_back(100+t.iter_num);
    });
    // 2 番目のコールバックの実行順を記録する。
    trainer.add_callback("on_batch_end",[&](const Trainer& t) { events.push_back(200+t.iter_num); });
    trainer.trigger_callbacks("未登録のイベント");
    for (int run = 0; run < 2; ++run) {
        events.clear();
        trainer.run();
        assert(events == Tokens({100,200,101,201}) && replaced == 0);
        assert(trainer.iter_num == 2 && trainer.optimizer->steps == 2);
    }
    cout << "Module の動作・マスク・最適化器生成・コールバック: 検証成功\n";
}

// 学習・評価モードの両方で、モデル全体の勾配を数値微分と照合する。
void model_gradient_test() {
    GPT model(fixture_config(), 123);
    fixture_weights(model);
    vector<Tensor> parameters;
    for (const auto& [name, p] : model.parameters.named) parameters.push_back(p.value);
    for (bool training : {false, true}) {
        model.training = training;
        // 同じ乱数マスクを再現し、現在の重みに対する損失だけを計算する。
        auto compute_loss = [&] {
            model.rng.seed(91);
            return model(fixture_x, 2, 3, fixture_y).loss;
        };
        auto loss = compute_loss();
        model.zero_grad();
        loss.backward();
        // 数値微分側では、同じ損失のスカラー値を返す。
        double worst = check_gradients(parameters, [&] { return compute_loss().item(); }, 2e-6);
        cout << (training ? "学習モード（ドロップアウト有効）" : "評価モード")
             << "の勾配検証: パラメータ数=" << model.parameters.count()
             << " 最大絶対誤差=" << worst << '\n';
    }
}

// 未来の入力と別系列の独立性、文脈を切り詰めた生成結果を検証する。
void generation_test() {
    GPT model(fixture_config(), 123);
    fixture_weights(model);
    model.eval();
    NoGrad no_grad;
    Tokens changed = fixture_x;
    changed[2] = 4;
    auto original = model(fixture_x, 2, 3).logits;
    auto perturbed = model(changed, 2, 3).logits;
    auto isolated = model(Tokens{3,2,1}, 1, 3).logits;
    auto earlier = original.narrow(1,0,2);
    auto other_sequence = original.narrow(0,1,1);
    assert(earlier.data() == perturbed.narrow(1,0,2).data());
    assert(other_sequence.data() == perturbed.narrow(0,1,1).data());
    assert(other_sequence.data() == isolated.data());

    auto greedy = model.generate(fixture_x, 2, 3, 5);
    auto top1 = model.generate(fixture_x, 2, 3, 5, 0.7, true, 1);
    assert(greedy == top1);
    cout << "因果性・系列間の独立性・生成と文脈の切り詰め: 検証成功\n";
}

// 各テストを順に実行し、テンソル演算からモデル・学習・生成まで検証する。
void self_test() {
    operator_test();
    tensor_test();
    model_trainer_test();
    model_gradient_test();
    generation_test();
}

// コマンドラインから指定するモデル・学習・評価・生成の設定をまとめる。
struct Options {
    GPTConfig model = GPT::get_default_config();
    TrainerConfig trainer = Trainer::get_default_config();
    int sort_length = 6;
    int num_digits = 3;
    int eval_examples = 64;
    int log_every = 25;
    uint32_t seed = 3407;
    double temperature = 1.0;
    int top_k = 0;
    bool do_sample = false;
    bool self_test = false;
    bool reference = false;
};

// 利用可能な引数と既定値を表示する。
void print_help() {
    cout << R"(逐次 C++ 版 minGPT（括弧内は既定値）
--model-type NAME     モデル種別（gpt-nano）
--n-layer N --n-head N --n-embd N   model-type の代わりに 3 項目を指定
--max-iters N         更新回数（300）
--batch-size N        バッチ数（8）
--sort-length N       問題列の長さ（6）
--num-digits N        語彙サイズ（3）
--learning-rate X     学習率（0.0003）
--beta1 X --beta2 X   AdamW の係数（0.9, 0.95）
--weight-decay X      重み減衰（0.1）
--grad-norm-clip X    勾配ノルム上限（1.0）
--embd-pdrop X --resid-pdrop X --attn-pdrop X   ドロップアウト率（各 0.1）
--seed N             乱数シード（3407）
--eval-examples N    評価例数（64）
--log-every N        表示間隔（25 更新）
--do-sample          最後の生成例をサンプリング（評価は常に貪欲法）
--temperature X     生成温度（1.0）
--top-k N           生成候補数（0 は無制限）
--self-test         テンソル・勾配・因果性・生成の検証
--reference-dump    固定条件で損失・ロジット・勾配・重みを出力
)";
}

// 引数を設定表に従って読み、ソート課題に応じた語彙サイズと文脈長を設定する。
Options parse(int argc, char** argv) {
    Options options;
    auto& model = options.model;
    auto& trainer = options.trainer;
    string model_type = "gpt-nano";
    bool custom_dimensions = false;

    map<string, int*> integers{
        {"--n-layer", &model.n_layer},
        {"--n-head", &model.n_head},
        {"--n-embd", &model.n_embd},
        {"--max-iters", &trainer.max_iters},
        {"--batch-size", &trainer.batch_size},
        {"--sort-length", &options.sort_length},
        {"--num-digits", &options.num_digits},
        {"--eval-examples", &options.eval_examples},
        {"--log-every", &options.log_every},
        {"--top-k", &options.top_k}
    };
    map<string, double*> reals{
        {"--learning-rate", &trainer.learning_rate},
        {"--beta1", &trainer.betas[0]},
        {"--beta2", &trainer.betas[1]},
        {"--weight-decay", &trainer.weight_decay},
        {"--grad-norm-clip", &trainer.grad_norm_clip},
        {"--embd-pdrop", &model.embd_pdrop},
        {"--resid-pdrop", &model.resid_pdrop},
        {"--attn-pdrop", &model.attn_pdrop},
        {"--temperature", &options.temperature}
    };
    map<string, bool*> flags{
        {"--self-test", &options.self_test},
        {"--reference-dump", &options.reference},
        {"--do-sample", &options.do_sample}
    };

    for (int i = 1; i < argc; ++i) {
        string key = argv[i];
        if (key == "--help") {
            print_help();
            exit(0);
        }
        if (flags.count(key)) {
            *flags.at(key) = true;
            continue;
        }
        if (i + 1 == argc) throw invalid_argument("値が必要です: " + key);
        string value = argv[++i];
        if (key == "--model-type") {
            model_type = value;
        } else if (key == "--seed") {
            options.seed = uint32_t(stoul(value));
        } else if (integers.count(key)) {
            *integers.at(key) = stoi(value);
            if (key.rfind("--n-",0) == 0) custom_dimensions = true;
        } else {
            *reals.at(key) = stod(value);
        }
    }
    if (!custom_dimensions) model.preset(model_type);
    model.block_size = 2*options.sort_length - 1;
    model.vocab_size = options.num_digits;
    return options;
}

// モデルと学習の設定を、実行前にまとめて表示する。
void print_config(const GPT& model, const Options& options) {
    const auto& config = model.config;
    const auto& train_config = options.trainer;
    cout << "minGPT の逐次 C++ 移植版（移植元: 37baab71b9ab）\n"
         << "層数=" << config.n_layer
         << " ヘッド数=" << config.n_head
         << " 埋め込み次元数=" << config.n_embd
         << " 語彙サイズ=" << config.vocab_size
         << " 文脈長=" << config.block_size << '\n'
         << "パラメータ数=" << model.parameters.count()
         << "（lm_head を含む） 数値型=double 乱数シード=" << options.seed << '\n'
         << "更新回数=" << train_config.max_iters
         << " バッチ数=" << train_config.batch_size
         << " 学習率=" << train_config.learning_rate
         << " モーメント係数=" << train_config.betas[0] << ',' << train_config.betas[1]
         << " 重み減衰係数=" << train_config.weight_decay << '\n';
}

// 指定間隔または最終更新で、現在の損失・勾配ノルム・学習時間を表示する。
void print_progress(const Trainer& trainer, int log_every) {
    int step = trainer.iter_num + 1;
    if (step % log_every != 0 && step != trainer.config.max_iters) return;
    cout << "更新=" << step
         << " 損失=" << trainer.loss.item()
         << " 勾配ノルム=" << trainer.grad_norm
         << " 学習時間（秒）=" << trainer.training_seconds << '\n';
}

// 学習前後の損失・正解率と、学習ループの処理速度を表示する。
void print_summary(const Evaluation& initial, const Evaluation& final,
                   const Trainer& trainer, const Options& options) {
    double seconds = trainer.training_seconds;
    double tokens = double(options.trainer.max_iters) * options.trainer.batch_size * options.model.block_size;
    cout << "学習後のテスト損失=" << final.loss
         << " 損失の減少量=" << initial.loss - final.loss
         << " ソート完全一致数=" << final.correct << '/' << options.eval_examples
         << " 正解率=" << 100.0 * final.correct / options.eval_examples << "%\n"
         << "学習時間（秒）=" << seconds << '\n'
         << "処理した入力トークン数=" << tokens
         << " 1 秒あたりのトークン数=" << (seconds > 0 ? tokens / seconds : 0) << '\n';
}

// テスト用の問題を一つ生成し、GPT の答えと正解を並べて表示する。
void show_prediction(GPT& model, const Options& options) {
    model.eval();
    SortDataset data(options.sort_length, options.num_digits, true, options.seed + 3u);
    Tokens input = data.sample();
    Tokens truth = input;
    sort(truth.begin(), truth.end());
    auto output = model.generate(input, 1, options.sort_length, options.sort_length,
                                 options.temperature, options.do_sample, options.top_k);
    Tokens prediction(output.begin() + input.size(), output.end());
    print_values("入力列:", input);
    print_values("GPT のソート結果:", prediction);
    print_values("正しいソート結果:", truth);
}

// モデルと学習器を接続し、学習前の評価・学習・学習後の評価・生成例を順に実行する。
void run_training(const Options& options) {
    GPT model(options.model, options.seed);
    SortDataset data(options.sort_length, options.num_digits, false, options.seed + 1u);
    print_config(model, options);

    auto initial = evaluate(model, options.sort_length, options.num_digits, options.eval_examples, options.seed + 2u);
    cout << fixed << setprecision(6) << "学習前のテスト損失=" << initial.loss
         << " ソート完全一致数=" << initial.correct << '/' << options.eval_examples << '\n';

    // 学習器から要求された入力と教師を、データ生成器で用意する。
    auto next_batch = [&](int B, Tokens& x, Tokens& y) { data.batch(B, x, y); };
    Trainer trainer(model, data.get_block_size(), next_batch, options.trainer);
    // 各更新の終了時の通知を、表示専用の関数へ渡す。
    trainer.set_callback("on_batch_end", [&](const Trainer& t) { print_progress(t, options.log_every); });
    trainer.run();

    auto final = evaluate(model, options.sort_length, options.num_digits, options.eval_examples, options.seed + 2u);
    print_summary(initial, final, trainer, options);
    show_prediction(model, options);
}

} // 実行用ラッパーの無名名前空間の終わり

// 引数を解析し、自己テスト・参照出力・通常学習のいずれかへ振り分ける。
int main(int argc, char** argv) {
    try {
        auto options = parse(argc, argv);
        if (options.self_test) self_test();
        else if (options.reference) reference_dump();
        else run_training(options);
    } catch (const exception& e) {
        cerr << "エラー: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
