// minGPT の学習ループと AdamW。モデル構造・データ生成・表示から独立。
// 原実装: karpathy/minGPT 37baab71b9abea1b76ab957409a1cc2fbfba8a26
// Copyright (c) 2020 Andrej Karpathy（LICENSE.minGPT を参照）
#include "trainer.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>

namespace mingpt {
using namespace std;

// 全勾配の L2 ノルムを計算して上限以下に縮小し、縮小前の値を返す。
double clip_grad_norm_(Parameters& parameters, double max_norm) {
    double norm = 0;
    for (auto& [name, p] : parameters.named)
        for (double g : p.value.grad()) norm += g*g;
    norm = sqrt(norm);
    double scale = min(1.0, max_norm/(norm+1e-6));
    for (auto& [name, p] : parameters.named)
        for (double& g : p.value.grad()) g *= scale;
    return norm;
}

// 更新対象と設定を保存し、各パラメータのモーメント配列を確保する。
AdamW::AdamW(Parameters& p, const TrainerConfig& c) : parameters(p), config(c) {
    for (const auto& [name, param] : p.named)
        state[name] = {Values(param.value.numel(),0), Values(param.value.numel(),0)};
}

// 重み減衰、モーメント更新、偏り補正を行い、パラメータを 1 回更新する。
void AdamW::step() {
    ++steps;
    double lr = config.learning_rate;
    auto [b1,b2] = config.betas;
    double correction1 = 1-pow(b1,steps), correction2 = 1-pow(b2,steps);
    for (auto& [name, p] : parameters.named) {
        auto& [m,v] = state.at(name);
        auto& w = p.value.data();
        const auto& g = p.value.grad();
        for (size_t i = 0; i < w.size(); ++i) {
            w[i] *= 1-lr*(p.decay ? config.weight_decay : 0);
            m[i] = b1*m[i]+(1-b1)*g[i];
            v[i] = b2*v[i]+(1-b2)*g[i]*g[i];
            w[i] -= lr*(m[i]/correction1)/(sqrt(v[i]/correction2)+1e-8);
        }
    }
}

// 課題用の有限回の学習設定を返す。
TrainerConfig Trainer::get_default_config() { return {}; }

// モデル・系列長・データ供給関数・設定を保存する。最適化器は run で作る。
Trainer::Trainer(GPT& m, int length, BatchProvider provider, const TrainerConfig& c)
    : model(m), sequence_length(length), next_batch(std::move(provider)), config(c) {}

// イベントに対応するコールバック列の末尾に処理を追加する。
void Trainer::add_callback(const string& onevent, Callback callback) {
    callbacks[onevent].push_back(std::move(callback));
}

// イベントの既存コールバックを、指定した 1 個の処理で置き換える。
void Trainer::set_callback(const string& onevent, Callback callback) {
    callbacks[onevent] = {std::move(callback)};
}

// イベントに登録されている処理を、登録順に実行する。
void Trainer::trigger_callbacks(const string& onevent) const {
    auto it = callbacks.find(onevent);
    if (it != callbacks.end()) for (const auto& callback : it->second) callback(*this);
}

// 最適化器と反復状態を初期化し、バッチ学習・時間計測・通知を繰り返す。
void Trainer::run() {
    using Clock = chrono::steady_clock;
    optimizer = model.configure_optimizers(config);
    model.train();
    iter_num = 0;
    iter_dt = training_seconds = 0;
    iter_time = chrono::duration<double>(Clock::now().time_since_epoch()).count();
    while (iter_num < config.max_iters) {
        auto start = Clock::now();
        {
            Tokens x, y;
            next_batch(config.batch_size, x, y);
            auto result = model(x, config.batch_size, sequence_length, y);
            loss = result.loss;
            model.zero_grad();
            loss.backward();
            grad_norm = clip_grad_norm_(model.parameters, config.grad_norm_clip);
            optimizer->step();
        }
        // 授業の計測用に、表示時間を含まない合計も別途記録する。
        training_seconds += chrono::duration<double>(Clock::now()-start).count();
        trigger_callbacks("on_batch_end");
        ++iter_num;
        double tnow = chrono::duration<double>(Clock::now().time_since_epoch()).count();
        iter_dt = tnow-iter_time;
        iter_time = tnow;
    }
}
} // 名前空間 mingpt
