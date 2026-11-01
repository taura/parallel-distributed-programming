// 学習設定・勾配クリップ・AdamW・学習ループ。
#ifndef MINGPT_TRAINER_H
#define MINGPT_TRAINER_H
#include "model.h"
#include <array>
#include <functional>

namespace mingpt {
using namespace std;

// 更新回数・学習率・AdamW の係数など、学習の設定をまとめる。
struct TrainerConfig {
    int max_iters = 300, batch_size = 8;
    double learning_rate = 3e-4;
    array<double,2> betas{0.9,0.95};
    double weight_decay = 0.1, grad_norm_clip = 1.0;
};

// 全勾配のノルムを上限以下に縮小し、縮小前のノルムを返す。
double clip_grad_norm_(Parameters& parameters, double max_norm);

// 勾配の移動平均と、勾配から分離した重み減衰でパラメータを更新する。
struct AdamW {
    Parameters& parameters;
    TrainerConfig config;
    // 各パラメータの一次・二次モーメントを保持する。
    struct State { Values m, v; };
    map<string, State> state;
    int steps = 0;
    // 更新対象と設定を保存し、モーメント配列をゼロで初期化する。
    AdamW(Parameters& parameters, const TrainerConfig& config);
    // 現在の勾配で、重み減衰と AdamW の更新を 1 回行う。
    void step();
};

// 指定されたバッチ数の入力列と教師ラベルを用意する関数の型。
using BatchProvider = function<void(int, Tokens&, Tokens&)>;

// バッチ取得・順伝播・逆伝播・更新・コールバックを繰り返す学習器。
struct Trainer {
    GPT& model;
    int sequence_length;
    BatchProvider next_batch;
    TrainerConfig config;
    unique_ptr<AdamW> optimizer; // 原版同様、run() でモデルから作る
    int iter_num = 0;
    Tensor loss;
    double iter_time = 0, iter_dt = 0, grad_norm = 0, training_seconds = 0;
    using Callback = function<void(const Trainer&)>;
    map<string, vector<Callback>> callbacks;
    // 課題用の有限回の学習設定を返す。
    static TrainerConfig get_default_config();
    // モデル・入力系列長・データ供給関数・学習設定を保存する。
    Trainer(GPT& model, int length, BatchProvider provider, const TrainerConfig& config);
    // 指定イベントのコールバック列の末尾に関数を追加する。
    void add_callback(const string& onevent, Callback callback);
    // 指定イベントのコールバック列を、指定した 1 個の関数で置き換える。
    void set_callback(const string& onevent, Callback callback);
    // 指定イベントのコールバックを登録順に呼ぶ。
    void trigger_callbacks(const string& onevent) const;
    // 最適化器を作り直し、指定回数の学習と時間計測を行う。
    void run();
};

} // 名前空間 mingpt
#endif
