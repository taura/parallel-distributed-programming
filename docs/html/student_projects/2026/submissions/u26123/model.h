// minGPT のモデル。学習ループと最適化器は trainer.h を参照。
// 原実装: Andrej Karpathy / minGPT（MIT ライセンス、LICENSE.minGPT を参照）。
#ifndef MINGPT_MODEL_H
#define MINGPT_MODEL_H

#include "tensor.h"
#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace mingpt {
using namespace std;

// モデルの層数・次元数・語彙サイズ・ドロップアウト率をまとめる。
struct GPTConfig {
    int n_layer = 3, n_head = 3, n_embd = 48; // 原デモの gpt-nano 設定
    int vocab_size = 3, block_size = 11;
    double embd_pdrop = 0.1, resid_pdrop = 0.1, attn_pdrop = 0.1;
    // モデル名から層数・ヘッド数・埋め込み次元数を設定する。
    void preset(const string& name);
};

// 学習対象のテンソルと、重み減衰を適用するかを保持する。
struct Parameter {
    Tensor value;
    bool decay; // nn.Linear.weight のみ重み減衰を適用する
};

// モデル全体のパラメータを名前付きで登録・管理する。
struct Parameters {
    // 名前と重み配置は原実装の model.named_parameters() に合わせる。
    map<string, Parameter> named;
    // 新しいパラメータを登録し、値と勾配を共有するハンドルを返す。
    Tensor add(const string& name, Shape shape, bool decay = false);
    // 登録済みの全パラメータの勾配をゼロにする。
    void zero_grad();
    // 学習対象のスカラー値の総数を返す。
    size_t count() const;
};

// nn.Module の呼び出し部分に相当する。各クラスの計算は forward に置く。
struct Module {
    // 基底クラス経由でも派生クラスを正しく破棄する。
    virtual ~Module();
    // 派生クラスが実装する、層の順伝播の共通インターフェース。
    virtual Tensor forward(Tensor x) const = 0;
    // layer(x) という呼び方を、その層の forward に転送する。
    Tensor operator()(Tensor x) const;
};

// 入力の最後の軸に重み行列とバイアスを適用する全結合層。
struct Linear : Module {
    Tensor weight, bias;
    // 重みと、必要ならバイアスをパラメータとして登録する。
    Linear(Parameters& p, const string& name, int in, int out, bool use_bias = true);
    // 保持しているパラメータで全結合変換を計算する。
    Tensor forward(Tensor x) const override;
};

// 各トークンの特徴ベクトルを正規化する層。
struct LayerNorm : Module {
    Tensor weight, bias;
    // 正規化後に適用する重みとバイアスを登録する。
    LayerNorm(Parameters& p, const string& name, int width);
    // 最後の軸に LayerNorm を適用する。
    Tensor forward(Tensor x) const override;
};

// トークン ID や位置番号を学習可能な特徴ベクトルへ変換する層。
struct Embedding {
    Tensor weight;
    // ID の種類数と特徴幅に応じた埋め込み行列を登録する。
    Embedding(Parameters& p, const string& name, int count, int width);
    // ID 列から埋め込みを取り出し、指定された入力形状に並べる。
    Tensor forward(const Tokens& ids, Shape shape) const;
    // 埋め込み層の関数形式の呼び出しを forward に転送する。
    Tensor operator()(const Tokens& ids, Shape shape) const;
};

// モデルの学習モードと乱数生成器を共有するドロップアウト層。
struct Dropout : Module {
    double p;
    const bool& training;
    mt19937& rng;
    // 確率と、モデル内のモード・乱数生成器への参照を保存する。
    Dropout(double p, const bool& training, mt19937& rng);
    // 学習モードのときだけ入力にドロップアウトを適用する。
    Tensor forward(Tensor x) const override;
};

// 原 minGPT の tanh 近似 GELU を使う活性化層。
struct NewGELU : Module {
    // 入力の各要素に GELU を適用する。
    Tensor forward(Tensor x) const override;
};

// 未来の位置を参照しない、マルチヘッド自己注意層。
struct CausalSelfAttention : Module {
    Linear c_attn, c_proj;
    Dropout attn_dropout, resid_dropout;
    Tensor bias; // 原版の register_buffer("bias", ...) に対応する非学習マスク
    int n_head, n_embd;
    // Q/K/V・出力射影・ドロップアウト層と下三角マスクを構築する。
    CausalSelfAttention(Parameters& p, const string& name, const GPTConfig& c, const bool& training, mt19937& rng);
    // Q/K/V に分け、因果マスク付き注意と出力射影を計算する。
    Tensor forward(Tensor x) const override;
};

// トークンごとの特徴変換に使う全結合層・活性化層・ドロップアウトを保持する。
// 原版の Block.mlp（ModuleDict）と同じメンバ。
struct MLP {
    Linear c_fc, c_proj;
    NewGELU act;
    Dropout dropout;
    // 特徴幅を 4 倍に広げて戻す全結合層と、活性化・ドロップアウトを構築する。
    MLP(Parameters& p, const string& name, const GPTConfig& c, const bool& training, mt19937& rng);
};

// 自己注意と MLP を、事前正規化と残差接続で組み合わせるブロック。
struct Block : Module {
    LayerNorm ln_1;
    CausalSelfAttention attn;
    LayerNorm ln_2;
    MLP mlp;
    // 2 個の正規化層、自己注意層、MLP を構築する。
    Block(Parameters& p, const string& name, const GPTConfig& c, const bool& training, mt19937& rng);
    // MLP の全結合・GELU・全結合・ドロップアウトを順に実行する。
    Tensor mlpf(Tensor x) const;
    // 自己注意と MLP の結果を、それぞれ残差として加算する。
    Tensor forward(Tensor x) const override;
};

// 埋め込み・Transformer ブロック列・最終正規化層を保持するコンテナ。
// 原版の GPT.transformer（ModuleDict）に対応する。
struct Transformer {
    Embedding wte, wpe;
    Dropout drop;
    vector<Block> h;
    LayerNorm ln_f;
    // 埋め込み、ブロック列、最後の正規化層を構築する。
    Transformer(Parameters& p, const GPTConfig& c, const bool& training, mt19937& rng);
};

// 順伝播のロジットと、教師がある場合の損失をまとめて返す。
struct ForwardResult { Tensor logits, loss; };
// 学習率などの最適化設定。定義は trainer.h に置く。
struct TrainerConfig;
// AdamW 最適化器。定義は trainer.h に置く。
struct AdamW;

// 埋め込み・Transformer・出力層を持ち、次トークン予測と生成を行うモデル。
struct GPT {
    GPTConfig config;
    Parameters parameters;
    bool training = true;
    mt19937 rng;
    Transformer transformer;
    Linear lm_head;
    int block_size;
    // 課題用の小規模モデルの既定設定を返す。
    static GPTConfig get_default_config();
    // 全ての層を構築し、重みと乱数生成器を初期化する。
    GPT(const GPTConfig& c, uint32_t seed);
    // Dropout の状態参照を別モデルへ持ち越さないため、コピーを禁止する。
    GPT(const GPT&) = delete;
    // Dropout の状態参照を保つため、コピー代入を禁止する。
    GPT& operator=(const GPT&) = delete;
    // Dropout が参照するモデル本体のアドレスを固定し、ムーブを禁止する。
    GPT(GPT&&) = delete;
    // モデル本体のアドレスを固定するため、ムーブ代入を禁止する。
    GPT& operator=(GPT&&) = delete;
    // Linear・Embedding・LayerNorm のパラメータを初期化する。
    void _init_weights();
    // 登録済みパラメータと学習設定から AdamW を生成する。
    unique_ptr<AdamW> configure_optimizers(const TrainerConfig& train_config);
    // モデル全体のドロップアウトを学習モードにする。
    void train();
    // モデル全体のドロップアウトを評価モードにする。
    void eval();
    // モデル全体のパラメータ勾配をゼロにする。
    void zero_grad();
    // 整数列から全位置のロジットと、必要なら次トークン予測の損失を計算する。
    ForwardResult forward(const Tokens& idx, int B, int T, const Tokens& targets = {});
    // model(...) という呼び方を forward に転送する。
    ForwardResult operator()(const Tokens& idx, int B, int T, const Tokens& targets = {});
    // 入力列に続くトークンを、貪欲法またはサンプリングで逐次生成する。
    Tokens generate(Tokens idx, int B, int T, int max_new_tokens, double temperature = 1.0,
                    bool do_sample = false, int top_k = 0);
};

} // 名前空間 mingpt の終わり
#endif
