// minGPT のモデルと生成処理。
// 移植元: https://github.com/karpathy/minGPT
// 対象コミット: 37baab71b9abea1b76ab957409a1cc2fbfba8a26
// mingpt/model.py に対応する。
// Copyright (c) 2020 Andrej Karpathy
// 原実装の MIT ライセンスは LICENSE.minGPT を参照。

#include "model.h"
#include "trainer.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace mingpt {
using namespace std;

// 原 minGPT のモデル名から、層数・ヘッド数・埋め込み次元数を設定する。
void GPTConfig::preset(const string& name) {
    const map<string, array<int, 3>> types{
        {"openai-gpt", {12,12,768}}, {"gpt2", {12,12,768}},
        {"gpt2-medium", {24,16,1024}}, {"gpt2-large", {36,20,1280}},
        {"gpt2-xl", {48,25,1600}}, {"gopher-44m", {8,16,512}},
        {"gpt-mini", {6,6,192}}, {"gpt-micro", {4,4,128}}, {"gpt-nano", {3,3,48}}
    };
    const auto it = types.find(name);
    if (it == types.end()) throw runtime_error("不明なモデル種別です: " + name);
    n_layer = it->second[0]; n_head = it->second[1]; n_embd = it->second[2];
}

// 微分可能なゼロのテンソルを名前付きで登録し、共有ハンドルを返す。
Tensor Parameters::add(const string& name, Shape shape, bool decay) {
    Tensor value(std::move(shape), 0, true);
    named.emplace(name, Parameter{value, decay});
    return value;
}

// 登録されている全パラメータの勾配をゼロにする。
void Parameters::zero_grad() {
    for (auto& [name, p] : named) p.value.zero_grad();
}

// 登録されているパラメータの全要素数を合計する。
size_t Parameters::count() const {
    size_t n = 0;
    for (const auto& [name, p] : named) n += p.value.numel();
    return n;
}

// 基底クラス経由でも派生クラスを正しく破棄できるようにする。
Module::~Module() = default;
// 関数形式の呼び出しを、実際の層の forward に転送する。
Tensor Module::operator()(Tensor x) const { return forward(std::move(x)); }

// 全結合層の重みと、必要ならバイアスを登録する。
Linear::Linear(Parameters& p, const string& name, int in, int out, bool use_bias)
    : weight(p.add(name + ".weight", {out,in}, true)),
      bias(use_bias ? p.add(name + ".bias", {out}) : Tensor()) {}

// この層の重み・バイアスで全結合変換を行う。
Tensor Linear::forward(Tensor x) const { return F::linear(x, weight, bias); }

// LayerNorm の学習可能な重みとバイアスを登録する。
LayerNorm::LayerNorm(Parameters& p, const string& name, int width)
    : weight(p.add(name + ".weight", {width})), bias(p.add(name + ".bias", {width})) {}

// この層の重み・バイアスを使って最後の軸を正規化する。
Tensor LayerNorm::forward(Tensor x) const { return F::layer_norm(x, weight, bias); }

// ID の種類数と特徴幅に応じた埋め込み行列を登録する。
Embedding::Embedding(Parameters& p, const string& name, int count, int width)
    : weight(p.add(name + ".weight", {count,width})) {}
// ID 列から特徴ベクトルを取り出し、指定された入力形状に並べる。
Tensor Embedding::forward(const Tokens& ids, Shape shape) const { return F::embedding(ids,weight,std::move(shape)); }
// 埋め込み層の関数形式の呼び出しを forward に転送する。
Tensor Embedding::operator()(const Tokens& ids, Shape shape) const { return forward(ids,std::move(shape)); }

// 確率と、モデルで共有する学習モード・乱数生成器への参照を保存する。
Dropout::Dropout(double p, const bool& training, mt19937& rng) : p(p), training(training), rng(rng) {}
// モデルの学習モードに従ってドロップアウトを適用する。
Tensor Dropout::forward(Tensor x) const { return F::dropout(x,p,training,rng); }
// 原版と同じ tanh 近似 GELU を活性化関数として適用する。
Tensor NewGELU::forward(Tensor x) const { return F::gelu(x); }

// 自己注意の射影層・ドロップアウト層と、非学習の因果マスクを構築する。
CausalSelfAttention::CausalSelfAttention(Parameters& p, const string& name, const GPTConfig& c,
                                       const bool& training, mt19937& rng)
    : c_attn(p, name + ".c_attn", c.n_embd, 3 * c.n_embd),
      c_proj(p, name + ".c_proj", c.n_embd, c.n_embd),
      attn_dropout(c.attn_pdrop,training,rng), resid_dropout(c.resid_pdrop,training,rng),
      bias(Tensor({c.block_size,c.block_size},1).tril().view({1,1,c.block_size,c.block_size})),
      n_head(c.n_head), n_embd(c.n_embd) {}

// Q/K/V をヘッドに分割し、未来を隠した注意計算の結果を射影する。
Tensor CausalSelfAttention::forward(Tensor x) const {
    int B = x.size(0), T = x.size(1), C = x.size(2), D = C/n_head;
    auto qkv = c_attn(x).split(n_embd, 2);
    auto k = qkv[1].view({B,T,n_head,D}).transpose(1,2);
    auto q = qkv[0].view({B,T,n_head,D}).transpose(1,2);
    auto v = qkv[2].view({B,T,n_head,D}).transpose(1,2);

    auto att = q.matmul(k.transpose(-2,-1)) * (1.0/sqrt(double(k.size(-1))));
    att = att.masked_fill(bias.narrow(2,0,T).narrow(3,0,T) == 0, -numeric_limits<double>::infinity());
    att = F::softmax(att, -1);
    att = attn_dropout(att);
    auto y = att.matmul(v);
    y = y.transpose(1,2).contiguous().view({B,T,C});
    y = resid_dropout(c_proj(y));
    return y;
}

// 特徴幅を 4 倍に広げて戻す全結合層と、活性化・ドロップアウトを構築する。
MLP::MLP(Parameters& p, const string& name, const GPTConfig& c, const bool& training, mt19937& rng)
    : c_fc(p, name + ".c_fc", c.n_embd, 4 * c.n_embd),
      c_proj(p, name + ".c_proj", 4 * c.n_embd, c.n_embd), dropout(c.resid_pdrop,training,rng) {}

// Transformer ブロックの正規化層・自己注意層・MLP を構築する。
Block::Block(Parameters& p, const string& name, const GPTConfig& c, const bool& training, mt19937& rng)
    : ln_1(p, name + ".ln_1", c.n_embd), attn(p, name + ".attn", c,training,rng),
      ln_2(p, name + ".ln_2", c.n_embd), mlp(p, name + ".mlp", c,training,rng) {}

// 原版の mlpf に対応し、MLP の各層を順に適用する。
Tensor Block::mlpf(Tensor x) const { return mlp.dropout(mlp.c_proj(mlp.act(mlp.c_fc(x)))); }

// 正規化後の自己注意と MLP を計算し、それぞれの入力に残差加算する。
Tensor Block::forward(Tensor x) const {
    x = x + attn(ln_1(x));
    x = x + mlpf(ln_2(x));
    return x;
}

// トークン・位置埋め込み、ブロック列、最終正規化層をまとめて構築する。
Transformer::Transformer(Parameters& p, const GPTConfig& c, const bool& training, mt19937& rng)
    : wte(p,"transformer.wte",c.vocab_size,c.n_embd),
      wpe(p,"transformer.wpe",c.block_size,c.n_embd), drop(c.embd_pdrop,training,rng),
      ln_f(p,"transformer.ln_f",c.n_embd) {
    for (int i = 0; i < c.n_layer; ++i)
        h.emplace_back(p, "transformer.h." + to_string(i), c,training,rng);
}

// 課題用の小規模モデルの既定設定を返す。
GPTConfig GPT::get_default_config() { return {}; }

// モデル全体を構築・初期化し、残差射影の初期分散を調整する。
GPT::GPT(const GPTConfig& c, uint32_t seed)
    : config(c), rng(seed), transformer(parameters,c,training,rng),
      lm_head(parameters,"lm_head",c.n_embd,c.vocab_size,false), block_size(c.block_size) {
    _init_weights();
    // 注意層と MLP の両方の残差射影を、原版と同じ係数で再初期化する。
    normal_distribution<double> residual(0.0, 0.02 / sqrt(2.0 * c.n_layer));
    for (auto& [name, p] : parameters.named)
        if (name.find("c_proj.weight") != string::npos)
            for (double& v : p.value.data()) v = residual(rng);
}

// 全パラメータを走査し、層の種類に応じた初期値を設定する。
void GPT::_init_weights() {
    // _init_weights: Linear/Embedding は平均 0・標準偏差 0.02 の正規分布。
    // LayerNorm の重みは 1、バイアスは 0 に初期化する。
    normal_distribution<double> normal(0.0, 0.02);
    for (auto& [name, p] : parameters.named) {
        for (double& v : p.value.data()) {
            if (name.compare(name.size() - 4, 4, "bias") == 0) v = 0.0;
            else if (name.find(".ln_") != string::npos) v = 1.0;
            else v = normal(rng);
        }
    }
}

// パラメータの重み減衰フラグを引き継いだ AdamW を作る。
unique_ptr<AdamW> GPT::configure_optimizers(const TrainerConfig& train_config) {
    // Linear.weight だけが decay=true。bias・LayerNorm・Embedding は false。
    return make_unique<AdamW>(parameters,train_config);
}

// 全 Dropout が参照する学習モードを有効にする。
void GPT::train() { training = true; }
// 全 Dropout が参照する学習モードを無効にする。
void GPT::eval() { training = false; }
// モデルの全パラメータの勾配をゼロにする。
void GPT::zero_grad() { parameters.zero_grad(); }

// 埋め込みから出力層まで順伝播し、教師があれば予測損失も計算する。
ForwardResult GPT::forward(const Tokens& idx, int B, int T, const Tokens& targets) {
    Tokens pos(T); iota(pos.begin(), pos.end(), 0);
    auto tok_emb = transformer.wte(idx, {B,T});
    auto pos_emb = transformer.wpe(pos, {1,T});
    auto x = transformer.drop(tok_emb + pos_emb);
    for (const auto& block : transformer.h) x = block(x);
    x = transformer.ln_f(x);
    auto logits = lm_head(x);
    Tensor loss;
    if (!targets.empty()) loss = F::cross_entropy(logits.view({-1,logits.size(-1)}),targets,-1);
    return {logits,loss};
}

// model(...) という呼び方を順伝播関数に転送する。
ForwardResult GPT::operator()(const Tokens& idx, int B, int T, const Tokens& targets) {
    return forward(idx,B,T,targets);
}

// 入力列の続きを 1 トークンずつ予測し、貪欲法またはサンプリングで追加する。
// GPT.generate と同じ仕様: 勾配計算の無効化だけでは eval() に切り替わらない。
// idx の形状は [B,T]、返すトークン列の形状は [B,T+max_new_tokens]。
Tokens GPT::generate(Tokens idx, int B, int T, int max_new_tokens, double temperature,
                     bool do_sample, int top_k) {
    NoGrad no_grad;
    for (int step = 0; step < max_new_tokens; ++step) {
        const int context = min(T, block_size);
        Tokens idx_cond(size_t(B) * context);
        for (int b = 0; b < B; ++b)
            for (int t = 0; t < context; ++t) idx_cond[b * context + t] = idx[b * T + T - context + t];
        auto logits = (*this)(idx_cond,B,context).logits;
        logits = logits.narrow(1,context-1,1).view({B,config.vocab_size}) / temperature;
        Tokens next(size_t(B) * (T + 1));
        if (top_k > 0) {
            Tensor threshold({B,1});
            for (int b = 0; b < B; ++b) {
                auto first = logits.data().begin() + size_t(b)*config.vocab_size;
                Values sorted(first,first+config.vocab_size);
                sort(sorted.begin(), sorted.end(), greater<double>());
                threshold.data()[b] = sorted[top_k-1];
            }
            // 原式 logits[logits < v[:,[-1]]] = -inf に対応し、閾値と同値の要素は残す。
            logits = logits.masked_fill(logits < threshold, -numeric_limits<double>::infinity());
        }
        auto probs = F::softmax(logits,-1);
        for (int b = 0; b < B; ++b) {
            auto first = probs.data().begin() + size_t(b)*config.vocab_size;
            const int id = do_sample ? discrete_distribution<int>(first,first+config.vocab_size)(rng)
                : int(max_element(first,first+config.vocab_size)-first);
            for (int t = 0; t < T; ++t) next[b * (T + 1) + t] = idx[b * T + t];
            next[b * (T + 1) + T] = id;
        }
        idx = std::move(next); ++T;
    }
    return idx;
}

} // 名前空間 mingpt の終わり
