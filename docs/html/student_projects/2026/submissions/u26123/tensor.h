// minGPT に必要な範囲の、逐次テンソル演算と自動微分。
#ifndef MINGPT_TENSOR_H
#define MINGPT_TENSOR_H

#include <functional>
#include <memory>
#include <random>
#include <vector>

namespace mingpt {
using namespace std;
using Shape = vector<int>;
using Values = vector<double>;
using Tokens = vector<int>;

// スコープ内の演算記録を止める。モデルの train/eval とは独立。
class NoGrad {
    bool previous;
public:
    // 現在の記録状態を保存し、演算グラフの記録を停止する。
    NoGrad();
    // スコープに入る前の記録状態へ戻す。
    ~NoGrad();
    // 記録状態の復元を重複させないため、コピーを禁止する。
    NoGrad(const NoGrad&) = delete;
    // 記録状態の管理を別のガードへ代入することを禁止する。
    NoGrad& operator=(const NoGrad&) = delete;
};

// 値・勾配・親への参照を持つ内部ノード。定義は tensor.cpp に置く。
struct Node;

// コピーは同じ値・勾配を指すハンドル。演算の親を保持し、loss.backward() で微分する。
class Tensor {
    shared_ptr<Node> node;
public:
    // 省略したバイアスや損失を表す、未定義テンソルを作る。
    Tensor();
    // 指定形状の値を初期化し、必要なら勾配配列も確保する。
    explicit Tensor(Shape shape, double value = 0, bool requires_grad = false);
    // 内部ノードが存在するかを返す。
    bool defined() const;
    // このテンソルが勾配を保持するかを返す。
    bool requires_grad() const;
    // 全ての軸の大きさを返す。
    const Shape& shape() const;
    // 指定軸の大きさを返す。負の番号は末尾から数える。
    int size(int dim) const;
    // テンソルの全要素数を返す。
    size_t numel() const;
    // スカラーテンソルの値を実数として取り出す。
    double item() const;
    // 共有している値配列への参照を返す。
    Values& data() const;
    // 勾配配列への参照を返す。
    Values& grad() const;
    // このテンソルの勾配をゼロにする。
    void zero_grad() const;
    // スカラー損失から演算グラフを逆にたどり、勾配を加算する。
    void backward() const;

    // 値配列を共有したまま形状を変える。1 個の -1 は自動推定する。
    Tensor view(Shape shape) const;
    // 2 軸を入れ替えた連続配列を作る。
    Tensor transpose(int dim0, int dim1) const;
    // 連続配列のハンドルを返す。この実装では追加コピーは不要。
    Tensor contiguous() const;
    // 最後の 2 軸を行列とみなし、上三角部分をゼロにする。
    Tensor tril() const;
    // 指定軸の区間を切り出したテンソルを返す。
    Tensor narrow(int dim, int start, int length) const;
    // 指定軸を一定幅で分割したテンソル列を返す。
    vector<Tensor> split(int size, int dim = -1) const;
    // 最後の 2 軸で行列積を計算し、バッチ軸はブロードキャストする。
    Tensor matmul(const Tensor& other) const;
    // ブロードキャストを適用して要素ごとに加算する。
    Tensor operator+(const Tensor& other) const;
    // ブロードキャストを適用して要素ごとに減算する。
    Tensor operator-(const Tensor& other) const;
    // ブロードキャストを適用して要素ごとに掛ける。行列積は matmul を使う。
    Tensor operator*(const Tensor& other) const;
    // ブロードキャストを適用して要素ごとに割る。
    Tensor operator/(const Tensor& other) const;
    // 全要素にスカラーを加える。
    Tensor operator+(double scalar) const;
    // 全要素からスカラーを引く。
    Tensor operator-(double scalar) const;
    // 全要素にスカラーを掛ける。
    Tensor operator*(double scalar) const;
    // 全要素をスカラーで割る。
    Tensor operator/(double scalar) const;
    // 全要素の符号を反転する。
    Tensor operator-() const;
    // 要素ごとの小なり比較を、ブロードキャスト可能な非微分マスクとして返す。
    Tensor operator<(const Tensor& other) const;
    // スカラーとの小なり比較を、非微分マスクとして返す。
    Tensor operator<(double scalar) const;
    // スカラーとの等値比較を、微分しない 0/1 テンソルとして返す。
    Tensor operator==(double scalar) const;
    // 全要素を指定した指数で累乗する。
    Tensor pow(double exponent) const;
    // 全要素に双曲線正接を適用する。
    Tensor tanh() const;
    // マスクが非ゼロの位置を指定値で置き換える。
    Tensor masked_fill(const Tensor& mask, double value) const;
    // 最後の軸のスコアを、和が 1 の確率へ変換する。
    Tensor softmax(int dim = -1) const;

    // 基本演算の出力を作り、必要なら親テンソルを記録する。
    static Tensor result(Shape shape, vector<Tensor> parents);
    // 出力勾配から親の勾配を計算する関数を登録する。
    void record(function<void(const Values&)> backward) const;
    // 勾配が必要な場合だけ、指定要素へ微分を加算する。
    void add_grad(size_t index, double value) const;
};

// スカラーを左側に置いた加算を、Tensor 側の演算へ転送する。
Tensor operator+(double scalar, const Tensor& x);
// スカラーから Tensor の各要素を引く。
Tensor operator-(double scalar, const Tensor& x);
// スカラーを左側に置いた乗算を、Tensor 側の演算へ転送する。
Tensor operator*(double scalar, const Tensor& x);
// スカラーを Tensor の各要素で割る。
Tensor operator/(double scalar, const Tensor& x);

// torch.nn.functional に相当する演算。
namespace F {
// 整数 ID に対応する埋め込み行を取り出し、入力形状に並べる。
Tensor embedding(const Tokens& ids, const Tensor& weight, Shape input_shape);
// 最後の軸に全結合変換を適用し、定義されていればバイアスを加える。
Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias = Tensor());
// 最後の軸で正規化し、学習可能な重みとバイアスを適用する。
Tensor layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias);
// 原 minGPT と同じ tanh 近似による GELU を適用する。
Tensor gelu(const Tensor& x);
// 学習時に要素を確率 p でゼロにし、残る要素の期待値を補正する。
Tensor dropout(const Tensor& x, double p, bool training, mt19937& rng);
// Tensor の softmax を呼ぶ、関数形式の入口。
Tensor softmax(const Tensor& x, int dim = -1);
// 無視ラベルを除く全位置で、次トークン予測の交差エントロピーを平均する。
Tensor cross_entropy(const Tensor& logits, const Tokens& targets, int ignore_index = -1);
}

} // 名前空間 mingpt
#endif
