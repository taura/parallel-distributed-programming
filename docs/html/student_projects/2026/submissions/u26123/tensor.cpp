// 全演算は行優先の配列と逐次ループ。PyTorch 風の表記を支える最小限の実装。
#include "tensor.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <utility>

namespace mingpt {

// テンソルの値・勾配と、演算グラフの親・逆伝播処理を保持する内部ノード。
struct Node {
    Shape shape;
    shared_ptr<Values> data;
    Values grad;
    vector<Tensor> parents;
    function<void(const Values&)> backward;
};

namespace {
bool grad_enabled = true; // この実装は単一スレッド専用

// 各軸の大きさを掛け合わせて全要素数を求める。
size_t elements(const Shape& shape) {
    size_t n = 1;
    for (int d : shape) n *= d;
    return n;
}

// 負の軸番号を、先頭から数える非負の番号に直す。
int axis(int dim, int rank) {
    return dim < 0 ? dim + rank : dim;
}

// 末尾の軸から形状を合わせ、ブロードキャスト後の形状を求める。
Shape broadcast(const Shape& a, const Shape& b) {
    Shape out(max(a.size(), b.size()), 1);
    for (size_t i = 0; i < out.size(); ++i) {
        int x = i < a.size() ? a[a.size()-1-i] : 1;
        int y = i < b.size() ? b[b.size()-1-i] : 1;
        out[out.size()-1-i] = max(x, y);
    }
    return out;
}

// 出力の平坦な添字を、ブロードキャスト前の入力の添字へ戻す。
size_t offset(size_t i, const Shape& out, const Shape& in) {
    size_t result = 0, stride = 1;
    for (int d = int(out.size())-1, j = int(in.size())-1; d >= 0; --d, --j) {
        size_t c = i % out[d]; i /= out[d];
        if (j >= 0) { result += (in[j] == 1 ? 0 : c) * stride; stride *= in[j]; }
    }
    return result;
}

// 添字表に従って要素を並べ替え、逆伝播では元の位置に勾配を戻す。
Tensor remap(const Tensor& x, Shape shape, const vector<size_t>& indices) {
    auto y = Tensor::result(shape, {x});
    for (size_t i = 0; i < indices.size(); ++i) y.data()[i] = x.data()[indices[i]];
    // 並べ替え後の勾配を、元の要素の位置へ加算する。
    y.record([=](const Values& g) {
        for (size_t i = 0; i < indices.size(); ++i) x.add_grad(indices[i], g[i]);
    });
    return y;
}

// 要素ごとの値と局所微分をまとめて計算する。
Tensor elementwise(const Tensor& x, const function<pair<double,double>(double)>& f) {
    auto y = Tensor::result(x.shape(), {x});
    Values derivative(x.numel());
    for (size_t i = 0; i < x.numel(); ++i) {
        auto [value, slope] = f(x.data()[i]);
        y.data()[i] = value; derivative[i] = slope;
    }
    // 保存した局所微分と出力勾配を掛け、入力へ伝える。
    y.record([=](const Values& g) {
        for (size_t i = 0; i < g.size(); ++i) x.add_grad(i, g[i] * derivative[i]);
    });
    return y;
}

// 二入力の値と両側の局所微分を計算し、ブロードキャストの勾配加算を共通化する。
Tensor binary(const Tensor& a, const Tensor& b,
              const function<array<double,3>(double,double)>& f) {
    Shape s = broadcast(a.shape(),b.shape());
    auto y = Tensor::result(s,{a,b});
    Values da(y.numel()), db(y.numel());
    for (size_t i = 0; i < y.numel(); ++i) {
        auto [value, left, right] = f(a.data()[offset(i,s,a.shape())], b.data()[offset(i,s,b.shape())]);
        y.data()[i] = value; da[i] = left; db[i] = right;
    }
    // 複製された入力要素へ、各出力からの微分を加算する。
    y.record([=](const Values& g) {
        for (size_t i = 0; i < g.size(); ++i) {
            a.add_grad(offset(i,s,a.shape()),g[i]*da[i]);
            b.add_grad(offset(i,s,b.shape()),g[i]*db[i]);
        }
    });
    return y;
}
} // 内部の補助関数

// 現在の記録状態を保存して、このスコープの演算記録を止める。
NoGrad::NoGrad() : previous(grad_enabled) { grad_enabled = false; }
// 保存した記録状態を復元する。
NoGrad::~NoGrad() { grad_enabled = previous; }

// 省略したバイアスや損失を表す、未定義のハンドルを作る。
Tensor::Tensor() = default;
// 値配列を指定値で初期化し、必要ならゼロの勾配配列も作る。
Tensor::Tensor(Shape shape, double value, bool requires_grad) : node(make_shared<Node>()) {
    node->shape = std::move(shape);
    node->data = make_shared<Values>(elements(node->shape), value);
    if (requires_grad) node->grad.resize(numel(), 0);
}
// 内部ノードを持つハンドルかどうかを返す。
bool Tensor::defined() const { return bool(node); }
// 逆伝播の結果を保持する勾配配列があるかを返す。
bool Tensor::requires_grad() const { return defined() && !node->grad.empty(); }
// 全軸の大きさを保持する配列への参照を返す。
const Shape& Tensor::shape() const { return node->shape; }
// 負の軸番号を変換して、その軸の大きさを返す。
int Tensor::size(int dim) const { return shape()[axis(dim, int(shape().size()))]; }
// 値配列の長さ、すなわち全要素数を返す。
size_t Tensor::numel() const { return data().size(); }
// スカラーテンソルの唯一の要素を取り出す。
double Tensor::item() const { return data()[0]; }
// 共有している値配列への参照を返す。
Values& Tensor::data() const { return *node->data; }
// このノードの勾配配列への参照を返す。
Values& Tensor::grad() const { return node->grad; }
// 勾配を持つ場合に、その全要素をゼロにする。
void Tensor::zero_grad() const { if (requires_grad()) fill(grad().begin(), grad().end(), 0); }
// 勾配が必要な入力にだけ、指定要素の微分を加算する。
void Tensor::add_grad(size_t i, double v) const { if (requires_grad()) grad()[i] += v; }

// 演算結果を作り、記録が有効なら微分に必要な親への参照を保存する。
Tensor Tensor::result(Shape shape, vector<Tensor> parents) {
    // 親のいずれかが勾配を必要とするかを調べ、記録の要否を決める。
    bool track = grad_enabled && any_of(parents.begin(), parents.end(), [](const Tensor& x) { return x.requires_grad(); });
    Tensor y(std::move(shape), 0, track);
    if (track) y.node->parents = std::move(parents);
    return y;
}
// 出力勾配を受け取って親へ戻す逆伝播関数を登録する。
void Tensor::record(function<void(const Values&)> backward) const {
    if (requires_grad()) node->backward = std::move(backward);
}

// 親を依存順に並べ、その逆順でスカラー損失の勾配を伝播する。
void Tensor::backward() const {
    vector<shared_ptr<Node>> order;
    unordered_set<Node*> visited;
    // 共有ノードを一度だけ訪れ、親が先になる順序を作る。
    function<void(shared_ptr<Node>)> visit = [&](shared_ptr<Node> n) {
        if (!visited.insert(n.get()).second) return;
        for (const auto& p : n->parents) visit(p.node);
        order.push_back(n);
    };
    visit(node);
    // 中間勾配は毎回初期化し、葉（パラメータ）の勾配は加算する。
    for (auto& n : order) if (!n->parents.empty()) fill(n->grad.begin(), n->grad.end(), 0);
    grad()[0] += 1;
    for (auto it = order.rbegin(); it != order.rend(); ++it)
        if ((*it)->backward) (*it)->backward((*it)->grad);
}

// -1 の軸を推定し、値配列を共有した別形状のテンソルを返す。
Tensor Tensor::view(Shape s) const {
    int inferred = -1;
    size_t known = 1;
    for (int i = 0; i < int(s.size()); ++i) {
        if (s[i] == -1) inferred = i;
        else known *= s[i];
    }
    if (inferred >= 0) s[inferred] = int(numel()/known);
    auto y = result(s, {*this});
    y.node->data = node->data; // 形状だけを変え、値配列は共有する
    // 要素の順序は同じなので、入力の同じ位置へ勾配を戻す。
    y.record([x=*this](const Values& g) {
        for (size_t i = 0; i < g.size(); ++i) x.add_grad(i, g[i]);
    });
    return y;
}

// 2 軸の交換に対応する添字表を作り、連続配列として並べ替える。
Tensor Tensor::transpose(int dim0, int dim1) const {
    Shape s = shape();
    int a = axis(dim0, int(s.size())), b = axis(dim1, int(s.size()));
    vector<size_t> strides(s.size(), 1), indices(numel());
    for (int d = int(s.size())-2; d >= 0; --d) strides[d] = strides[d+1] * s[d+1];
    swap(s[a], s[b]);
    for (size_t i = 0; i < numel(); ++i) {
        size_t q = i, index = 0;
        for (int d = int(s.size())-1; d >= 0; --d) {
            index += (q % s[d]) * strides[d == a ? b : d == b ? a : d]; q /= s[d];
        }
        indices[i] = index;
    }
    return remap(*this, s, indices); // 常に連続配列を返すので contiguous() は不要
}

// 既に連続配列なので、同じハンドルをそのまま返す。
Tensor Tensor::contiguous() const { return *this; }

// 最後の 2 軸について、対角線より上の要素をゼロにする。
Tensor Tensor::tril() const {
    int rows = size(-2), cols = size(-1);
    Tensor mask({rows,cols});
    for (int r = 0; r < rows; ++r)
        for (int c = r+1; c < cols; ++c) mask.data()[size_t(r)*cols+c] = 1;
    return masked_fill(mask,0);
}

// 指定軸の区間に対応する添字を求め、その部分を切り出す。
Tensor Tensor::narrow(int dim, int start, int length) const {
    Shape s = shape(); int d = axis(dim, int(s.size())), original = s[d];
    size_t inner = 1;
    for (int j = d+1; j < int(s.size()); ++j) inner *= s[j];
    s[d] = length;
    vector<size_t> indices(elements(s));
    for (size_t i = 0; i < indices.size(); ++i)
        indices[i] = (i/(length*inner)*original + start)*inner + i%(length*inner);
    return remap(*this, s, indices);
}

// narrow を繰り返し、指定軸を一定幅のテンソル列に分割する。
vector<Tensor> Tensor::split(int length, int dim) const {
    vector<Tensor> parts;
    for (int start = 0; start < size(dim); start += length)
        parts.push_back(narrow(dim, start, min(length, size(dim)-start)));
    return parts;
}

// ブロードキャストして加算し、逆伝播では複製された軸の勾配を合計する。
Tensor Tensor::operator+(const Tensor& b) const {
    return binary(*this,b,[](double a, double b) { return array{a+b,1.0,1.0}; });
}

// 要素ごとに引き、右側の入力には符号を反転した勾配を返す。
Tensor Tensor::operator-(const Tensor& b) const {
    return binary(*this,b,[](double a, double b) { return array{a-b,1.0,-1.0}; });
}

// 要素ごとの積を計算し、両側に相手の値を掛けた勾配を返す。
Tensor Tensor::operator*(const Tensor& b) const {
    return binary(*this,b,[](double a, double b) { return array{a*b,b,a}; });
}

// 要素ごとの商を計算し、分子・分母への微分を登録する。
Tensor Tensor::operator/(const Tensor& b) const {
    return binary(*this,b,[](double a, double b) { return array{a/b,1/b,-a/(b*b)}; });
}

// 各要素に定数を加え、勾配はそのまま返す。
Tensor Tensor::operator+(double scalar) const {
    return elementwise(*this,[=](double x) { return pair{x+scalar,1.0}; });
}

// 定数の符号を反転し、加算として減算する。
Tensor Tensor::operator-(double scalar) const { return *this + (-scalar); }

// 値と局所微分を同じスカラーで割る。
Tensor Tensor::operator/(double scalar) const {
    return elementwise(*this, [=](double x) { return pair{x/scalar, 1/scalar}; });
}

// 値と局所微分に同じスカラーを掛ける。
Tensor Tensor::operator*(double scalar) const {
    return elementwise(*this, [=](double x) { return pair{x*scalar, scalar}; });
}

// 値と勾配の符号を反転する。
Tensor Tensor::operator-() const { return *this * -1.0; }

// 二つの Tensor を比較し、真の位置を 1 とする非微分マスクを返す。
Tensor Tensor::operator<(const Tensor& b) const {
    Shape s = broadcast(shape(),b.shape());
    Tensor mask(s);
    for (size_t i = 0; i < mask.numel(); ++i)
        mask.data()[i] = data()[offset(i,s,shape())] < b.data()[offset(i,s,b.shape())];
    return mask;
}

// 定数をスカラーテンソルとしてブロードキャストし、大小比較する。
Tensor Tensor::operator<(double scalar) const { return *this < Tensor({},scalar); }

// 等しい要素を 1、それ以外を 0 とする非微分のマスクを作る。
Tensor Tensor::operator==(double scalar) const {
    Tensor mask(shape());
    for (size_t i = 0; i < numel(); ++i) mask.data()[i] = data()[i] == scalar;
    return mask; // 比較結果は微分しない
}

// 各要素の累乗と、その局所微分を計算する。
Tensor Tensor::pow(double exponent) const {
    return elementwise(*this,[=](double x) {
        return pair{std::pow(x,exponent), exponent == 0 ? 0.0 : exponent*std::pow(x,exponent-1)};
    });
}

// 各要素の tanh と、局所微分 1-tanh(x)^2 を計算する。
Tensor Tensor::tanh() const {
    return elementwise(*this,[](double x) {
        double t = std::tanh(x);
        return pair{t,1-t*t};
    });
}

// スカラーを左側に書いた加算を実行する。
Tensor operator+(double scalar, const Tensor& x) { return x + scalar; }
// スカラーから各要素を引く。
Tensor operator-(double scalar, const Tensor& x) { return -x + scalar; }
// スカラーを左側に書いた乗算を実行する。
Tensor operator*(double scalar, const Tensor& x) { return x * scalar; }
// スカラーを各要素で割り、その商の微分を登録する。
Tensor operator/(double scalar, const Tensor& x) {
    return elementwise(x,[=](double v) { return pair{scalar/v,-scalar/(v*v)}; });
}

// 最後の 2 次元を行列とし、それより前の次元はブロードキャストする。
Tensor Tensor::matmul(const Tensor& b) const {
    auto a = *this;
    int M = a.size(-2), K = a.size(-1), N = b.size(-1);
    Shape as(a.shape().begin(),a.shape().end()-2), bs(b.shape().begin(),b.shape().end()-2);
    Shape batch = broadcast(as,bs), out = batch;
    out.insert(out.end(), {M,N});
    auto y = result(out, {a,b});
    for (size_t t = 0; t < elements(batch); ++t) {
        size_t ai = offset(t,batch,as)*M*K, bi = offset(t,batch,bs)*K*N;
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n)
                for (int k = 0; k < K; ++k)
                    y.data()[(t*M+m)*N+n] += a.data()[ai+m*K+k] * b.data()[bi+k*N+n];
    }
    // 行列積の各積項から、左右両方の入力への微分を加算する。
    y.record([=](const Values& g) {
        for (size_t t = 0; t < elements(batch); ++t) {
            size_t ai = offset(t,batch,as)*M*K, bi = offset(t,batch,bs)*K*N;
            for (int m = 0; m < M; ++m)
                for (int n = 0; n < N; ++n)
                    for (int k = 0; k < K; ++k) {
                        double d = g[(t*M+m)*N+n];
                        a.add_grad(ai+m*K+k, d*b.data()[bi+k*N+n]);
                        b.add_grad(bi+k*N+n, d*a.data()[ai+m*K+k]);
                    }
        }
    });
    return y;
}

// マスク位置を定数に置き換え、その位置への勾配を遮断する。
Tensor Tensor::masked_fill(const Tensor& mask, double value) const {
    auto x = *this;
    auto y = result(shape(), {x});
    vector<bool> hidden(numel());
    for (size_t i = 0; i < numel(); ++i) {
        hidden[i] = mask.data()[offset(i,shape(),mask.shape())] != 0;
        y.data()[i] = hidden[i] ? value : data()[i];
    }
    // 定数で置き換えていない位置にだけ勾配を戻す。
    y.record([=](const Values& g) {
        for (size_t i = 0; i < g.size(); ++i) if (!hidden[i]) x.add_grad(i,g[i]);
    });
    return y;
}

// 最後の軸を確率化し、要素間の依存を含む softmax の微分を登録する。
Tensor Tensor::softmax(int dim) const {
    auto x = *this, y = result(shape(), {x}); int C = size(dim); // dim は最後の軸
    for (size_t row = 0; row < numel(); row += C) {
        double maximum = *max_element(data().begin()+row, data().begin()+row+C), sum = 0;
        for (int c = 0; c < C; ++c) sum += y.data()[row+c] = exp(data()[row+c]-maximum);
        for (int c = 0; c < C; ++c) y.data()[row+c] /= sum;
    }
    Values p = y.data();
    // softmax の同じ行の要素間の依存を含めて微分する。
    y.record([=](const Values& g) {
        for (size_t row = 0; row < g.size(); row += C) {
            double dot = 0;
            for (int c = 0; c < C; ++c) dot += p[row+c]*g[row+c];
            for (int c = 0; c < C; ++c) x.add_grad(row+c, p[row+c]*(g[row+c]-dot));
        }
    });
    return y;
}

namespace F {
// 関数形式の呼び出しを Tensor の softmax メソッドへ転送する。
Tensor softmax(const Tensor& x, int dim) { return x.softmax(dim); }

// ID に対応する行を取り出し、逆伝播では同じ ID の勾配を加算する。
Tensor embedding(const Tokens& ids, const Tensor& weight, Shape input_shape) {
    int C = weight.size(1); input_shape.push_back(C);
    auto y = Tensor::result(input_shape, {weight});
    for (size_t r = 0; r < ids.size(); ++r) {
        for (int c = 0; c < C; ++c) y.data()[r*C+c] = weight.data()[size_t(ids[r])*C+c];
    }
    // 取り出した埋め込み行へ戻し、同じ ID の寄与を合計する。
    y.record([=](const Values& g) {
        for (size_t r = 0; r < ids.size(); ++r)
            for (int c = 0; c < C; ++c) weight.add_grad(size_t(ids[r])*C+c, g[r*C+c]);
    });
    return y;
}

// 転置・行列積・任意のバイアス加算で全結合層を計算する。
Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    auto y = x.matmul(weight.transpose(-2,-1));
    return bias.defined() ? y + bias : y;
}

// 最後の軸で正規化し、入力・重み・バイアスへの微分を登録する。
Tensor layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    int C = x.size(-1);
    auto y = Tensor::result(x.shape(), {x,weight,bias});
    Values z(x.numel()), invstd(x.numel()/C);
    for (size_t row = 0; row < x.numel(); row += C) {
        double mean = 0, var = 0;
        for (int c = 0; c < C; ++c) mean += x.data()[row+c];
        mean /= C;
        for (int c = 0; c < C; ++c) var += pow(x.data()[row+c]-mean, 2);
        invstd[row/C] = 1/sqrt(var/C+1e-5);
        for (int c = 0; c < C; ++c) {
            z[row+c] = (x.data()[row+c]-mean)*invstd[row/C];
            y.data()[row+c] = z[row+c]*weight.data()[c]+bias.data()[c];
        }
    }
    // 平均と分散を通る微分も含め、入力・重み・バイアスへ勾配を戻す。
    y.record([=](const Values& g) {
        for (size_t row = 0; row < g.size(); row += C) {
            double sum = 0, dot = 0;
            for (int c = 0; c < C; ++c) {
                double d = g[row+c]*weight.data()[c];
                sum += d; dot += d*z[row+c];
                weight.add_grad(c,g[row+c]*z[row+c]); bias.add_grad(c,g[row+c]);
            }
            for (int c = 0; c < C; ++c)
                x.add_grad(row+c, invstd[row/C]*(g[row+c]*weight.data()[c]-(sum+z[row+c]*dot)/C));
        }
    });
    return y;
}

// 原 minGPT の GELU を Tensor の数式として組み立て、微分は各演算に任せる。
Tensor gelu(const Tensor& x) {
    constexpr double a = 0.7978845608028654; // sqrt(2/π)
    return 0.5*x * (1.0 + (a*(x + 0.044715*x.pow(3))).tanh());
}

// 学習時に乱数マスクを掛け、逆伝播にも同じマスクと倍率を使う。
Tensor dropout(const Tensor& x, double p, bool training, mt19937& rng) {
    if (!training || p == 0) return x;
    bernoulli_distribution keep(1-p);
    return elementwise(x, [&](double v) {
        double mask = keep(rng) ? 1/(1-p) : 0;
        return pair{v*mask, mask};
    });
}

// 無視ラベルを除いて交差エントロピーを平均し、ロジットへの勾配を登録する。
Tensor cross_entropy(const Tensor& logits, const Tokens& targets, int ignore_index) {
    int C = logits.size(-1), count = 0;
    for (int id : targets) count += id != ignore_index;
    auto loss = Tensor::result({}, {logits});
    Values derivative(logits.numel(),0);
    if (!count) loss.data()[0] = numeric_limits<double>::quiet_NaN();
    for (size_t r = 0; r < targets.size(); ++r) {
        if (targets[r] == ignore_index) continue;
        size_t row = r*C;
        double maximum = *max_element(logits.data().begin()+row,logits.data().begin()+row+C), sum = 0;
        for (int c = 0; c < C; ++c) sum += exp(logits.data()[row+c]-maximum);
        loss.data()[0] += (maximum+log(sum)-logits.data()[row+targets[r]])/count;
        for (int c = 0; c < C; ++c)
            derivative[row+c] = (exp(logits.data()[row+c]-maximum)/sum-(c == targets[r]))/count;
    }
    // スカラー損失からの勾配を局所微分に掛け、各ロジットへ戻す。
    loss.record([=](const Values& g) {
        for (size_t i = 0; i < derivative.size(); ++i) logits.add_grad(i,g[0]*derivative[i]);
    });
    return loss;
}
} // 名前空間 F
} // 名前空間 mingpt
