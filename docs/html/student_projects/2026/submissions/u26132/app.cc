#include <cstdio>
#include <cmath>
#include <random>

const int W = 512, H = 512;
const int N = 50000;          // ガウス数（調整可能）
const double PI = 3.1415926535;

// ガウスパラメータ構造体
struct Gauss {
    double pos[3];
    double dir[3];
    double sigma;
    double weight;
};

// 1 ガウスから (x,y) ピクセルへの寄与を計算（簡易化版）
inline double contrib(const Gauss& g, int x, int y) {
    // 視点はカメラ原点 (0,0,0) から -z 方向まで正射影すると仮定
    double nx = cos(g.dir[0]) * g.pos[0] + sin(g.dir[0]) * g.pos[2];
    double ny = cos(g.dir[1]) * g.pos[0] + sin(g.dir[1]) * g.pos[2];
    double nz = cos(g.dir[2]) * g.pos[0] + sin(g.dir[2]) * g.pos[2];

    // スクリーン平面上の対応座標
    double sx = nx * 10.0;         // スケールは調整可
    double sy = ny * 10.0;

    double dx = sx - x;
    double dy = sy - y;
    double r2 = dx*dx + dy*dy;
    double gauss2d = std::exp(-r2 / (2.0 * g.sigma * g.sigma));
    return g.weight * gauss2d;
}

int main(int argc, char* argv[]) {
    std::mt19937 gen(42);
    std::normal_distribution<double> nd(0.0, 1.0);

    Gauss gs[N];
    // ランダムにガウス配置
    for (int i=0;i<N;++i){
        gs[i].pos[0]=nd(gen);
        gs[i].pos[1]=nd(gen);
        gs[i].pos[2]=5.0+nd(gen)*5.0;
        double angle = nd(gen);
        gs[i].dir[0]=cos(angle);
        gs[i].dir[1]=sin(angle);
        gs[i].dir[2]=0.0;
        gs[i].sigma = 0.5+nd(gen)*1.0;
        gs[i].weight = 1.0+nd(gen);
    }

    // PPM ヘッダー書込
    FILE* f = fopen("out.ppm","wb");
    fprintf(f, "P6 %d %d 255\n", W, H);
    for (int y=0; y<H; ++y){
        for (int x=0; x<W; ++x){
            double sum = 0.0;
            for (int i=0;i<N;++i){
                sum += contrib(gs[i], x, y);
            }
            // 正規化 → 0‑255
            unsigned char col = static_cast<unsigned char>(std::min(255.0, sum*10.0));
            fputc(col, f);
            fputc(col, f);
            fputc(col, f);
        }
    }
    fclose(f);
    return 0;
}