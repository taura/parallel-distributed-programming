#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

// 画像サイズの定義（コンパイル時に -DWIDTH=... で上書き可能）
#ifndef WIDTH
#define WIDTH 800
#endif
#ifndef HEIGHT
#define HEIGHT 600
#endif

// シンプルな3次元ベクトルクラス
struct Vec3 {
    double x, y, z;
    Vec3(double x=0, double y=0, double z=0) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3& v) const { return Vec3(x+v.x, y+v.y, z+v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x-v.x, y-v.y, z-v.z); }
    Vec3 operator*(double d) const { return Vec3(x*d, y*d, z*d); }
    Vec3 normalize() const {
        double mg = std::sqrt(x*x + y*y + z*z);
        return Vec3(x/mg, y/mg, z/mg);
    }
};

double dot(const Vec3& a, const Vec3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

// 球体と光線の交差判定
double hit_sphere(const Vec3& center, double radius, const Vec3& ray_origin, const Vec3& ray_dir) {
    Vec3 oc = ray_origin - center;
    double a = dot(ray_dir, ray_dir);
    double b = 2.0 * dot(oc, ray_dir);
    double c = dot(oc, oc) - radius*radius;
    double discriminant = b*b - 4*a*c;
    
    if (discriminant < 0) {
        return -1.0;
    } else {
        return (-b - std::sqrt(discriminant)) / (2.0*a);
    }
}

// 背景色または球体の色を計算
Vec3 color(const Vec3& ray_origin, const Vec3& ray_dir) {
    double t = hit_sphere(Vec3(0, 0, -1), 0.5, ray_origin, ray_dir);
    if (t > 0.0) {
        // 法線ベクトルから色を計算（球体の表面）
        Vec3 N = (ray_origin + ray_dir * t - Vec3(0, 0, -1)).normalize();
        return Vec3(0.5*(N.x+1), 0.5*(N.y+1), 0.5*(N.z+1));
    }
    // 背景のグラデーション
    Vec3 unit_direction = ray_dir.normalize();
    t = 0.5 * (unit_direction.y + 1.0);
    return Vec3(1.0, 1.0, 1.0) * (1.0 - t) + Vec3(0.5, 0.7, 1.0) * t;
}

int main() {
    std::ofstream out("output.ppm");
    out << "P3\n" << WIDTH << " " << HEIGHT << "\n255\n";

    Vec3 origin(0, 0, 0);

    // ピクセルごとの計算（将来的な並列化のターゲット）
    for (int j = HEIGHT - 1; j >= 0; --j) {
        for (int i = 0; i < WIDTH; ++i) {
            double u = double(i) / double(WIDTH);
            double v = double(j) / double(HEIGHT);
            
            // 簡易的なカメラ設定
            Vec3 lower_left_corner(-2.0, -1.5, -1.0);
            Vec3 horizontal(4.0, 0.0, 0.0);
            Vec3 vertical(0.0, 3.0, 0.0);
            
            Vec3 direction = lower_left_corner + horizontal * u + vertical * v - origin;
            Vec3 c = color(origin, direction);

            int ir = int(255.99 * c.x);
            int ig = int(255.99 * c.y);
            int ib = int(255.99 * c.z);
            out << ir << " " << ig << " " << ib << "\n";
        }
    }
    out.close();
    std::cout << "Rendered output.ppm (" << WIDTH << "x" << HEIGHT << ")" << std::endl;
    return 0;
}