# 两种光照计算模式

`raytrace.cc` 现在有两个串行计算模式，用 `--integrator` 选择：

- `whitted`：原来的环境光、Phong 直接光照、阴影和理想镜面反射。
- `path`：蒙特卡洛路径追踪。漫反射光线也继续传播，估计来自其他表面的间接光。

`path_trace.h` 实现路径追踪，`microfacet.h` 实现 GGX/VNDF，`character_scene.h` 定义人物模型。
程序本身只使用 C++17 标准库；无需下载模型或贴图。

## 计算的是什么

路径追踪估计渲染方程中的半球积分：

\[
L_o(x,\omega_o) = L_e(x,\omega_o) +
\int f(x,\omega_i,\omega_o)L_i(x,\omega_i)
|n\cdot\omega_i|\,d\omega_i.
\]

从每个像素发射多条随机光线，每次碰撞选择一种传播事件。
路径权重 `beta` 累积 `f * cos / PDF`，最终对样本求平均。
红色表面反射的光因此可以照亮旁边的白色表面。

### 材质与采样

1. **普通表面，`--bsdf ideal`（默认）**：Lambert 漫反射与理想镜面反射的混合。
   `color` 是漫反射反照率，`reflectivity` 是镜面分量的权重。
   漫反射按余弦加权半球分布采样，`PDF = cos(theta) / pi`。
   原来的 `ambient/diffuse/specular/shininess` 参数只用于 `whitted` 模式。
2. **玻璃**：`ior > 1` 表示空气中的透明实体。
   用 Snell 定律计算折射方向，用非偏振 Fresnel 公式选择反射或折射，支持全反射。
   透射路径包含辐亮度传输所需的折射率平方因子。
3. **光源采样（NEE）**：在可见表面主动采样面积光源，发送遮挡检测光线。
   面积概率密度转换为立体角概率密度，包含距离平方与光源朝向。
4. **多重重要性采样（MIS）**：以 power heuristic 合并光源采样和材质采样。
   两种方法都可能找到同一条光路；MIS 避免重复计算能量，并通常降低方差。
5. **Russian roulette**：达到指定深度后随机终止路径，存活路径除以存活概率。
   这个随机终止步骤不改变期望值。

每次碰撞只继续一条随机路径，程序并不展开一棵指数增长的光线树。

### GGX 微表面模型：`--bsdf ggx`

把表面看成许多微小镜面的集合。粗糙度控制微表面法线分布，因而控制高光宽度与反射模糊程度。
反射项采用 Cook–Torrance 形式：

\[
f_s = \frac{D_{GGX}(h)\,G_{Smith}(\omega_o,\omega_i)\,F(\omega_o\cdot h)}
{4(n\cdot\omega_o)(n\cdot\omega_i)}.
\]

- 使用各向同性 GGX 分布，`alpha = roughness^2`。
- 用可分离 Smith 遮蔽项表示微表面间的遮挡。
- 用 RGB Schlick Fresnel 近似反射率：非金属 `F0=0.04`，金属 `F0=color`。
  `Material::metallic` 在两者之间混合；金属部分没有漫反射。
- 非金属的漫反射乘上 `(1-F)`。GGX 模式用 `color/metallic/roughness` 定义不透明材质，
  原来的 `reflectivity` 只用于 ideal/whitted 模式。
- 使用 GGX **可见法线采样（VNDF）**：先按观察方向拉伸采样空间，再从投影圆盘采样微法线。
  这比不考虑可见性的法线采样更适合掠射方向。
- 材质采样同时包含漫反射和 GGX 分量；路径权重与 MIS 都使用完整的混合 PDF。
  反射到表面下方的样本贡献为零，不进行条件重采样。

`--roughness 0.1` 较光滑，`--roughness 0.6` 较粗糙，允许范围为 `[0.02,1]`。
此参数会覆盖所有不透明材质的粗糙度；省略它则使用各材质的数值。
玻璃仍使用前述光滑界面折射模型。

本实现是**单次微表面散射模型**，没有多次微表面散射能量补偿。
在粗糙度较高时，即使理想白金属也会有反射能量损失；这正是白炉测试测量的量。
金属 Fresnel 使用 RGB 近似，没有计算随波长变化的复折射率。

### 自适应采样：`--adaptive T`

默认 `--adaptive 0`，每个像素使用固定数量的路径。
启用后，`--spp` 变为最大路径数，`--min-spp` 指定最小路径数（至少 8，默认 16）。
用 Welford 在线算法分别维护线性 RGB 三个通道的均值和方差，达到最小样本数后每 8 个样本检查：

\[
1.96\sqrt{s_c^2/N} \le T\max(|\overline C_c|, 0.05),\qquad c\in\{R,G,B\}.
\]

三个通道都通过时，该像素提前停止；否则继续直到样本上限。
`T` 越小，停止条件越严格。统计计算在色调映射和 gamma 变换之前进行。

这是**经验停止指标，不是严格误差保证**。用同一批样本判断停止并估计颜色可能引入停止偏差，
少量样本也可能漏掉罕见高亮路径。做画质/性能比较时应同时保留较高固定样本数的参考图。
相同参数与种子仍可复现。

`--samples-out samples.ppm` 输出实际采样数热图：颜色按“实际样本数 / 样本上限”映射，
偏蓝表示少，红色表示达到上限。终端也输出实际样本数的最小值、平均值、最大值与总路径数。

## 光源定义与模型范围

- 在 `path` 模式中，方形光源水平放置，向下发光；其辐亮度为
  `20 * intensity * color`，背面不发光。改变面积会改变总发光功率。
- `--light-radius 0` 表示点光源，按距离平方衰减；点光源始终用直接光源采样估计。
- `--env` 缩放渐变天空的辐亮度。天空光仅通过路径方向采样。
- 玻璃支持空气中互不嵌套、闭合且法线朝外的实体，例如验证场景中的球体。
  当前没有体积吸收、散射、光谱色散或粗糙玻璃模型。
- `--max-depth` 是硬上限，截断长光路会引入偏差。
  增大它可检查截断影响；RR 的概率补偿并不能消除硬截断偏差。
- 玻璃焦散可以通过随机光路出现，但这个单向路径追踪器对焦散的采样效率较低。

## 运行与验证

重新执行 notebook 的编译单元，然后在 Miyabi 的计算作业中运行。

先用小型光传输验证场景观察间接光：

```bash
./raytrace --integrator path --scene transport --w 320 --h 320 --spp 64 --light-samples 1 --max-depth 8 --env 0 --seed 42 --out transport.ppm
```

场景包含红、绿墙壁，白色漫反射球、镜面球、玻璃球和顶灯。
同一场景用 `--max-depth 0` 可观察只估计直接光时的区别。
`--objects` 不改变这个固定的验证场景。

人物也可以使用新的光照模型：

```bash
./raytrace --integrator path --scene character --detail 1 --w 240 --h 300 --spp 8 --light-samples 1 --max-depth 6 --seed 42 --out gallery.ppm
```

模型复杂度和采样参数是不同的控制项：

| 参数 | 含义 |
|---|---|
| `--detail` | 人物三角网格精细度 |
| `--spp` | 每像素路径数，主要用于降低噪声 |
| `--max-depth` | 最大散射次数；路径模式默认 8 |
| `--light-samples` | 每次表面碰撞、每个面积光源的 NEE 样本数 |
| `--nee 1` / `--nee 0` | 开启 NEE+MIS / 仅用材质方向采样面积光源 |
| `--rr-depth` | 从第几次散射起启用随机终止，默认 3 |
| `--env` | 天空光强度，默认 0.15，设为 0 可隔离室内光源 |
| `--bsdf ideal/ggx` | 理想漫反射/镜面混合，或 GGX/VNDF 粗糙表面模型 |
| `--roughness` | 覆盖 GGX 不透明材质的粗糙度 |
| `--adaptive` | RGB 自适应停止阈值；0 表示固定采样 |
| `--min-spp` | 自适应采样的最小路径数 |
| `--samples-out` | 保存实际采样数热图（PPM） |

GGX 与自适应采样组合示例（先在 Miyabi 上测量小规模耗时）：

```bash
./raytrace --integrator path --bsdf ggx --scene character --detail 1 --w 160 --h 200 --spp 32 --min-spp 16 --adaptive 0.08 --light-samples 1 --max-depth 6 --out gallery.ppm --samples-out samples.ppm
```

先测运行时间，再增大参数。人物有大量三角形，明显比验证场景慢。
所有模式仍逐一遍历物体求交，尚未加入并行化或 BVH。

短自检：

```bash
./raytrace --check-geometry
./raytrace --check-transport
./raytrace --check-advanced
```

光传输自检检查法向入射的玻璃反射率约为 4%、Snell 折射、全反射、
闭合玻璃出入界面的权重抵消、余弦采样统计量、镜面到光源的辐亮度，
并比较 NEE、BSDF、MIS 与 MIS+RR 对同一个漫反射积分的估计。
高级自检比较均匀半球采样与 VNDF 的白炉反射能量、PDF 积分与样本接受率，
检查 BRDF 互易性、掠射角和粗糙度边界、GGX 混合材质的 NEE/MIS 一致性，
以及 RGB 在线方差的停止判据。

输出的平均亮度、路径段数、阴影光线数、各类散射事件数与运行时间，
可用于检查结果和分析性能。增加 `spp` 通常降低噪声，但不保证每个像素单调变好。
