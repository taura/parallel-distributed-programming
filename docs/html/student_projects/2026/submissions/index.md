<link rel="stylesheet" href="../../../../scripts/style.css">

# Submissions of pd02_app — grouped by area / 分野別一覧

- Each folder `uXXXXX/` contains `problem.md` (problem description) and the code / 各フォルダに `problem.md`（問題設定）とコードがあります
- Groups below are a rough clustering by application area, made only to help you browse; they are **not** teams / 下の分類は閲覧用の大まかなもので，チーム分けではありません
- "Lang" = language `problem.md` is written in (EN/JA) / problem.md の記述言語

- Take [this survey on making teams](https://forms.gle/HTLtmzxbp76omQuq8) by 2026/11/07 (Sat) and don't forget to report it to UTOL


## Computer graphics: ray tracing / path tracing / rendering (コンピュータグラフィクス)

### Path tracing (Monte Carlo global illumination)

| ID                                                          | Topic                                             | Summary                                                               | Lang |
|-------------------------------------------------------------|---------------------------------------------------|-----------------------------------------------------------------------|------|
| u26103 [md](u26103/problem.md), [html](u26103/problem.html) | Indoor path tracing with sunlight through windows | Room with furniture, window light, indirect light, mirror sphere      | JA   |
| u26106 [md](u26106/problem.md), [html](u26106/problem.html) | Cornell box path tracer                           | Lambertian BRDF + next event estimation, area light                   | EN   |
| u26136 [md](u26136/problem.md), [html](u26136/problem.html) | Pure path tracer                                  | Cornell box with spheres, triangle mesh, area light                   | EN   |
| u26147 [md](u26147/problem.md), [html](u26147/problem.html) | Path tracer with BVH                              | Cornell box with matte / mirror / glass spheres, BVH                  | EN   |
| u26177 [md](u26177/problem.md), [html](u26177/problem.html) | Ray tracer with several scenes                    | Procedural scenes (character, gallery, ...), microfacet, path tracing | EN   |
| u26133 [md](u26133/problem.md), [html](u26133/problem.html) | Ray tracing many spheres                          | Stochastic sampling, many material spheres                            | EN   |
| u26158 [md](u26158/problem.md), [html](u26158/problem.html) | "Ray Tracing in One Weekend"                      | Implementation following the book                                     | EN   |
| u26153 [md](u26153/problem.md), [html](u26153/problem.html) | Ray tracing (metal sphere + terrain)              | Multi-sample rendering to BMP                                         | EN   |

### Whitted-style / simple ray tracing

| ID                                                          | Topic                          | Summary                                               | Lang |
|-------------------------------------------------------------|--------------------------------|-------------------------------------------------------|------|
| u26129 [md](u26129/problem.md), [html](u26129/problem.html) | Spheres and checkerboard floor | Shadows, mirror reflection                            | JA   |
| u26134 [md](u26134/problem.md), [html](u26134/problem.html) | Ray tracer                     | Spheres, mirror, plane, point light, supersampling    | EN   |
| u26143 [md](u26143/problem.md), [html](u26143/problem.html) | Whitted-style ray tracer       | Direct illumination, specular reflection / refraction | EN   |
| u26155 [md](u26155/problem.md), [html](u26155/problem.html) | Simple ray tracer              | Serial ray tracer, PPM output                         | JA   |

### Other graphics / images

| ID                                                          | Topic                  | Summary                                                    | Lang |
|-------------------------------------------------------------|------------------------|------------------------------------------------------------|------|
| u26112 [md](u26112/problem.md), [html](u26112/problem.html) | 3D software rasterizer | Triangle rasterization, z-buffer, lighting of many objects | EN   |
| u26117 [md](u26117/problem.md), [html](u26117/problem.html) | Mandelbrot fractal     | Escape-time rendering, load imbalance                      | EN   |
| u26141 [md](u26141/problem.md), [html](u26141/problem.html) | Mandelbrot set         | Rendering + area estimation                                | EN   |
| u26127 [md](u26127/problem.md), [html](u26127/problem.html) | Image stitching        | Panorama by feature matching and warping                   | EN   |

## LLM / Transformer / attention (大規模言語モデル・Transformer)

### LLM inference

| ID                                                          | Topic                           | Summary                                                      | Lang |
|-------------------------------------------------------------|---------------------------------|--------------------------------------------------------------|------|
| u26108 [md](u26108/problem.md), [html](u26108/problem.html) | LLM inference with KV cache     | Transformer with random weights, focus on matrix computation | JA   |
| u26154 [md](u26154/problem.md), [html](u26154/problem.html) | Tiny LLM inference              | Pretrained pythia-14m, autoregressive decoding               | EN   |
| u26170 [md](u26170/problem.md), [html](u26170/problem.html) | Llama-2 inference               | TinyStories stories110M checkpoint                           | EN   |
| u26186 [md](u26186/problem.md), [html](u26186/problem.html) | Small LM inference              | TinyStories 260K (Llama-2 style), prefill / decode           | EN   |
| u26182 [md](u26182/problem.md), [html](u26182/problem.html) | Transformer inference benchmark | Decoder-only, random weights, synthetic tokens               | EN   |

### LM / Transformer training

| ID                                                          | Topic                                | Summary                                        | Lang |
|-------------------------------------------------------------|--------------------------------------|------------------------------------------------|------|
| u26123 [md](u26123/problem.md), [html](u26123/problem.html) | Training a small GPT to sort digits  | C++ port of minGPT                             | EN   |
| u26135 [md](u26135/problem.md), [html](u26135/problem.html) | Training a small LLM                 | Character-level, Tiny Shakespeare              | EN   |
| u26173 [md](u26173/problem.md), [html](u26173/problem.html) | Tiny GPT-style character LM training | Byte-level tokens, next-character prediction   | EN   |
| u26149 [md](u26149/problem.md), [html](u26149/problem.html) | Character-level MLP language model   | MLP (not Transformer) LM, forward/backward/SGD | EN   |
| u26146 [md](u26146/problem.md), [html](u26146/problem.html) | Tiny Transformer for router commands | English request → configuration commands       | EN   |
| u26185 [md](u26185/problem.md), [html](u26185/problem.html) | Small Transformer on MQAR            | In-context associative recall task             | EN   |

### Attention kernels

| ID                                                          | Topic                              | Summary                                                  | Lang |
|-------------------------------------------------------------|------------------------------------|----------------------------------------------------------|------|
| u26132 [md](u26132/problem.md), [html](u26132/problem.html) | Scaled dot-product attention       | softmax(QK^T/√d)V (+ small LM)                           | EN   |
| u26139 [md](u26139/problem.md), [html](u26139/problem.html) | Multi-head attention               | Attention with seq_len / dim / heads parameters          | EN   |
| u26152 [md](u26152/problem.md), [html](u26152/problem.html) | Attention                          | Serial attention, visualized as image                    | EN   |
| u26101 [md](u26101/problem.md), [html](u26101/problem.html) | Clustering-routed sparse attention | k-means routes queries to clusters (Routing Transformer) | EN   |

## Other machine learning / reinforcement learning (その他の機械学習)

| ID                                                          | Topic                            | Summary                                          | Lang |
|-------------------------------------------------------------|----------------------------------|--------------------------------------------------|------|
| u26102 [md](u26102/problem.md), [html](u26102/problem.html) | MLP regression                   | Fit a sine wave with a small MLP, mini-batch SGD | JA   |
| u26142 [md](u26142/problem.md), [html](u26142/problem.html) | PPO (reinforcement learning)     | Robot navigation on a grid                       | EN   |
| u26178 [md](u26178/problem.md), [html](u26178/problem.html) | Mini vector database (IVF index) | Approximate nearest-neighbor search              | EN   |

## Physical simulation (物理シミュレーション)

### Fluid dynamics

| ID                                                          | Topic                  | Summary                                                            | Lang |
|-------------------------------------------------------------|------------------------|--------------------------------------------------------------------|------|
| u26109 [md](u26109/problem.md), [html](u26109/problem.html) | Flow around an airfoil | Lattice Boltzmann method, lift / drag                              | EN   |
| u26115 [md](u26115/problem.md), [html](u26115/problem.html) | SPH fluid              | Dam-break-like water block collapse, particles                     | EN   |
| u26124 [md](u26124/problem.md), [html](u26124/problem.html) | 2D waterdrop           | Incompressible grid-based fluid, falling drop                      | EN   |
| u26168 [md](u26168/problem.md), [html](u26168/problem.html) | 2D Navier–Stokes       | Taylor–Green vortex, finite difference, compared to exact solution | EN   |
| u26171 [md](u26171/problem.md), [html](u26171/problem.html) | Particle hydrodynamics | Rotne–Prager–Yamakawa N-body interaction                           | EN   |

### Waves, fields, materials, others

| ID                                                          | Topic                                 | Summary                                             | Lang |
|-------------------------------------------------------------|---------------------------------------|-----------------------------------------------------|------|
| u26160 [md](u26160/problem.md), [html](u26160/problem.html) | 2D acoustic wave                      | Finite difference wave equation                     | EN   |
| u26120 [md](u26120/problem.md), [html](u26120/problem.html) | 2D MOS electrostatics                 | Poisson equation for a semiconductor structure      | EN   |
| u26169 [md](u26169/problem.md), [html](u26169/problem.html) | Atomic DFT                            | Kohn–Sham calculation of He atom                    | JA   |
| u26140 [md](u26140/problem.md), [html](u26140/problem.html) | Trajectory analysis (MSD)             | Mean-squared displacement and diffusion coefficient | EN   |
| u26130 [md](u26130/problem.md), [html](u26130/problem.html) | Quantum surface code error correction | Monte Carlo simulation of decoder                   | EN   |

## Robotics / control (ロボティクス・制御)

| ID                                                          | Topic                                        | Summary                                     | Lang |
|-------------------------------------------------------------|----------------------------------------------|---------------------------------------------|------|
| u26104 [md](u26104/problem.md), [html](u26104/problem.html) | Pose graph optimization (Graph SLAM backend) | 2D poses, loop closure, sparse Cholesky     | JA   |
| u26164 [md](u26164/problem.md), [html](u26164/problem.html) | LiDAR occupancy grid mapping                 | Ray casting + grid update from scans        | EN   |
| u26125 [md](u26125/problem.md), [html](u26125/problem.html) | Model predictive path integral (MPPI)        | Sampling-based control of pendulum swing-up | EN   |
| u26183 [md](u26183/problem.md), [html](u26183/problem.html) | Monte Carlo trajectory optimization          | Mobile robot with obstacles                 | EN   |

(see also u26142 PPO navigation in group 3)

## Cryptography / security (暗号・セキュリティ)

| ID                                                          | Topic                                    | Summary                                           | Lang |
|-------------------------------------------------------------|------------------------------------------|---------------------------------------------------|------|
| u26113 [md](u26113/problem.md), [html](u26113/problem.html) | CKKS homomorphic encryption              | NTT, polynomial multiplication, encrypt / decrypt | EN   |
| u26163 [md](u26163/problem.md), [html](u26163/problem.html) | CKKS-style NTT polynomial multiplication | Negacyclic product over multiple RNS moduli       | EN   |
| u26174 [md](u26174/problem.md), [html](u26174/problem.html) | SHA-256 proof of work                    | Nonce search with leading zero bits               | JA   |

## Strings / data structures / search (文字列・データ構造・検索)

| ID                                                          | Topic                     | Summary                                   | Lang |
|-------------------------------------------------------------|---------------------------|-------------------------------------------|------|
| u26172 [md](u26172/problem.md), [html](u26172/problem.html) | BWT / FM-index DNA search | Exact pattern search on E. coli genome    | EN   |
| u26105 [md](u26105/problem.md), [html](u26105/problem.html) | Wavelet matrix            | Construction of a succinct data structure | EN   |
| u26126 [md](u26126/problem.md), [html](u26126/problem.html) | Byte pattern scanner      | Searching a pattern in a large buffer     | EN   |

(see also u26178 vector database in group 3)

## Finance / Monte Carlo (金融)

| ID                                                          | Topic                       | Summary                                     | Lang |
|-------------------------------------------------------------|-----------------------------|---------------------------------------------|------|
| u26156 [md](u26156/problem.md), [html](u26156/problem.html) | European option pricing     | Monte Carlo under Black–Scholes model       | EN   |
| u26111 [md](u26111/problem.md), [html](u26111/problem.html) | Historical portfolio search | Search many portfolios on historical prices | EN   |

## Signal processing / wireless (信号処理・無線)

| ID                                                          | Topic                               | Summary                                             | Lang |
|-------------------------------------------------------------|-------------------------------------|-----------------------------------------------------|------|
| u26189 [md](u26189/problem.md), [html](u26189/problem.html) | FIR decimation and energy detection | IQ samples, SDR pipeline                            | EN   |
| u26166 [md](u26166/problem.md), [html](u26166/problem.html) | Indoor radio power map              | 2D ray tracing of radio waves (direct + reflection) | JA   |

## Not yet described / 未記入 {.nocount}

| ID                                                          | Note                             |
|-------------------------------------------------------------|----------------------------------|
| u26180 [md](u26180/problem.md), [html](u26180/problem.html) | problem.md is still the template |
