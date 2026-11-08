<link rel="stylesheet" href="../scripts/style.css">

# Grace (Neoverse V2) Assembly Cheat Sheet: Scalar, NEON and SVE {.unnumbered}

* This extends the AArch64 (ARM64) cheat sheet of the programming languages course with the vector instructions (NEON and SVE) you see in the assembly produced by `clang++ -O3 -mcpu=neoverse-v2 -S`
* Grace = 72 Neoverse V2 cores; it supports NEON (128 bit) and SVE/SVE2 with **vector length (VL) = 128 bits**
* Authoritative sources
  * [Arm A64 Instruction Set Architecture](https://developer.arm.com/documentation/ddi0602/latest/) (every instruction)
  * [Arm Intrinsics (NEON and SVE)](https://developer.arm.com/architectures/instruction-sets/intrinsics/) (C functions corresponding to instructions)
  * [Arm Neoverse V2 Software Optimization Guide](https://developer.arm.com/documentation/PJDOC-466751330-593177/latest/) (latency/throughput of each instruction)

# Basics

* A C/C++ function is translated into a series of assembly instructions, which begins with a label similar to function name
* In the case of C, the label is exactly the same as the function name (e.g., `hoge()` begins with a label `hoge:`)
* In C++, function names are _mangled_ to encode parameter types (e.g., `int hoge(int x, double y)` begins with a label `_Z4hogeid`), so that overloaded functions (same name, different parameter types) get different labels
  * the return type is _not_ encoded (except for function templates)
  * `extern "C"` before a function definition turns mangling off (`extern "C" int hoge(int x, double y)` begins with `hoge:`)
  * `c++filt` demangles a name: `echo _Z4hogeid | c++filt` prints `hoge(int, double)`; `c++filt < a.s` demangles an entire assembly file
* How to read a mangled name (Itanium C++ ABI, used by clang, gcc and nvc++ on Linux)

<div>
| Part                     | Meaning                                                                 |
|--------------------------|-------------------------------------------------------------------------|
| `_Z`                     | "this is a mangled name"                                                |
| `4hoge`                  | the function name, preceded by its length                               |
| `N2ns3barE`              | a qualified name `ns::bar` (namespace or class), enclosed by `N` ... `E` |
| `v`                      | no parameters (`void`)                                                  |
| `c` `s` `i` `l` `x`      | `char` `short` `int` `long` `long long`                                 |
| `h` `t` `j` `m` `y`      | `unsigned char` `unsigned short` `unsigned int` `unsigned long` `unsigned long long` |
| `f` `d` `b`              | `float` `double` `bool`                                                 |
| `P` `R` `K`              | pointer, reference, `const` (prefix of a type: `PKf` = `const float *`)  |
| `8record_t`              | a user-defined type, preceded by the length of its name                 |
| `Dv4_f`                  | a GNU vector type of 4 floats (`float __attribute__((vector_size(16)))`) |
| `13__Float32x4_t`        | the NEON type `float32x4_t`                                             |
| `u13__SVFloat32_t`       | the SVE type `svfloat32_t`                                              |
</div>

* Examples you will see in the course

<div>
| Label                                | Function                                                       |
|--------------------------------------|----------------------------------------------------------------|
| `_Z6branchfPffi`                     | `branch(float, float*, float, int)`                            |
| `_Z8iter_funddlllP8record_tPl`       | `iter_fun(double, double, long, long, long, record_t*, long*)` |
| `_Z1ffff`                            | `f(float, float, float)`                                       |
| `_ZGVnN4uvu__Z1ffff`                 | a vector version of `f` generated for `#pragma omp declare simd` (see below) |
</div>

* `_ZGV` names are vector versions of a function (the AArch64 vector function ABI): `_ZGVnN4uvu_` + the scalar name means "NEON (`n`), not masked (`N`), 4 lanes, and the 1st/2nd/3rd parameters are uniform/vector/uniform (`uvu`)"
* Labels starting with `.L` (e.g., `.LBB0_4`) are local labels inside a function (see [Things in Compiler Output That Are Not Instructions](#things-in-compiler-output-that-are-not-instructions))

# Registers

## Overview

<div>
| Name                         | Width                  | What                                                       | Count      |
|------------------------------|------------------------|------------------------------------------------------------|------------|
| `x0`–`x30`                   | 64 bit                 | general purpose (integers, pointers)                       | 31         |
| `w0`–`w30`                   | 32 bit                 | lower half of `xN`                                         | (alias)    |
| `sp`                         | 64 bit                 | stack pointer                                              | 1          |
| `xzr` / `wzr`                | 64 / 32 bit            | zero register (reads as 0, writes are discarded)           | 1          |
| `v0`–`v31`                   | 128 bit                | FP / NEON (SIMD) registers                                 | 32         |
| `q`/`d`/`s`/`h`/`b` `0`–`31` | 128/64/32/16/8 bit     | lower part of `vN` used as a scalar                        | (alias)    |
| `z0`–`z31`                   | VL bit (128 on Grace)  | SVE vector registers; lower 128 bits = `vN`                | 32 (alias) |
| `p0`–`p15`                   | VL/8 bit (16 on Grace) | SVE predicate (mask) registers, 1 bit per byte of a vector | 16         |
| `nzcv`                       | 4 bit                  | condition flags set by `cmp`, `subs`, `whilelt`, ...       | 1          |
</div>

## How the FP/SIMD/SVE registers alias each other

* `bN`, `hN`, `sN`, `dN`, `qN`, `vN` and `zN` are all names for (parts of) **the same physical register N**

```
bit:  VL-1 ...  128 127          64 63      32 31  16 15  8 7   0
     +-------------+---------------+----------+------+-----+-----+
 zN  |  (SVE only) |                                             |   VL bits (= 128 on Grace)
     +-------------+---------------+----------+------+-----+-----+
 vN/qN             |                                             |   128 bits
                   +---------------+----------+------+-----+-----+
 dN                                |                             |   64 bits  (double)
                                   +----------+------+-----+-----+
 sN                                           |                  |   32 bits  (float)
                                              +------+-----+-----+
 hN                                                  |           |   16 bits  (half)
                                                     +-----+-----+
 bN                                                        |     |    8 bits
                                                           +-----+
```

* scalar floating point arithmetic uses `sN` (float) / `dN` (double), i.e., the lowest element of `vN`
  * so there are no separate "floating point registers"; `s0` _is_ the lowest 32 bits of `v0`
* writing to a narrower name (e.g., `s0`, or `v0` in an SVE machine) **zeroes the upper bits** of the register
* the same goes for general purpose registers: `w0` is the lower half of `x0`; writing `w0` zeroes the upper 32 bits of `x0`
* on Grace VL = 128, so `zN` and `vN` are exactly the same 128 bits; on a machine with longer vectors (e.g., A64FX, VL = 512) `zN` is longer than `vN`

## Vector arrangements (how a register is viewed as a vector)

* NEON: `vN.<count><size>` — the count is part of the name, because the length (128 or 64 bits) is fixed

<div>
| Suffix                                | Elements           | Typical C type      |
|---------------------------------------|--------------------|---------------------|
| `v0.16b`                              | 16 × 8 bit         | `char`              |
| `v0.8h`                               | 8 × 16 bit         | `short`, `_Float16` |
| `v0.4s`                               | 4 × 32 bit         | `float`, `int`      |
| `v0.2d`                               | 2 × 64 bit         | `double`, `long`    |
| `v0.8b` / `v0.4h` / `v0.2s` / `v0.1d` | lower 64 bits only |                     |
</div>

* `b` = byte (8), `h` = halfword (16), `s` = single word (32), `d` = doubleword (64), `q` = quadword (128)
* note that "s" and "d" do _not_ mean single/double precision here; `v0.4s` can hold 4 floats or 4 ints
* `mov v0.16b, v1.16b` copies all 128 bits; it is written with `.16b` just because a copy does not care about element boundaries
* a single element: `v0.s[1]` = element 1 of `v0` viewed as 32-bit elements (elements are numbered from the lowest bits)
* SVE: `zN.<size>` — **no count**, because the number of elements depends on VL (VL/32 for `.s`; 4 on Grace)

<div>
| Suffix           | Elements (Grace)                                            | Elements (general) |
|------------------|-------------------------------------------------------------|--------------------|
| `z0.b`           | 16 × 8 bit                                                  | VL/8               |
| `z0.h`           | 8 × 16 bit                                                  | VL/16              |
| `z0.s`           | 4 × 32 bit                                                  | VL/32              |
| `z0.d`           | 2 × 64 bit                                                  | VL/64              |
| `z0.q`           | 1 × 128 bit                                                 | VL/128             |
| `z0` (no suffix) | the whole register (e.g., `ldr z0, [x0]`, `movprfx z7, z3`) |                    |
</div>

## SVE predicate registers

* `p0`–`p15` hold one bit per _byte_ of a vector register; for 32-bit elements (`p0.s`) every 4th bit is the one that matters
  * think of `p0.s` as "a mask with one true/false per 32-bit element"
* most predicated instructions (loads, stores, arithmetic) can only use `p0`–`p7` as the governing predicate; `p8`–`p15` are mostly for holding and combining masks
* the predicate is written after the destination; its _qualifier_ says what happens to inactive (false) elements

<div>
| Notation            | Meaning                                                                                |
|---------------------|----------------------------------------------------------------------------------------|
| `p0/z`              | **zeroing**: inactive elements of the result become 0 (loads always use `/z`)          |
| `p0/m`              | **merging**: inactive elements of the result keep the old value of the destination     |
| `p0` (no qualifier) | stores: only active elements are written; reductions: only active elements are reduced |
</div>

* "predicate" = "mask" in other SIMD terminology; NEON has no predicate registers

## Calling convention (AAPCS64) for these registers

<div>
| Registers                   | Role                                                  |
|-----------------------------|-------------------------------------------------------|
| `x0`–`x7`                   | integer/pointer arguments and return value            |
| `v0`–`v7` (`s0`, `d0`, ...) | floating point / vector arguments and return value    |
| `x19`–`x28`                 | callee-saved                                          |
| `v8`–`v15`                  | callee-saved, **only the lower 64 bits** (`d8`–`d15`) |
| `x9`–`x15`, `v16`–`v31`     | temporaries (caller-saved)                            |
</div>

* e.g., `void axpb(float a, float * x, float b, int n)` receives `a` in `s0`, `x` in `x0`, `b` in `s1`, `n` in `w1` (integer and FP arguments are numbered independently; a 32-bit `int` uses the `w` name)
* so a function that keeps a `double` live across a call saves `d8`/`d9` in its prologue (`stp d9, d8, [sp, #-32]!`) — you often see it in loops calling functions
* `.variant_pcs foo` in GCC output marks a function using a different convention (e.g., a vector function variant generated by `declare simd`)

# Reading Operands

<div>
| Syntax                       | Meaning                                                                              |
|------------------------------|--------------------------------------------------------------------------------------|
| `#16`, `#0x1`, `#3.0`        | immediate value                                                                      |
| `x1, lsl #2`                 | `x1 << 2` (used to scale an index by `sizeof(float)` = 4)                            |
| `[x0]`                       | memory at address `x0`                                                               |
| `[x0, #16]`                  | memory at `x0 + 16`                                                                  |
| `[x0, x8, lsl #2]`           | memory at `x0 + x8 * 4` (i.e., `&x[i]` for a `float` array)                          |
| `[x0, #16]!`                 | **pre-index**: `x0 += 16`, then access `[x0]`                                        |
| `[x0], #16`                  | **post-index**: access `[x0]`, then `x0 += 16`                                       |
| `[x0, #1, mul vl]`           | (SVE) memory at `x0 + 1 × (vector length in bytes)` — the offset counts in _vectors_ |
| `[x0, z1.s, sxtw #2]`        | (SVE gather/scatter) a vector of addresses `x0 + sign_extend(z1.s[j]) * 4`           |
| `[x0, z1.d, lsl #2]`         | (SVE gather/scatter) a vector of addresses `x0 + z1.d[j] * 4`                        |
| `{ v0.4s, v1.4s }`           | a list of registers (for `ld2`, `st2`, ...); `{ z7.s }` is a list of one register    |
| `v0.s[0]`                    | element 0 of `v0` (32-bit elements)                                                  |
| `p0/z`, `p0/m`               | governing predicate (zeroing / merging), see above                                   |
| `sxtw` / `uxtw`              | sign- / zero-extend a 32-bit word to 64 bits                                         |
| `adrp x8, sym` + `:lo12:sym` | address of a global symbol: 4KB page address + offset within the page                |
</div>

# Scalar Integer Instructions

```asm
mov   x0, x1              // x0 = x1          (also: mov x0, #5; mov x0, xzr)
movk  w8, #16285, lsl #16 // insert a 16-bit immediate into bits 16-31 (build large constants)
add   x0, x1, x2          // x0 = x1 + x2
add   x0, x1, x2, lsl #2  // x0 = x1 + (x2 << 2)
sub   x0, x1, #4          // x0 = x1 - 4
neg   x0, x1              // x0 = -x1
mul   x0, x1, x2          // x0 = x1 * x2
madd  x0, x1, x2, x3      // x0 = x3 + x1 * x2
msub  x0, x1, x2, x3      // x0 = x3 - x1 * x2   (used to compute a % b = a - (a/b)*b)
sdiv  x0, x1, x2          // signed divide   (udiv: unsigned)
and   x0, x1, #0x7ffffffffffffff0  // bitwise AND (also orr, eor, bic = and not)
lsl   x0, x1, #4          // shift left   (lsr: logical right, asr: arithmetic right)
bfi   x8, x0, #2, #3      // take the lowest 3 bits of x0 and insert them into bits 2,3,4 of x8 (i.e., x8[2,3,4] = x0[0,1,2]); other bits of x8 unchanged
csel  x0, x1, x2, lt      // x0 = (flags say "lt") ? x1 : x2
csinc x0, x1, x2, ne      // x0 = (ne) ? x1 : x2 + 1
adrp  x8, .LCPI0_0        // address of the 4KB page containing a symbol
mrs   x8, TPIDR_EL0       // read a system register (here: thread pointer, for thread-local variables)
```

* `sub` and `add` with `s` (`subs`, `adds`) also set the condition flags; loop counters are often decremented by `subs x8, x8, #16` followed by `b.ne`

# Compare and Branch

```asm
cmp   x0, x1              // set flags by x0 - x1 (result discarded)
cmn   x0, #1              // set flags by x0 + 1  ("compare negative")
ccmp  x0, x1, #0, ne      // conditional compare: if ne then cmp x0, x1 else flags = #0
b     label               // unconditional branch
b.ne  label               // branch if the condition holds (see the table below)
cbz   x0, label           // branch if x0 == 0     (cbnz: if x0 != 0) — no flags involved
tbz   w9, #0, label       // branch if bit 0 of w9 is 0   (tbnz: if it is 1)
bl    func                // call: x30 = return address; jump to func
ret                       // return: jump to x30
```

<div>
| Condition | After `cmp a, b` | After SVE `whilelt` / `ptest` (alias) |
|-----------|-------------------|----------------------------------|
| `eq` / `ne` | a == b / a != b | `none` / `any` (no / some element active) |
| `lt` / `le` / `gt` / `ge` | signed <, ≤, >, ≥ | |
| `lo` (= `cc`) / `ls` / `hi` / `hs` (= `cs`) | unsigned <, ≤, >, ≥ | `last` / ... / ... / `nlast` |
| `mi` / `pl` | negative / non-negative | `first` / `nfrst` (first element active / inactive) |
</div>

* GCC writes conditional branches without the dot (`bne`, `ble`, `bls`, ...); they are the same as `b.ne`, `b.le`, `b.ls`

# Scalar Floating Point Instructions

```asm
fmov   s0, s1             // copy (also fmov s0, #1.0; fmov x9, d0 copies the bits to an integer register)
fadd   s0, s1, s2         // float add   (d0, d1, d2 for double); also fsub, fmul, fdiv
fmadd  s0, s1, s2, s3     // s0 = s3 + s1 * s2   (FMA; note the order: the addend comes LAST)
fmsub  s0, s1, s2, s3     // s0 = s3 - s1 * s2
fnmadd s0, s1, s2, s3     // s0 = -(s3 + s1 * s2)
fsqrt  s0, s1             // square root
fabs   s0, s1             // also fneg, fmax, fmin, fmaxnm (max ignoring NaN)
fcmp   s0, s1             // compare and set flags (then b.gt etc.)
fcsel  s0, s1, s2, gt     // s0 = (gt) ? s1 : s2
scvtf  s0, w0             // signed int -> float   (ucvtf: unsigned)
fcvtzs w0, s0             // float -> signed int, rounding toward zero
fcvt   d0, s0             // float -> double
ldr    s0, [x0]           // load a float  (str s0, [x0]: store)
```

# NEON (Advanced SIMD) Instructions

* NEON instructions are mostly **the scalar mnemonic applied to `vN.<arrangement>`**; they operate element-wise
  * `fadd s0, s1, s2` (1 float) vs. `fadd v0.4s, v1.4s, v2.4s` (4 floats)
* NEON has no predicates; every instruction processes all elements (all 128 or 64 bits)
* NEON has no gather/scatter and no vector integer division

## Arithmetic

```asm
fadd  v0.4s, v1.4s, v2.4s     // v0[j] = v1[j] + v2[j]  for j = 0..3 (also fsub, fmul, fdiv)
fmla  v0.4s, v1.4s, v2.4s     // v0[j] += v1[j] * v2[j]  (FMA; the destination is also the addend)
fmla  v0.4s, v1.4s, v2.s[0]   // v0[j] += v1[j] * v2[0]  ("by element": multiply by one element, e.g., a scalar a)
fmls  v0.4s, v1.4s, v2.4s     // v0[j] -= v1[j] * v2[j]
fsqrt v0.4s, v1.4s            // also fabs, fneg, fmax, fmin, fmaxnm, frintz (round toward 0), ...
add   v0.4s, v1.4s, v2.4s     // integer add (also sub, mul, mla, and, orr, eor, bic)
shl   v0.2d, v1.2d, #2        // shift each element left (ushr / sshr: right)
```

* because `fmla` overwrites the addend, computing `a * x + b` with `b` in a register needs a copy first; you will see pairs like

```asm
mov   v7.16b, v2.16b          // v7 = b (all 128 bits)
fmla  v7.4s, v3.4s, v0.s[0]   // v7 += x * a
```

* the `mov` is not free: it apparently occupies the same SIMD execution units as `fmla` (measured), so a loop of such pairs runs at most 2 `fmla`s per cycle instead of 4; SVE's `fmad` (which overwrites the multiplicand) needs no copy (see [NEON or SVE?](#neon-or-sve))

## Moving data around

```asm
mov   v0.16b, v1.16b          // copy the whole register (an alias of orr v0.16b, v1.16b, v1.16b)
dup   v0.4s, v1.s[0]          // broadcast element 0 of v1 to all 4 elements
dup   v0.4s, w0               // broadcast a general purpose register
movi  v0.2d, #0               // v0 = 0
movi  v1.4s, #128, lsl #24    // every element = 0x80000000 (the sign bit; used to negate floats with eor)
fmov  v1.4s, #3.0             // every element = 3.0
mov   v0.s[1], w0             // insert w0 into element 1 (an alias of ins)
mov   v0.s[1], v2.s[0]        // copy one element into another
umov  w0, v0.s[1]             // extract element 1 into a general purpose register (also fmov s0 / mov s0, v0.s[1])
```

## Load and store

```asm
ldr   q0, [x0]                // load 128 bits (4 floats)       str q0, [x0]: store
ldp   q0, q1, [x0, #32]       // load 2 × 128 bits from x0+32   stp q0, q1, [x0]: store
ld1   { v0.4s }, [x0]         // load 4 floats (same as ldr q0 for our purpose)
ld1   { v0.s }[1], [x0]       // load ONE float into element 1 (others unchanged)
st1   { v0.s }[1], [x0]       // store element 1 only (used for strided stores)
ld1r  { v0.4s }, [x0]         // load one float and broadcast it
ld2   { v0.4s, v1.4s }, [x0]  // load 8 floats and de-interleave: v0 = even elements, v1 = odd elements
ld3 / ld4                     // the same for 3 / 4 interleaved components (e.g., struct {float x, y, z;})
st2 / st3 / st4               // store with interleaving
```

* `ld2` is how the compiler vectorizes a stride-2 access `x[2*i]` (it uses `v0` and discards `v1`)

## Compare and select (branches inside vectorized loops)

```asm
fcmgt v0.4s, v1.4s, v2.4s     // v0[j] = (v1[j] > v2[j]) ? 0xffffffff : 0   (also fcmge, fcmeq, fcmlt #0.0, ...)
cmgt  v0.4s, v1.4s, v2.4s     // the same for signed integers (cmhi: unsigned)
bsl   v0.16b, v1.16b, v2.16b  // bitwise select: v0 = (v0 & v1) | (~v0 & v2), v0 is the mask
bit / bif                     // the same with different operand roles (insert if true / if false)
```

* the result of a comparison is a vector of all-ones / all-zeros elements, not flags; it is used as a mask for `bsl`

## Horizontal operations (within a vector) and permutations

```asm
faddp v0.4s, v1.4s, v2.4s     // pairwise add: v0 = {v1[0]+v1[1], v1[2]+v1[3], v2[0]+v2[1], v2[2]+v2[3]}
faddp s0, v0.2s               // s0 = v0[0] + v0[1]
addv  s0, v0.4s               // s0 = sum of 4 integer elements (also fmaxv, fminv, fmaxnmv, umaxv, ...)
zip1 / zip2                   // interleave the lower / upper halves of two vectors
uzp1 / uzp2                   // take even / odd elements of two vectors (the inverse of zip)
ext   v0.16b, v1.16b, v2.16b, #4   // concatenate and extract 16 bytes starting at byte 4
tbl   v0.16b, { v1.16b }, v2.16b   // arbitrary byte shuffle by an index vector
rev64 v0.4s, v1.4s            // reverse elements within each 64-bit half
scvtf v0.4s, v1.4s            // int -> float element-wise (fcvtzs: float -> int; fcvtl/fcvtn: widen/narrow)
```

# SVE Instructions

## Key ideas

* **vector length agnostic (VLA)**: the same machine code works for any VL (128 to 2048 bits); the code never assumes "4 floats per vector"; instead it asks the hardware (`cntw`, `incw`, `rdvl`) or uses `mul vl` offsets
* **predication**: almost every instruction takes a predicate, so the last partial vector of a loop needs no special remainder loop
* **destructive encoding**: most arithmetic instructions overwrite their first source (`fmad z7.s, p0/m, z6.s, z2.s` updates `z7`); `movprfx` is used to make a non-destructive version (see below)
* on Grace (VL = 128) an SVE vector holds the same 4 floats as a NEON vector; clang uses NEON for simple loops and SVE when it needs predicates, gathers, etc. (see [NEON or SVE?](#neon-or-sve))

## Vector length, counting and loop control

```asm
cntw   x8                     // x8 = number of 32-bit elements in a vector (4 on Grace)
                              //   cntb / cnth / cntw / cntd: number of 8/16/32/64-bit elements
incw   x8                     // x8 += cntw        (decw: -=) — "i += (elements per vector)"
incb   x10, all, mul #4       // x10 += 4 × cntb   (move a pointer by 4 vectors)
decb   x11                    // x11 -= cntb
incd   z4.d                   // every element of z4 += cntd  (vector form)
rdvl   x9, #1                 // x9 = VL in bytes (16 on Grace)  (rdvl x9, #-1 gives -VL)
addvl  sp, sp, #-2            // sp -= 2 × VL bytes (stack space for 2 vectors)
whilelt p0.s, x8, x1          // p0.s[j] = (x8 + j < x1)  for each element j; also sets flags
                              //   whilelo: unsigned version. "the elements i, i+1, ... that are still < n"
ptrue  p0.s                   // all elements true
pfalse p0.b                   // all false
ptest  p0, p1.b               // set flags from p1 (b.none / b.any / b.first afterwards)
```

* a typical VLA loop without a remainder loop (clang output for `y[i] = a * x[i] + b` written with SVE intrinsics; `p0` = all true)

```asm
.LBB0_2:
  whilelt p1.s, x8, x2                        // p1 = mask of i, i+1, ..., that are < n
  ld1w    { z2.s }, p1/z, [x0, x8, lsl #2]    // z2 = x[i:i+VL/32] (inactive elements = 0)
  fmad    z2.s, p0/m, z0.s, z1.s              // z2 = z2 * a + b (on all elements; garbage in inactive ones is harmless)
  st1w    { z2.s }, p1, [x1, x8, lsl #2]      // y[i:...] = z2 (only active elements)
  incw    x8                                  // i += VL/32
  cmp     x8, x2
  b.lt    .LBB0_2
```

* only the memory accesses need the `whilelt` mask; arithmetic on inactive elements does no harm
* compilers usually do _not_ generate this form for ordinary loops; they generate a vector loop plus a scalar remainder loop instead

## Arithmetic

```asm
fadd  z0.s, p0/m, z0.s, z1.s  // z0 = z0 + z1 on active elements (also fsub, fmul, fdiv, fmax, fmin, fmaxnm)
fadd  z0.s, z1.s, z2.s        // unpredicated form (exists for a few common ops: fadd, fsub, fmul, add, ...)
fdivr z0.s, p0/m, z0.s, z1.s  // reversed: z0 = z1 / z0  (also fsubr)
fmla  z0.s, p0/m, z1.s, z2.s  // z0 = z0 + z1 * z2   (the destination is the ADDEND)
fmad  z0.s, p0/m, z1.s, z2.s  // z0 = z0 * z1 + z2   (the destination is a MULTIPLICAND)
fmls / fmsb                   // the same with subtraction (z0 - z1*z2 / z2 - z0*z1); fnmla, fnmad: negated
fsqrt z0.s, p0/m, z1.s        // also fabs, fneg, frintz, scvtf (int->float), fcvtzs (float->int), ...
add   z3.d, z5.d, z3.d        // integer (also sub, mul, and, orr, eor, lsl, lsr)
and   z7.d, z7.d, #0x1        // with an immediate
udiv  z1.d, p0/m, z1.d, z2.d  // vector integer division (sdiv: signed); 32/64-bit only; slow
mls   z0.d, p0/m, z1.d, z2.d  // z0 = z0 - z1 * z2  (with udiv, computes a % b)
```

* `fmla` vs `fmad`: both are FMAs; the compiler picks the one whose destination is the register it wants to overwrite

## movprfx

```asm
movprfx z7, z3                // "the next instruction's destination starts as a copy of z3"
and     z7.d, z7.d, #0x1      // together: z7 = z3 & 1   (z3 is preserved)
```

* `movprfx` must be immediately followed by a destructive instruction using the same destination; the hardware may fuse the pair into a single non-destructive operation

## Moving data around

```asm
mov   z2.s, s1                // broadcast the float in s1 to all elements (an alias of dup z2.s, z1.s[0])
mov   z5.d, x10               // broadcast a general purpose register (alias of dup)
mov   z0.s, #0                // all elements = 0   (fmov z0.s, #1.0: all = 1.0)
mov   z4.d, z3.d              // copy
index z3.d, #0, #1            // z3 = {0, 1, 2, ...}: start 0, step 1 — a vector of loop indices
sel   z0.s, p0, z1.s, z2.s    // z0[j] = p0[j] ? z1[j] : z2[j]
mov   z0.s, p0/m, z1.s        // z0[j] = z1[j] where p0 is true (merge)
```

## Load and store

```asm
ld1w  { z0.s }, p0/z, [x0, x8, lsl #2]     // load VL/32 words from &x[i] (w = 32-bit word)
ld1w  { z0.s }, p0/z, [x0, #1, mul vl]     // from x0 + 1 vector (immediate in -8..7 vectors)
ld1d  { z0.d }, p0/z, [x0, x8, lsl #3]     // 64-bit elements (ld1b / ld1h / ld1w / ld1d by element size)
ld1w  { z0.d }, p0/z, [x0, x8, lsl #2]    // load 32-bit words into 64-bit elements (zero-extended)
ld1rw { z0.s }, p0/z, [x0]                 // load one word and broadcast it
ld2w  { z0.s, z1.s }, p0/z, [x0]           // de-interleaving load (like NEON ld2); also ld3w, ld4w
ldr   z0, [x0, #1, mul vl]                 // unpredicated load of a whole register
st1w  { z0.s }, p0, [x0, x8, lsl #2]       // store only the active elements
str   z0, [x0]                             // unpredicated store
```

* the name tells the element size in memory: `ld1b`/`ld1h`/`ld1w`/`ld1d` = 8/16/32/64 bits; `ld1sw` etc. sign-extend
* the `1` means "one register, contiguous"; `ld2`, `ld3`, `ld4` load 2-4 registers with de-interleaving

## Gather and scatter (SVE only)

```asm
ld1w  { z0.s }, p0/z, [x0, z1.s, sxtw #2]  // gather: z0[j] = *(float*)(x0 + sext(z1[j]) * 4)  (int indices)
ld1w  { z0.s }, p0/z, [x0, z1.s, uxtw #2]  // the same with unsigned indices (zero-extended)
ld1w  { z0.d }, p0/z, [x0, z1.d, lsl #2]   // the same with 64-bit indices -> only VL/64 (2 on Grace) elements
ld1d  { z0.d }, p0/z, [x0, z1.d, lsl #3]   // gather of doubles
ld1w  { z0.s }, p0/z, [z1.s, #0]           // addresses taken directly from z1
st1w  { z0.s }, p0, [x0, z1.s, sxtw #2]    // scatter
```

* if the index is 64-bit (e.g., `long i`, or an `int` index the compiler widened to 64 bits, as it does for an index computed from an `int` loop variable such as `x[i * i]`), a gather loads only 2 floats on Grace; the compiler then does 2 gathers and packs the results with `uzp1 z16.s, z16.s, z17.s`
  * computing such an index in `unsigned` (e.g., `x[((unsigned)i * i) & mask]`) keeps it 32-bit, giving one 4-element gather with `uxtw`
  * an index loaded from an `int` array (`x[idx[i]]`) stays 32-bit (`sxtw`)

## Compare and predicates

```asm
fcmgt p1.s, p0/z, z0.s, z1.s  // p1[j] = (z0[j] > z1[j]) for active j  (fcmge, fcmeq, fcmlt, ... also #0.0)
cmpeq p1.d, p0/z, z7.d, #0    // integer compare with an immediate (cmpgt, cmphi (unsigned >), ...)
and   p2.b, p0/z, p1.b, p3.b  // predicate logic (also orr, eor, bic, not)
uzp1  p1.s, p1.s, p2.s        // permutation also works on predicates (here: two .d masks -> one .s mask)
brka / brkb                   // "break after / before the first true": for loops with early exit
```

* a vectorized `if` typically: compare into a predicate, then compute everywhere and store with that predicate (clang output for `if (x[i] > 0.0) x[i] = a * x[i] + b;`)

```asm
ldr    z4, [x10]                   // z4 = x[i:i+4]
fcmgt  p1.s, p0/z, z4.s, #0.0      // p1 = (x[i:i+4] > 0.0)
fmad   z4.s, p0/m, z3.s, z2.s      // z4 = z4 * a + b, computed for ALL elements (harmless)
st1w   { z4.s }, p1, [x10]         // store only where the condition holds
```

* when a load in the then-part may be unsafe for elements whose condition is false (e.g., `if (c[i]) y[i] = x[idx[i]]`), the loads are predicated as well (`ld1w { z0.s }, p1/z, [x0, z0.s, sxtw #2]`)

## Reductions and permutations

```asm
faddv s0, p0, z0.s            // s0 = sum of active elements (tree order; needs -ffast-math or omp simd reduction)
fadda s0, p0, s0, z0.s        // s0 = ((s0 + z0[0]) + z0[1]) + ... strictly in order (keeps the C semantics, slow)
fmaxv / fminv / fmaxnmv       // max / min (also uaddv, smaxv, ... for integers)
uzp1  z0.s, z1.s, z2.s        // even elements of z1 then z2 (uzp2: odd); zip1/zip2: interleave
compact z0.s, p0, z1.s        // pack the active elements to the front
splice  z0.s, p0, z0.s, z1.s  // concatenate the active part of z0 with z1
tbl   z0.s, { z1.s }, z2.s    // arbitrary shuffle by an index vector
rev   z0.s, z1.s              // reverse elements
```

* `fadda` (in-order addition) appears in a sum loop compiled without permission to reorder additions; with `#pragma omp simd reduction(+:s)` or `-ffast-math` you get parallel partial sums instead

# NEON or SVE?

* on Grace both instruction sets have the same vector length (128 bits), and the compiler chooses one for each operation; which one you get depends on how you write the code

<div>
| How the code is written                                                                 | What you get                                                                   |
|-----------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| ordinary loop, simple (contiguous accesses, no `if`)                                     | NEON (`v` registers, remainder loop in scalar)                                  |
| ordinary loop that needs predicates (`if`) or gathers (`x[idx[i]]`)                      | SVE (`z`/`p` registers; `vectorization width: vscale x 4` in the report)         |
| GNU vector type (`float __attribute__((vector_size(16)))`) or NEON type (`float32x4_t`) with operators | NEON                                                                |
| SVE intrinsics (`svmad_f32_x`, `svld1`, `svwhilelt_b32`, ...)                            | SVE, except that an operation also available in NEON may be turned into NEON (e.g., `svadd_x(svptrue_b32(), ...)` became `fadd v.4s`) |
| sizeless SVE type (`svfloat32_t`) with operators (`a * v + b`)                           | SVE                                                                            |
</div>

* both are usually equally fast, but NEON `fmla` overwrites the addend, so `x = a * x + b` needs a `mov` per `fmla` and reaches only half of the peak FMA throughput, whereas SVE `fmad` does it in one instruction
* to get SVE for code written with a GNU vector type, cast to the sizeless type in the expression and compile with `-msve-vector-bits=128`

```c
typedef float floatv __attribute__((vector_size(16)));
floatv x[8];
x[c] = a * (svfloat32_t)x[c] + b;    // -> fmad z.s, p0/m, ...   (needs -msve-vector-bits=128)
```

  * `-msve-vector-bits=128` tells the compiler that the SVE vector length is 128 bits (the binary then works correctly only on such processors, like Grace); without it the cast is an error
  * the cast works only for GNU vector types (and `arm_sve_vector_bits` types), not for NEON types such as `float32x4_t`; for them, use `svset_neonq_f32(svundef_f32(), v)` / `svget_neonq_f32(z)` from `<arm_neon_sve_bridge.h>`, or cast through a GNU vector type
  * alternatively, pass the GNU vector to the SVE intrinsic directly (`x[c] = svmad_f32_x(svptrue_b32(), x[c], svdup_f32(a), svdup_f32(b))`), again with `-msve-vector-bits=128`
* the C types for vectors

<div>
| Type                                         | Header          | Size       | Notes                                                                 |
|----------------------------------------------|-----------------|------------|-----------------------------------------------------------------------|
| `float __attribute__((vector_size(16)))`     | (none)          | 16 bytes   | GNU vector type; operators, arrays, structs all OK                    |
| `float32x4_t`, `float64x2_t`, `int64x2_t`, ... | `<arm_neon.h>` | 16 (or 8) bytes | NEON types: `<type><bits>x<count>_t`                              |
| `svfloat32_t`, `svfloat64_t`, `svint64_t`, `svbool_t`, ... | `<arm_sve.h>` | unknown (sizeless) | SVE types: no count in the name; no `sizeof`, arrays, struct members or globals |
</div>

# Things in Compiler Output That Are Not Instructions

<div>
| What                                   | Meaning                                                                                                        |
|----------------------------------------|----------------------------------------------------------------------------------------------------------------|
| `.LBB0_4:`                             | a label (basic block) inside function 0; `// =>This Inner Loop Header: Depth=1` marks an innermost loop        |
| `.Lfunc_end0`, `.LFE0`                 | end of a function                                                                                              |
| `.p2align 5, , 16`                     | pad with nops so the next instruction starts at a 32-byte boundary (if it costs ≤ 16 bytes)                    |
| `.cfi_*`                               | information for debuggers / exception handling (stack layout); ignore                                          |
| `.type f,@function`, `.size`, `.globl` | symbol information; ignore                                                                                     |
| `// kill: def $s1 killed $s1 def $z1`  | a note from the register allocator ("s1 is used as part of z1 from here"); not an instruction                  |
| `//APP` ... `//NO_APP`                 | the text of an `asm volatile(...)` in the source (e.g., our `# ===== loop begins =====` markers)               |
| `_Z6branchfPffi`                       | a C++ mangled name (= `branch(float, float*, float, int)`); see [Basics](#basics)                              |
| `.LCPI0_0`                             | a constant (e.g., a vector of constants) placed in memory; loaded with `adrp` + `ldr q0, [x8, :lo12:.LCPI0_0]` |
</div>

# Performance on Grace (rough numbers)

* measured with small benchmarks on a Miyabi login node (data in L1 cache); throughput = instructions per cycle

<div>
| Instruction                               | Latency (cycles)                      | Throughput (/cycle)                       |
|-------------------------------------------|---------------------------------------|-------------------------------------------|
| `fmla`/`fmad`/`fmadd` (scalar, NEON, SVE) | 4 (2 when chained through the addend) | 4                                         |
| NEON `mov` + `fmla` pairs (`x = a * x + b`) | 4                                   | 2 (`fmla`s)                               |
| `fadd` / `fmul`                           | 2 / 3                                 | 4                                         |
| `fdiv` / `fsqrt` (vector)                 | ~10                                   | ~0.2                                      |
| `ldr x` / `ldr s` / `ldr q`               | 4 / ~6 / ~6                           | ~3                                        |
| SVE `ld1w` (contiguous)                   | ~6                                    | ~2                                        |
| `ldp q, q` / `ld2` / `ld4`                |                                       | 1.5 / ~1.2 / 0.2                          |
| SVE gather, 4 × 32 bit                    | ~9                                    | 0.5–0.67 (= 2–2.7 floats/cycle)           |
| SVE gather, 2 × 64 bit                    | ~9                                    | ~0.9                                      |
| stores (`str q`, `st1w`)                  |                                       | up to 2, typically ~0.7–1.3               |
| `udiv z.d` / `udiv z.s`                   | 20 / 12                               | 0.05 / 0.09 (from the optimization guide) |
</div>

* a loop is limited by whichever resource saturates first: e.g., `y[i] = a * x[i] + b` needs 1 FMA, 1 load and 1 store per element; with 4 FMAs, ~3 loads and ~1–2 stores per cycle, the loads/stores are the bottleneck, not the FMAs
* unaligned loads/stores are allowed; they cost nothing within a 64-byte cache line, loads crossing a line count twice, and a store crossing a 4KB page is very slow (~11 cycles)
* the FMA throughput of 4 per cycle is reached only sometimes: a loop needing more than 3 independent FMAs per cycle runs at either ~4 or exactly 2 FMAs per cycle, chosen (apparently at random) each time the loop is entered; we do not know why
