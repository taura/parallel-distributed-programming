//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-31442593
// Cuda compilation tools, release 11.7, V11.7.99
// Based on NVVM 7.0.1
//

.version 7.7
.target sm_80
.address_size 64


.entry _Z14get_clock_cudaPx(
	.param .u64 _Z14get_clock_cudaPx_param_0
)
{
	.reg .b64 	%rd<4>;


	ld.param.u64 	%rd2, [_Z14get_clock_cudaPx_param_0];
	cvta.to.global.u64 	%rd3, %rd2;
	// begin inline asm
	mov.u64 	%rd1, %clock64;
	// end inline asm
	st.global.u64 	[%rd3], %rd1;
	ret;

}
	// .globl	_Z9gemm_cuda6matrixS_S_
.visible .entry _Z9gemm_cuda6matrixS_S_(
	.param .align 8 .b8 _Z9gemm_cuda6matrixS_S__param_0[32],
	.param .align 8 .b8 _Z9gemm_cuda6matrixS_S__param_1[32],
	.param .align 8 .b8 _Z9gemm_cuda6matrixS_S__param_2[32]
)
{
	.reg .pred 	%p<16>;
	.reg .f32 	%f<49>;
	.reg .b32 	%r<67>;
	.reg .b64 	%rd<44>;


	ld.param.v2.u32 	{%r40, %r41}, [_Z9gemm_cuda6matrixS_S__param_0];
	ld.param.v2.u32 	{%r44, %r45}, [_Z9gemm_cuda6matrixS_S__param_2];
	ld.param.u64 	%rd23, [_Z9gemm_cuda6matrixS_S__param_2+24];
	ld.param.u32 	%r39, [_Z9gemm_cuda6matrixS_S__param_2+8];
	ld.param.u64 	%rd21, [_Z9gemm_cuda6matrixS_S__param_1+24];
	ld.param.u32 	%r36, [_Z9gemm_cuda6matrixS_S__param_1+8];
	ld.param.u64 	%rd19, [_Z9gemm_cuda6matrixS_S__param_0+24];
	ld.param.u32 	%r33, [_Z9gemm_cuda6matrixS_S__param_0+8];
	cvta.to.global.u64 	%rd1, %rd19;
	cvta.to.global.u64 	%rd2, %rd21;
	cvta.to.global.u64 	%rd3, %rd23;
	setp.lt.s32 	%p1, %r44, 1;
	@%p1 bra 	$L__BB1_21;

	setp.lt.s32 	%p2, %r45, 1;
	@%p2 bra 	$L__BB1_21;

	add.s32 	%r5, %r45, -1;
	add.s32 	%r6, %r41, -1;
	and.b32  	%r7, %r41, 3;
	sub.s32 	%r8, %r41, %r7;
	and.b32  	%r9, %r45, 3;
	sub.s32 	%r10, %r45, %r9;
	shl.b32 	%r47, %r36, 2;
	mul.wide.s32 	%rd4, %r47, 4;
	add.s64 	%rd5, %rd1, 8;
	mul.wide.s32 	%rd6, %r36, 4;
	mov.u32 	%r59, 0;
	setp.gt.s32 	%p3, %r41, 0;
	setp.lt.u32 	%p9, %r6, 3;
	mov.f32 	%f24, 0f00000000;
	setp.eq.s32 	%p11, %r7, 0;
	setp.eq.s32 	%p12, %r7, 1;
	bra.uni 	$L__BB1_3;

$L__BB1_4:
	setp.lt.u32 	%p4, %r5, 3;
	mov.u32 	%r62, 0;
	@%p4 bra 	$L__BB1_7;

	mov.u32 	%r61, %r10;

$L__BB1_6:
	add.s32 	%r50, %r62, %r14;
	mul.wide.s32 	%rd24, %r50, 4;
	add.s64 	%rd25, %rd3, %rd24;
	ld.global.f32 	%f9, [%rd25];
	add.f32 	%f10, %f9, 0f00000000;
	st.global.f32 	[%rd25], %f10;
	ld.global.f32 	%f11, [%rd25+4];
	add.f32 	%f12, %f11, 0f00000000;
	st.global.f32 	[%rd25+4], %f12;
	ld.global.f32 	%f13, [%rd25+8];
	add.f32 	%f14, %f13, 0f00000000;
	st.global.f32 	[%rd25+8], %f14;
	ld.global.f32 	%f15, [%rd25+12];
	add.f32 	%f16, %f15, 0f00000000;
	st.global.f32 	[%rd25+12], %f16;
	add.s32 	%r62, %r62, 4;
	add.s32 	%r61, %r61, -4;
	setp.ne.s32 	%p5, %r61, 0;
	@%p5 bra 	$L__BB1_6;

$L__BB1_7:
	setp.eq.s32 	%p6, %r9, 0;
	@%p6 bra 	$L__BB1_20;

	setp.eq.s32 	%p7, %r9, 1;
	add.s32 	%r51, %r62, %r14;
	mul.wide.s32 	%rd26, %r51, 4;
	add.s64 	%rd7, %rd3, %rd26;
	ld.global.f32 	%f17, [%rd7];
	add.f32 	%f18, %f17, 0f00000000;
	st.global.f32 	[%rd7], %f18;
	@%p7 bra 	$L__BB1_20;

	setp.eq.s32 	%p8, %r9, 2;
	ld.global.f32 	%f19, [%rd7+4];
	add.f32 	%f20, %f19, 0f00000000;
	st.global.f32 	[%rd7+4], %f20;
	@%p8 bra 	$L__BB1_20;

	ld.global.f32 	%f21, [%rd7+8];
	add.f32 	%f22, %f21, 0f00000000;
	st.global.f32 	[%rd7+8], %f22;
	bra.uni 	$L__BB1_20;

$L__BB1_3:
	mul.lo.s32 	%r14, %r59, %r39;
	mul.lo.s32 	%r15, %r59, %r33;
	@%p3 bra 	$L__BB1_11;
	bra.uni 	$L__BB1_4;

$L__BB1_11:
	mul.wide.s32 	%rd27, %r15, 4;
	add.s64 	%rd8, %rd5, %rd27;
	mov.u32 	%r52, 0;
	mov.u32 	%r63, %r52;

$L__BB1_12:
	mov.u32 	%r66, %r52;
	mov.f32 	%f48, %f24;
	@%p9 bra 	$L__BB1_15;

	add.s32 	%r55, %r36, %r63;
	mul.wide.s32 	%rd28, %r55, 4;
	add.s64 	%rd43, %rd2, %rd28;
	mul.wide.s32 	%rd29, %r63, 4;
	add.s64 	%rd42, %rd2, %rd29;
	mov.u64 	%rd41, %rd8;
	mov.u32 	%r66, %r52;
	mov.f32 	%f48, %f24;
	mov.u32 	%r65, %r8;

$L__BB1_14:
	ld.global.f32 	%f26, [%rd42];
	ld.global.f32 	%f27, [%rd41+-8];
	fma.rn.f32 	%f28, %f27, %f26, %f48;
	ld.global.f32 	%f29, [%rd43];
	ld.global.f32 	%f30, [%rd41+-4];
	fma.rn.f32 	%f31, %f30, %f29, %f28;
	add.s64 	%rd30, %rd43, %rd6;
	ld.global.f32 	%f32, [%rd30];
	ld.global.f32 	%f33, [%rd41];
	fma.rn.f32 	%f34, %f33, %f32, %f31;
	add.s64 	%rd31, %rd30, %rd6;
	ld.global.f32 	%f35, [%rd31];
	ld.global.f32 	%f36, [%rd41+4];
	fma.rn.f32 	%f48, %f36, %f35, %f34;
	add.s32 	%r66, %r66, 4;
	add.s64 	%rd43, %rd43, %rd4;
	add.s64 	%rd42, %rd42, %rd4;
	add.s64 	%rd41, %rd41, 16;
	add.s32 	%r65, %r65, -4;
	setp.ne.s32 	%p10, %r65, 0;
	@%p10 bra 	$L__BB1_14;

$L__BB1_15:
	@%p11 bra 	$L__BB1_19;

	add.s32 	%r56, %r66, %r15;
	mul.wide.s32 	%rd32, %r56, 4;
	add.s64 	%rd17, %rd1, %rd32;
	mad.lo.s32 	%r27, %r66, %r36, %r63;
	mul.wide.s32 	%rd33, %r27, 4;
	add.s64 	%rd34, %rd2, %rd33;
	ld.global.f32 	%f37, [%rd34];
	ld.global.f32 	%f38, [%rd17];
	fma.rn.f32 	%f48, %f38, %f37, %f48;
	@%p12 bra 	$L__BB1_19;

	setp.eq.s32 	%p13, %r7, 2;
	add.s32 	%r28, %r27, %r36;
	mul.wide.s32 	%rd35, %r28, 4;
	add.s64 	%rd36, %rd2, %rd35;
	ld.global.f32 	%f39, [%rd36];
	ld.global.f32 	%f40, [%rd17+4];
	fma.rn.f32 	%f48, %f40, %f39, %f48;
	@%p13 bra 	$L__BB1_19;

	add.s32 	%r57, %r28, %r36;
	mul.wide.s32 	%rd37, %r57, 4;
	add.s64 	%rd38, %rd2, %rd37;
	ld.global.f32 	%f41, [%rd38];
	ld.global.f32 	%f42, [%rd17+8];
	fma.rn.f32 	%f48, %f42, %f41, %f48;

$L__BB1_19:
	add.s32 	%r58, %r63, %r14;
	mul.wide.s32 	%rd39, %r58, 4;
	add.s64 	%rd40, %rd3, %rd39;
	ld.global.f32 	%f43, [%rd40];
	add.f32 	%f44, %f48, %f43;
	st.global.f32 	[%rd40], %f44;
	add.s32 	%r63, %r63, 1;
	setp.lt.s32 	%p14, %r63, %r45;
	@%p14 bra 	$L__BB1_12;

$L__BB1_20:
	add.s32 	%r59, %r59, 1;
	setp.lt.s32 	%p15, %r59, %r44;
	@%p15 bra 	$L__BB1_3;

$L__BB1_21:
	ret;

}

