	.text
	.file	"axpb.c"
	.globl	axpb                            # -- Begin function axpb
	.p2align	4, 0x90
	.type	axpb,@function
axpb:                                   # @axpb
	.cfi_startproc
# %bb.0:
	testq	%rsi, %rsi
	jle	.LBB0_13
# %bb.1:
	cmpq	$8, %rsi
	jae	.LBB0_3
# %bb.2:
	xorl	%eax, %eax
	jmp	.LBB0_12
.LBB0_3:
	cmpq	$64, %rsi
	jae	.LBB0_5
# %bb.4:
	xorl	%eax, %eax
	jmp	.LBB0_9
.LBB0_5:
	movq	%rsi, %rax
	andq	$-64, %rax
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm3
	xorl	%ecx, %ecx
	.p2align	4, 0x90
.LBB0_6:                                # =>This Inner Loop Header: Depth=1
	vmovups	(%rdi,%rcx,4), %zmm4
	vfmadd213ps	%zmm3, %zmm2, %zmm4     # zmm4 = (zmm2 * zmm4) + zmm3
	vmovups	64(%rdi,%rcx,4), %zmm5
	vfmadd213ps	%zmm3, %zmm2, %zmm5     # zmm5 = (zmm2 * zmm5) + zmm3
	vmovups	128(%rdi,%rcx,4), %zmm6
	vfmadd213ps	%zmm3, %zmm2, %zmm6     # zmm6 = (zmm2 * zmm6) + zmm3
	vmovups	192(%rdi,%rcx,4), %zmm7
	vfmadd213ps	%zmm3, %zmm2, %zmm7     # zmm7 = (zmm2 * zmm7) + zmm3
	vmovups	%zmm4, (%rdi,%rcx,4)
	vmovups	%zmm5, 64(%rdi,%rcx,4)
	vmovups	%zmm6, 128(%rdi,%rcx,4)
	vmovups	%zmm7, 192(%rdi,%rcx,4)
	addq	$64, %rcx
	cmpq	%rcx, %rax
	jne	.LBB0_6
# %bb.7:
	cmpq	%rsi, %rax
	je	.LBB0_13
# %bb.8:
	testb	$56, %sil
	je	.LBB0_12
.LBB0_9:
	movq	%rax, %rcx
	movq	%rsi, %rax
	andq	$-8, %rax
	vbroadcastss	%xmm0, %ymm2
	vbroadcastss	%xmm1, %ymm3
	.p2align	4, 0x90
.LBB0_10:                               # =>This Inner Loop Header: Depth=1
	vmovups	(%rdi,%rcx,4), %ymm4
	vfmadd213ps	%ymm3, %ymm2, %ymm4     # ymm4 = (ymm2 * ymm4) + ymm3
	vmovups	%ymm4, (%rdi,%rcx,4)
	addq	$8, %rcx
	cmpq	%rcx, %rax
	jne	.LBB0_10
# %bb.11:
	cmpq	%rsi, %rax
	je	.LBB0_13
	.p2align	4, 0x90
.LBB0_12:                               # =>This Inner Loop Header: Depth=1
	vmovss	(%rdi,%rax,4), %xmm2            # xmm2 = mem[0],zero,zero,zero
	vfmadd213ss	%xmm1, %xmm0, %xmm2     # xmm2 = (xmm0 * xmm2) + xmm1
	vmovss	%xmm2, (%rdi,%rax,4)
	incq	%rax
	cmpq	%rax, %rsi
	jne	.LBB0_12
.LBB0_13:
	vzeroupper
	retq
.Lfunc_end0:
	.size	axpb, .Lfunc_end0-axpb
	.cfi_endproc
                                        # -- End function
	.ident	"clang version 15.0.0"
	.section	".note.GNU-stack","",@progbits
	.addrsig
