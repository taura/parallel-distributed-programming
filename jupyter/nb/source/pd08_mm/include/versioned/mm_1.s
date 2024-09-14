	.file	"mm_1.cc"
	.text
	.p2align 4
	.globl	_Z4gemm6matrixS_S_
	.type	_Z4gemm6matrixS_S_, @function
_Z4gemm6matrixS_S_:
.LFB5691:
	.cfi_startproc
	endbr64
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
	movq	80(%rsp), %r10
	movq	88(%rsp), %r9
	movq	24(%rsp), %rbx
	testq	%r10, %r10
	jle	.L12
	xorl	%r8d, %r8d
	vxorps	%xmm1, %xmm1, %xmm1
	testq	%r9, %r9
	jle	.L12
	.p2align 4,,10
	.p2align 3
.L3:
	xorl	%edi, %edi
	.p2align 4,,10
	.p2align 3
.L6:
#APP
# 11 "versioned/mm_1.cc" 1
	# loop begins
# 0 "" 2
#NO_APP
	testq	%rbx, %rbx
	jle	.L8
	movq	32(%rsp), %rsi
	movq	40(%rsp), %r11
	vmovaps	%xmm1, %xmm0
	movq	64(%rsp), %rcx
	movq	72(%rsp), %rdx
	imulq	%r8, %rsi
	salq	$2, %rcx
	leaq	(%rdx,%rdi,4), %rdx
	leaq	(%r11,%rsi,4), %rax
	addq	%rbx, %rsi
	leaq	(%r11,%rsi,4), %rsi
	.p2align 4,,10
	.p2align 3
.L5:
	vmovss	(%rax), %xmm2
	addq	$4, %rax
	vfmadd231ss	(%rdx), %xmm2, %xmm0
	addq	%rcx, %rdx
	cmpq	%rsi, %rax
	jne	.L5
.L4:
#APP
# 15 "versioned/mm_1.cc" 1
	# loop ends
# 0 "" 2
#NO_APP
	movq	96(%rsp), %rax
	movq	104(%rsp), %rdx
	imulq	%r8, %rax
	addq	%rdi, %rax
	addq	$1, %rdi
	leaq	(%rdx,%rax,4), %rax
	vaddss	(%rax), %xmm0, %xmm0
	vmovss	%xmm0, (%rax)
	cmpq	%rdi, %r9
	jne	.L6
	addq	$1, %r8
	cmpq	%r8, %r10
	jne	.L3
.L12:
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L8:
	.cfi_restore_state
	vmovaps	%xmm1, %xmm0
	jmp	.L4
	.cfi_endproc
.LFE5691:
	.size	_Z4gemm6matrixS_S_, .-_Z4gemm6matrixS_S_
	.ident	"GCC: (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
