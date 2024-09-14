	.file	"mm_2.cc"
	.text
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC0:
	.string	"void gemm(matrix, matrix, matrix)"
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC1:
	.string	"versioned/mm_2.cc"
.LC2:
	.string	"N % L == 0"
	.text
	.p2align 4
	.globl	_Z4gemm6matrixS_S_
	.type	_Z4gemm6matrixS_S_, @function
_Z4gemm6matrixS_S_:
.LFB5691:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	movq	88(%rbp), %r10
	movq	80(%rbp), %r11
	.cfi_offset 3, -24
	movq	24(%rbp), %rbx
	andq	$-64, %rsp
	testb	$15, %r10b
	jne	.L2
	testq	%r11, %r11
	jle	.L14
	xorl	%r9d, %r9d
	testq	%r10, %r10
	jle	.L14
	.p2align 4,,10
	.p2align 3
.L4:
	xorl	%edi, %edi
	.p2align 4,,10
	.p2align 3
.L7:
#APP
# 12 "versioned/mm_2.cc" 1
	# loop begins
# 0 "" 2
#NO_APP
	testq	%rbx, %rbx
	jle	.L9
	movq	32(%rbp), %rsi
	movq	40(%rbp), %r8
	vxorps	%xmm0, %xmm0, %xmm0
	movq	64(%rbp), %rcx
	movq	72(%rbp), %rdx
	imulq	%r9, %rsi
	salq	$2, %rcx
	leaq	(%rdx,%rdi,4), %rdx
	leaq	(%r8,%rsi,4), %rax
	addq	%rbx, %rsi
	leaq	(%r8,%rsi,4), %rsi
	.p2align 4,,10
	.p2align 3
.L6:
	vbroadcastss	(%rax), %zmm1
	addq	$4, %rax
	vfmadd231ps	(%rdx), %zmm1, %zmm0
	addq	%rcx, %rdx
	cmpq	%rsi, %rax
	jne	.L6
.L5:
#APP
# 16 "versioned/mm_2.cc" 1
	# loop ends
# 0 "" 2
#NO_APP
	movq	96(%rbp), %rax
	movq	104(%rbp), %rdx
	imulq	%r9, %rax
	addq	%rdi, %rax
	addq	$16, %rdi
	leaq	(%rdx,%rax,4), %rax
	vaddps	(%rax), %zmm0, %zmm0
	vmovaps	%zmm0, (%rax)
	cmpq	%rdi, %r10
	jg	.L7
	addq	$1, %r9
	cmpq	%r9, %r11
	jne	.L4
	vzeroupper
.L14:
	movq	-8(%rbp), %rbx
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L9:
	.cfi_restore_state
	vxorps	%xmm0, %xmm0, %xmm0
	jmp	.L5
.L2:
	leaq	.LC0(%rip), %rcx
	movl	$8, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC2(%rip), %rdi
	call	__assert_fail@PLT
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
