	.file	"mm_3.cc"
	.text
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC0:
	.string	"void gemm(matrix, matrix, matrix)"
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC1:
	.string	"versioned/mm_3.cc"
.LC2:
	.string	"M % bM == 0"
.LC3:
	.string	"N % L == 0"
	.text
	.p2align 4
	.globl	_Z4gemm6matrixS_S_
	.type	_Z4gemm6matrixS_S_, @function
_Z4gemm6matrixS_S_:
.LFB5691:
	.cfi_startproc
	endbr64
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-64, %rsp
	pushq	-8(%r10)
	pushq	%rbp
	movq	%rsp, %rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	pushq	%rbx
	addq	$-128, %rsp
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	movq	72(%r10), %rsi
	vmovq	8(%r10), %xmm13
	movq	%fs:40, %rax
	movq	%rax, -56(%rbp)
	movq	64(%r10), %rax
	movq	%rsi, -152(%rbp)
	movq	%rax, -160(%rbp)
	testb	$7, %al
	jne	.L24
	movq	%rsi, %rax
	testb	$15, %al
	jne	.L3
	cmpq	$0, -160(%rbp)
	jle	.L1
	movq	$0, -80(%rbp)
	movq	%r10, %r15
	vxorps	%xmm11, %xmm11, %xmm11
	testq	%rsi, %rsi
	jle	.L20
.L5:
	movq	-80(%rbp), %rax
	leaq	1(%rax), %rbx
	leaq	2(%rax), %rsi
	movq	%rbx, -136(%rbp)
	leaq	3(%rax), %rbx
	movq	%rsi, -128(%rbp)
	leaq	4(%rax), %rsi
	movq	%rbx, -120(%rbp)
	leaq	5(%rax), %rbx
	movq	%rsi, -112(%rbp)
	leaq	6(%rax), %rsi
	addq	$7, %rax
	movq	%rbx, -104(%rbp)
	movq	%rax, -88(%rbp)
	movq	%rsi, -96(%rbp)
	xorl	%esi, %esi
	.p2align 4,,10
	.p2align 3
.L12:
	movq	%rsp, -72(%rbp)
	cmpq	-72(%rbp), %rsp
	je	.L8
.L25:
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	cmpq	-72(%rbp), %rsp
	jne	.L25
.L8:
	subq	$576, %rsp
	orq	$0, 568(%rsp)
	leaq	63(%rsp), %rcx
	andq	$-64, %rcx
	vmovaps	%zmm11, (%rcx)
	vmovaps	%zmm11, 64(%rcx)
	vmovaps	%zmm11, 128(%rcx)
	vmovaps	%zmm11, 192(%rcx)
	vmovaps	%zmm11, 256(%rcx)
	vmovaps	%zmm11, 320(%rcx)
	vmovaps	%zmm11, 384(%rcx)
	vmovaps	%zmm11, 448(%rcx)
#APP
# 17 "versioned/mm_3.cc" 1
	# loop begins
# 0 "" 2
#NO_APP
	vmovq	%xmm13, %rax
	testq	%rax, %rax
	jle	.L10
	movq	16(%r15), %rax
	movq	-128(%rbp), %rbx
	vmovq	%xmm13, %r13
	movq	-136(%rbp), %r12
	movq	-120(%rbp), %r11
	movq	-112(%rbp), %r10
	movq	-104(%rbp), %r9
	imulq	%rax, %rbx
	movq	-96(%rbp), %r8
	movq	-88(%rbp), %rdi
	imulq	%rax, %r12
	imulq	%rax, %r11
	vmovaps	(%rcx), %zmm8
	vmovaps	64(%rcx), %zmm7
	imulq	%rax, %r10
	vmovaps	128(%rcx), %zmm6
	vmovaps	192(%rcx), %zmm5
	vmovaps	256(%rcx), %zmm4
	imulq	%rax, %r9
	vmovaps	320(%rcx), %zmm3
	vmovaps	384(%rcx), %zmm2
	imulq	%rax, %r8
	vmovaps	448(%rcx), %zmm1
	imulq	%rax, %rdi
	imulq	-80(%rbp), %rax
	vmovq	%rax, %xmm0
	movq	24(%r15), %rax
	vmovq	%xmm0, %rdx
	leaq	(%rax,%rdx,4), %rax
	movq	48(%r15), %rdx
	movq	%rax, -144(%rbp)
	vmovq	%xmm0, %rax
	addq	%rax, %r13
	movq	24(%r15), %rax
	leaq	0(,%rdx,4), %r14
	movq	56(%r15), %rdx
	leaq	(%rax,%r13,4), %r13
	vmovq	%xmm0, %rax
	leaq	(%rdx,%rsi,4), %rdx
	subq	%rax, %r12
	subq	%rax, %rbx
	subq	%rax, %r11
	subq	%rax, %r10
	subq	%rax, %r9
	subq	%rax, %r8
	subq	%rax, %rdi
	movq	-144(%rbp), %rax
	.p2align 4,,10
	.p2align 3
.L11:
	vmovaps	(%rdx), %zmm0
	addq	%r14, %rdx
	vfmadd231ps	(%rax){1to16}, %zmm0, %zmm8
	vfmadd231ps	(%rax,%r12,4){1to16}, %zmm0, %zmm7
	vfmadd231ps	(%rax,%rbx,4){1to16}, %zmm0, %zmm6
	vfmadd231ps	(%rax,%r11,4){1to16}, %zmm0, %zmm5
	vfmadd231ps	(%rax,%r10,4){1to16}, %zmm0, %zmm4
	vfmadd231ps	(%rax,%r9,4){1to16}, %zmm0, %zmm3
	vfmadd231ps	(%rax,%r8,4){1to16}, %zmm0, %zmm2
	vfmadd231ps	(%rax,%rdi,4){1to16}, %zmm0, %zmm1
	addq	$4, %rax
	cmpq	%rax, %r13
	jne	.L11
	vmovaps	%zmm8, (%rcx)
	vmovaps	%zmm7, 64(%rcx)
	vmovaps	%zmm6, 128(%rcx)
	vmovaps	%zmm5, 192(%rcx)
	vmovaps	%zmm4, 256(%rcx)
	vmovaps	%zmm3, 320(%rcx)
	vmovaps	%zmm2, 384(%rcx)
	vmovaps	%zmm1, 448(%rcx)
.L10:
#APP
# 23 "versioned/mm_3.cc" 1
	# loop ends
# 0 "" 2
#NO_APP
	movq	80(%r15), %rax
	movq	-80(%rbp), %rdi
	movq	88(%r15), %rdx
	imulq	%rax, %rdi
	addq	%rsi, %rdi
	leaq	(%rdx,%rdi,4), %rdi
	vmovaps	(%rdi), %zmm1
	vaddps	(%rcx), %zmm1, %zmm0
	vmovaps	%zmm0, (%rdi)
	movq	-136(%rbp), %rdi
	imulq	%rax, %rdi
	addq	%rsi, %rdi
	leaq	(%rdx,%rdi,4), %rdi
	vmovaps	(%rdi), %zmm2
	vaddps	64(%rcx), %zmm2, %zmm0
	vmovaps	%zmm0, (%rdi)
	movq	-128(%rbp), %rdi
	imulq	%rax, %rdi
	addq	%rsi, %rdi
	leaq	(%rdx,%rdi,4), %rdi
	vmovaps	(%rdi), %zmm3
	vaddps	128(%rcx), %zmm3, %zmm0
	vmovaps	%zmm0, (%rdi)
	movq	-120(%rbp), %rdi
	imulq	%rax, %rdi
	addq	%rsi, %rdi
	leaq	(%rdx,%rdi,4), %rdi
	vmovaps	(%rdi), %zmm4
	vaddps	192(%rcx), %zmm4, %zmm0
	vmovaps	%zmm0, (%rdi)
	movq	-112(%rbp), %rdi
	imulq	%rax, %rdi
	addq	%rsi, %rdi
	leaq	(%rdx,%rdi,4), %rdi
	vmovaps	(%rdi), %zmm5
	vaddps	256(%rcx), %zmm5, %zmm0
	vmovaps	%zmm0, (%rdi)
	movq	-104(%rbp), %rdi
	imulq	%rax, %rdi
	addq	%rsi, %rdi
	leaq	(%rdx,%rdi,4), %rdi
	vmovaps	(%rdi), %zmm6
	vaddps	320(%rcx), %zmm6, %zmm0
	vmovaps	%zmm0, (%rdi)
	movq	-96(%rbp), %rdi
	imulq	%rax, %rdi
	imulq	-88(%rbp), %rax
	addq	%rsi, %rdi
	leaq	(%rdx,%rdi,4), %rdi
	addq	%rsi, %rax
	addq	$16, %rsi
	vmovaps	(%rdi), %zmm7
	leaq	(%rdx,%rax,4), %rax
	vaddps	384(%rcx), %zmm7, %zmm0
	vmovaps	%zmm0, (%rdi)
	vmovaps	(%rax), %zmm1
	vaddps	448(%rcx), %zmm1, %zmm0
	vmovaps	%zmm0, (%rax)
	movq	-72(%rbp), %rsp
	cmpq	%rsi, -152(%rbp)
	jg	.L12
	addq	$8, -80(%rbp)
	movq	-80(%rbp), %rax
	cmpq	%rax, -160(%rbp)
	jg	.L5
.L20:
	vzeroupper
.L1:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L26
	leaq	-48(%rbp), %rsp
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
.L24:
	.cfi_restore_state
	leaq	.LC0(%rip), %rcx
	movl	$9, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC2(%rip), %rdi
	call	__assert_fail@PLT
.L26:
	call	__stack_chk_fail@PLT
.L3:
	leaq	.LC0(%rip), %rcx
	movl	$10, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC3(%rip), %rdi
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
