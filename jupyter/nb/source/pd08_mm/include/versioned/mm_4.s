	.file	"mm_4.cc"
	.text
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC0:
	.string	"void gemm(matrix, matrix, matrix)"
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC1:
	.string	"versioned/mm_4.cc"
.LC2:
	.string	"M % bM == 0"
.LC3:
	.string	"N % (bN * L) == 0"
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
	vmovq	8(%r10), %xmm22
	movq	%fs:40, %rax
	movq	%rax, -56(%rbp)
	movq	64(%r10), %rax
	movq	%rsi, -152(%rbp)
	movq	%rax, -168(%rbp)
	testb	$7, %al
	jne	.L24
	movq	%rsi, %rax
	testb	$31, %al
	jne	.L3
	cmpq	$0, -168(%rbp)
	jle	.L1
	movq	$0, -80(%rbp)
	movq	%r10, %r15
	vpxord	%zmm21, %zmm21, %zmm21
	testq	%rsi, %rsi
	jle	.L1
.L5:
	movq	-80(%rbp), %rax
	leaq	1(%rax), %rsi
	movq	%rsi, -136(%rbp)
	leaq	2(%rax), %rsi
	movq	%rsi, -128(%rbp)
	leaq	3(%rax), %rsi
	movq	%rsi, -120(%rbp)
	leaq	4(%rax), %rsi
	movq	%rsi, -112(%rbp)
	leaq	5(%rax), %rsi
	movq	%rsi, -104(%rbp)
	leaq	6(%rax), %rsi
	addq	$7, %rax
	movq	%rax, -88(%rbp)
	movq	%rsi, -96(%rbp)
	xorl	%esi, %esi
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
	subq	$1088, %rsp
	orq	$0, 1080(%rsp)
	leaq	63(%rsp), %rdx
	andq	$-64, %rdx
	vmovaps	%zmm21, (%rdx)
	vmovaps	%zmm21, 64(%rdx)
	vmovaps	%zmm21, 128(%rdx)
	vmovaps	%zmm21, 192(%rdx)
	vmovaps	%zmm21, 256(%rdx)
	vmovaps	%zmm21, 320(%rdx)
	vmovaps	%zmm21, 384(%rdx)
	vmovaps	%zmm21, 448(%rdx)
	vmovaps	%zmm21, 512(%rdx)
	vmovaps	%zmm21, 576(%rdx)
	vmovaps	%zmm21, 640(%rdx)
	vmovaps	%zmm21, 704(%rdx)
	vmovaps	%zmm21, 768(%rdx)
	vmovaps	%zmm21, 832(%rdx)
	vmovaps	%zmm21, 896(%rdx)
	vmovaps	%zmm21, 960(%rdx)
#APP
# 20 "versioned/mm_4.cc" 1
	# loop begins
# 0 "" 2
#NO_APP
	leaq	16(%rsi), %rax
	movq	%rax, -160(%rbp)
	vmovq	%xmm22, %rax
	testq	%rax, %rax
	jle	.L10
	movq	16(%r15), %rax
	movq	-128(%rbp), %rbx
	vmovq	%xmm22, %r13
	movq	-136(%rbp), %r12
	movq	-120(%rbp), %r11
	movq	-112(%rbp), %r10
	movq	-104(%rbp), %r9
	imulq	%rax, %rbx
	movq	-96(%rbp), %r8
	movq	-88(%rbp), %rdi
	imulq	%rax, %r12
	imulq	%rax, %r11
	vmovaps	(%rdx), %zmm18
	vmovaps	64(%rdx), %zmm17
	imulq	%rax, %r10
	vmovaps	128(%rdx), %zmm16
	vmovaps	192(%rdx), %zmm15
	vmovaps	256(%rdx), %zmm14
	imulq	%rax, %r9
	vmovaps	320(%rdx), %zmm13
	vmovaps	384(%rdx), %zmm12
	imulq	%rax, %r8
	vmovaps	448(%rdx), %zmm11
	vmovaps	512(%rdx), %zmm10
	imulq	%rax, %rdi
	vmovaps	576(%rdx), %zmm9
	vmovaps	640(%rdx), %zmm8
	imulq	-80(%rbp), %rax
	vmovaps	704(%rdx), %zmm7
	vmovaps	768(%rdx), %zmm6
	vmovaps	832(%rdx), %zmm5
	vmovaps	896(%rdx), %zmm4
	vmovaps	960(%rdx), %zmm3
	vmovq	%rax, %xmm0
	movq	24(%r15), %rax
	vmovq	%xmm0, %rcx
	leaq	(%rax,%rcx,4), %rax
	movq	48(%r15), %rcx
	movq	%rax, -144(%rbp)
	vmovq	%xmm0, %rax
	addq	%rax, %r13
	movq	24(%r15), %rax
	leaq	0(,%rcx,4), %r14
	movq	56(%r15), %rcx
	leaq	(%rax,%r13,4), %r13
	vmovq	%xmm0, %rax
	leaq	(%rcx,%rsi,4), %rcx
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
	vmovaps	(%rcx), %zmm1
	vmovaps	64(%rcx), %zmm0
	addq	%r14, %rcx
	vbroadcastss	(%rax), %zmm2
	vfmadd231ps	%zmm1, %zmm2, %zmm18
	vfmadd231ps	%zmm0, %zmm2, %zmm17
	vbroadcastss	(%rax,%r12,4), %zmm2
	vfmadd231ps	%zmm2, %zmm1, %zmm16
	vfmadd231ps	%zmm2, %zmm0, %zmm15
	vbroadcastss	(%rax,%rbx,4), %zmm2
	vfmadd231ps	%zmm2, %zmm1, %zmm14
	vfmadd231ps	%zmm2, %zmm0, %zmm13
	vbroadcastss	(%rax,%r11,4), %zmm2
	vfmadd231ps	%zmm2, %zmm1, %zmm12
	vfmadd231ps	%zmm2, %zmm0, %zmm11
	vbroadcastss	(%rax,%r10,4), %zmm2
	vfmadd231ps	%zmm2, %zmm1, %zmm10
	vfmadd231ps	%zmm2, %zmm0, %zmm9
	vbroadcastss	(%rax,%r9,4), %zmm2
	vfmadd231ps	%zmm2, %zmm1, %zmm8
	vfmadd231ps	%zmm2, %zmm0, %zmm7
	vbroadcastss	(%rax,%r8,4), %zmm2
	vfmadd231ps	%zmm2, %zmm1, %zmm6
	vfmadd231ps	%zmm2, %zmm0, %zmm5
	vbroadcastss	(%rax,%rdi,4), %zmm2
	addq	$4, %rax
	vfmadd231ps	%zmm2, %zmm1, %zmm4
	vfmadd231ps	%zmm2, %zmm0, %zmm3
	cmpq	%r13, %rax
	jne	.L11
	vmovaps	%zmm18, (%rdx)
	vmovaps	%zmm17, 64(%rdx)
	vmovaps	%zmm16, 128(%rdx)
	vmovaps	%zmm15, 192(%rdx)
	vmovaps	%zmm14, 256(%rdx)
	vmovaps	%zmm13, 320(%rdx)
	vmovaps	%zmm12, 384(%rdx)
	vmovaps	%zmm11, 448(%rdx)
	vmovaps	%zmm10, 512(%rdx)
	vmovaps	%zmm9, 576(%rdx)
	vmovaps	%zmm8, 640(%rdx)
	vmovaps	%zmm7, 704(%rdx)
	vmovaps	%zmm6, 768(%rdx)
	vmovaps	%zmm5, 832(%rdx)
	vmovaps	%zmm4, 896(%rdx)
	vmovaps	%zmm3, 960(%rdx)
.L10:
#APP
# 28 "versioned/mm_4.cc" 1
	# loop ends
# 0 "" 2
#NO_APP
	movq	80(%r15), %rcx
	movq	-80(%rbp), %rdi
	movq	88(%r15), %rax
	movq	-160(%rbp), %rbx
	imulq	%rcx, %rdi
	leaq	(%rdi,%rsi), %r8
	addq	%rbx, %rdi
	leaq	(%rax,%r8,4), %r8
	leaq	(%rax,%rdi,4), %rdi
	vmovaps	(%r8), %zmm3
	vaddps	(%rdx), %zmm3, %zmm0
	vmovaps	%zmm0, (%r8)
	vmovaps	(%rdi), %zmm4
	vaddps	64(%rdx), %zmm4, %zmm0
	vmovaps	%zmm0, (%rdi)
	movq	-136(%rbp), %rdi
	imulq	%rcx, %rdi
	leaq	(%rdi,%rsi), %r8
	addq	%rbx, %rdi
	leaq	(%rax,%r8,4), %r8
	leaq	(%rax,%rdi,4), %rdi
	vmovaps	(%r8), %zmm5
	vaddps	128(%rdx), %zmm5, %zmm0
	vmovaps	%zmm0, (%r8)
	vmovaps	(%rdi), %zmm6
	vaddps	192(%rdx), %zmm6, %zmm0
	vmovaps	%zmm0, (%rdi)
	movq	-128(%rbp), %rdi
	imulq	%rcx, %rdi
	leaq	(%rdi,%rsi), %r8
	addq	%rbx, %rdi
	leaq	(%rax,%r8,4), %r8
	leaq	(%rax,%rdi,4), %rdi
	vmovaps	(%r8), %zmm7
	vaddps	256(%rdx), %zmm7, %zmm0
	vmovaps	%zmm0, (%r8)
	vmovaps	(%rdi), %zmm3
	vaddps	320(%rdx), %zmm3, %zmm0
	vmovaps	%zmm0, (%rdi)
	movq	-120(%rbp), %rdi
	imulq	%rcx, %rdi
	leaq	(%rdi,%rsi), %r8
	addq	%rbx, %rdi
	leaq	(%rax,%r8,4), %r8
	leaq	(%rax,%rdi,4), %rdi
	vmovaps	(%r8), %zmm4
	vaddps	384(%rdx), %zmm4, %zmm0
	vmovaps	%zmm0, (%r8)
	vmovaps	(%rdi), %zmm5
	vaddps	448(%rdx), %zmm5, %zmm0
	vmovaps	%zmm0, (%rdi)
	movq	-112(%rbp), %rdi
	imulq	%rcx, %rdi
	leaq	(%rdi,%rsi), %r8
	addq	%rbx, %rdi
	leaq	(%rax,%r8,4), %r8
	leaq	(%rax,%rdi,4), %rdi
	vmovaps	(%r8), %zmm6
	vaddps	512(%rdx), %zmm6, %zmm0
	vmovaps	%zmm0, (%r8)
	vmovaps	(%rdi), %zmm7
	vaddps	576(%rdx), %zmm7, %zmm0
	vmovaps	%zmm0, (%rdi)
	movq	-104(%rbp), %rdi
	imulq	%rcx, %rdi
	leaq	(%rdi,%rsi), %r8
	addq	%rbx, %rdi
	leaq	(%rax,%r8,4), %r8
	leaq	(%rax,%rdi,4), %rdi
	vmovaps	(%r8), %zmm3
	vaddps	640(%rdx), %zmm3, %zmm0
	vmovaps	%zmm0, (%r8)
	vmovaps	(%rdi), %zmm4
	vaddps	704(%rdx), %zmm4, %zmm0
	vmovaps	%zmm0, (%rdi)
	movq	-96(%rbp), %rdi
	imulq	%rcx, %rdi
	imulq	-88(%rbp), %rcx
	leaq	(%rdi,%rsi), %r8
	addq	%rbx, %rdi
	leaq	(%rax,%r8,4), %r8
	leaq	(%rax,%rdi,4), %rdi
	vmovaps	(%r8), %zmm5
	vaddps	768(%rdx), %zmm5, %zmm0
	vmovaps	%zmm0, (%r8)
	vmovaps	(%rdi), %zmm6
	vaddps	832(%rdx), %zmm6, %zmm0
	vmovaps	%zmm0, (%rdi)
	leaq	(%rcx,%rsi), %rdi
	addq	%rbx, %rcx
	addq	$32, %rsi
	leaq	(%rax,%rdi,4), %rdi
	leaq	(%rax,%rcx,4), %rax
	vmovaps	(%rdi), %zmm7
	vaddps	896(%rdx), %zmm7, %zmm0
	vmovaps	%zmm0, (%rdi)
	vmovaps	(%rax), %zmm3
	vaddps	960(%rdx), %zmm3, %zmm0
	vmovaps	%zmm0, (%rax)
	movq	-72(%rbp), %rsp
	cmpq	%rsi, -152(%rbp)
	jg	.L12
	addq	$8, -80(%rbp)
	movq	-80(%rbp), %rax
	cmpq	%rax, -168(%rbp)
	jg	.L5
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
	movl	$10, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC2(%rip), %rdi
	call	__assert_fail@PLT
.L26:
	call	__stack_chk_fail@PLT
.L3:
	leaq	.LC0(%rip), %rcx
	movl	$11, %edx
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
