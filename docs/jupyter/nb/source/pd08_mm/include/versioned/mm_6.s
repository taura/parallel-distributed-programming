	.file	"mm_6.cc"
	.text
	.section	.rodata._Z5gemmcILl288ELl544EEvR6matricIXT0_EERS0_IXT_EES4_.str1.8,"aMS",@progbits,1
	.align 8
.LC0:
	.string	"void gemmc(matric<K>&, matric<N>&, matric<N>&) [with long int N = 288; long int K = 544]"
	.section	.rodata._Z5gemmcILl288ELl544EEvR6matricIXT0_EERS0_IXT_EES4_.str1.1,"aMS",@progbits,1
.LC1:
	.string	"versioned/mm_6.cc"
.LC2:
	.string	"M % bM == 0"
	.section	.text._Z5gemmcILl288ELl544EEvR6matricIXT0_EERS0_IXT_EES4_,"axG",@progbits,_Z5gemmcILl288ELl544EEvR6matricIXT0_EERS0_IXT_EES4_,comdat
	.p2align 4
	.weak	_Z5gemmcILl288ELl544EEvR6matricIXT0_EERS0_IXT_EES4_
	.type	_Z5gemmcILl288ELl544EEvR6matricIXT0_EERS0_IXT_EES4_, @function
_Z5gemmcILl288ELl544EEvR6matricIXT0_EERS0_IXT_EES4_:
.LFB5706:
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
	movq	%fs:40, %rax
	movq	%rax, -56(%rbp)
	movq	(%rdx), %rax
	movq	%rax, %r9
	movq	%rax, -120(%rbp)
	andl	$7, %r9d
	jne	.L2
	testq	%rax, %rax
	jle	.L1
	movq	$0, -80(%rbp)
	vmovq	%rdi, %xmm21
	vmovq	%rsi, %xmm22
	movq	%rdx, %r10
	movq	$1152, -88(%rbp)
	vpxord	%zmm20, %zmm20, %zmm20
	movq	$864, -96(%rbp)
	movq	$576, -104(%rbp)
	movq	$288, -112(%rbp)
	movq	$0, -72(%rbp)
.L3:
	movq	-80(%rbp), %rax
	movq	-112(%rbp), %r14
	leaq	0(,%r9,4), %rdi
	xorl	%r11d, %r11d
	movq	-104(%rbp), %r13
	movq	-96(%rbp), %r12
	leaq	64(%rdi), %r8
	movq	-88(%rbp), %rbx
	salq	$2, %rax
	subq	%r9, %r14
	subq	%r9, %r13
	subq	%r9, %r12
	vmovq	%rax, %xmm19
	salq	$2, %r14
	salq	$2, %r13
	subq	%r9, %rbx
	salq	$2, %r12
	salq	$2, %rbx
.L8:
	movq	%rsp, %r15
	cmpq	%r15, %rsp
	je	.L5
.L18:
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	cmpq	%r15, %rsp
	jne	.L18
.L5:
	subq	$1088, %rsp
	orq	$0, 1080(%rsp)
	leaq	63(%rsp), %rcx
	andq	$-64, %rcx
	vmovaps	%zmm20, (%rcx)
	vmovaps	%zmm20, 64(%rcx)
	vmovaps	%zmm20, 128(%rcx)
	vmovaps	%zmm20, 192(%rcx)
	vmovaps	%zmm20, 256(%rcx)
	vmovaps	%zmm20, 320(%rcx)
	vmovaps	%zmm20, 384(%rcx)
	vmovaps	%zmm20, 448(%rcx)
	vmovaps	%zmm20, 512(%rcx)
	vmovaps	%zmm20, 576(%rcx)
	vmovaps	%zmm20, 640(%rcx)
	vmovaps	%zmm20, 704(%rcx)
	vmovaps	%zmm20, 768(%rcx)
	vmovaps	%zmm20, 832(%rcx)
	vmovaps	%zmm20, 896(%rcx)
	vmovaps	%zmm20, 960(%rcx)
#APP
# 63 "versioned/mm_6.cc" 1
	# loop begins
# 0 "" 2
#NO_APP
	vmovq	%xmm21, %rsi
	vmovq	%xmm19, %rax
	vmovaps	(%rcx), %zmm18
	vmovaps	64(%rcx), %zmm17
	vmovaps	128(%rcx), %zmm16
	addq	8(%rsi), %rax
	leaq	0(,%r11,4), %rsi
	vmovaps	192(%rcx), %zmm15
	movq	%rax, -128(%rbp)
	vmovq	%xmm22, %rax
	vmovaps	256(%rcx), %zmm14
	movq	8(%rax), %rdx
	vmovaps	320(%rcx), %zmm13
	movq	8(%rax), %rax
	vmovaps	384(%rcx), %zmm12
	addq	%rsi, %rdx
	vmovaps	448(%rcx), %zmm11
	leaq	626688(%rax,%rsi), %rsi
	movq	-128(%rbp), %rax
	vmovaps	512(%rcx), %zmm10
	vmovaps	576(%rcx), %zmm9
	vmovaps	640(%rcx), %zmm8
	vmovaps	704(%rcx), %zmm7
	vmovaps	768(%rcx), %zmm6
	vmovaps	832(%rcx), %zmm5
	vmovaps	896(%rcx), %zmm4
	vmovaps	960(%rcx), %zmm3
	.p2align 4,,10
	.p2align 3
.L7:
	vmovaps	(%rdx), %zmm1
	vmovaps	64(%rdx), %zmm0
	addq	$4, %rax
	addq	$1152, %rdx
	vbroadcastss	-4(%rax), %zmm2
	vfmadd231ps	%zmm1, %zmm2, %zmm18
	vfmadd231ps	%zmm0, %zmm2, %zmm17
	vbroadcastss	2172(%rax), %zmm2
	vfmadd231ps	%zmm2, %zmm1, %zmm16
	vfmadd231ps	%zmm2, %zmm0, %zmm15
	vbroadcastss	4348(%rax), %zmm2
	vfmadd231ps	%zmm2, %zmm1, %zmm14
	vfmadd231ps	%zmm2, %zmm0, %zmm13
	vbroadcastss	6524(%rax), %zmm2
	vfmadd231ps	%zmm2, %zmm1, %zmm12
	vfmadd231ps	%zmm2, %zmm0, %zmm11
	vbroadcastss	8700(%rax), %zmm2
	vfmadd231ps	%zmm2, %zmm1, %zmm10
	vfmadd231ps	%zmm2, %zmm0, %zmm9
	vbroadcastss	10876(%rax), %zmm2
	vfmadd231ps	%zmm2, %zmm1, %zmm8
	vfmadd231ps	%zmm2, %zmm0, %zmm7
	vbroadcastss	13052(%rax), %zmm2
	vfmadd231ps	%zmm2, %zmm1, %zmm6
	vfmadd231ps	%zmm2, %zmm0, %zmm5
	vbroadcastss	15228(%rax), %zmm2
	vfmadd231ps	%zmm2, %zmm1, %zmm4
	vfmadd231ps	%zmm2, %zmm0, %zmm3
	cmpq	%rdx, %rsi
	jne	.L7
	vmovaps	%zmm18, (%rcx)
	vmovaps	%zmm17, 64(%rcx)
	vmovaps	%zmm16, 128(%rcx)
	vmovaps	%zmm15, 192(%rcx)
	vmovaps	%zmm14, 256(%rcx)
	vmovaps	%zmm13, 320(%rcx)
	vmovaps	%zmm12, 384(%rcx)
	vmovaps	%zmm11, 448(%rcx)
	vmovaps	%zmm10, 512(%rcx)
	vmovaps	%zmm9, 576(%rcx)
	vmovaps	%zmm8, 640(%rcx)
	vmovaps	%zmm7, 704(%rcx)
	vmovaps	%zmm6, 768(%rcx)
	vmovaps	%zmm5, 832(%rcx)
	vmovaps	%zmm4, 896(%rcx)
	vmovaps	%zmm3, 960(%rcx)
#APP
# 71 "versioned/mm_6.cc" 1
	# loop ends
# 0 "" 2
#NO_APP
	movq	8(%r10), %rax
	addq	$32, %r11
	addq	%rdi, %rax
	vmovaps	(%rax), %zmm3
	vaddps	(%rcx), %zmm3, %zmm0
	vmovaps	%zmm0, (%rax)
	movq	8(%r10), %rax
	addq	%r8, %rax
	vmovaps	(%rax), %zmm4
	vaddps	64(%rcx), %zmm4, %zmm0
	vmovaps	%zmm0, (%rax)
	leaq	(%r14,%rdi), %rax
	addq	8(%r10), %rax
	vmovaps	(%rax), %zmm5
	vaddps	128(%rcx), %zmm5, %zmm0
	vmovaps	%zmm0, (%rax)
	leaq	(%r14,%r8), %rax
	addq	8(%r10), %rax
	vmovaps	(%rax), %zmm6
	vaddps	192(%rcx), %zmm6, %zmm0
	vmovaps	%zmm0, (%rax)
	leaq	0(%r13,%rdi), %rax
	addq	8(%r10), %rax
	vmovaps	(%rax), %zmm7
	vaddps	256(%rcx), %zmm7, %zmm0
	vmovaps	%zmm0, (%rax)
	leaq	0(%r13,%r8), %rax
	addq	8(%r10), %rax
	vmovaps	(%rax), %zmm3
	vaddps	320(%rcx), %zmm3, %zmm0
	vmovaps	%zmm0, (%rax)
	leaq	(%r12,%rdi), %rax
	addq	8(%r10), %rax
	vmovaps	(%rax), %zmm4
	vaddps	384(%rcx), %zmm4, %zmm0
	vmovaps	%zmm0, (%rax)
	leaq	(%r12,%r8), %rax
	addq	8(%r10), %rax
	vmovaps	(%rax), %zmm5
	vaddps	448(%rcx), %zmm5, %zmm0
	vmovaps	%zmm0, (%rax)
	leaq	(%rbx,%rdi), %rax
	addq	8(%r10), %rax
	vmovaps	(%rax), %zmm6
	vaddps	512(%rcx), %zmm6, %zmm0
	vmovaps	%zmm0, (%rax)
	leaq	(%rbx,%r8), %rax
	addq	8(%r10), %rax
	vmovaps	(%rax), %zmm7
	vaddps	576(%rcx), %zmm7, %zmm0
	vmovaps	%zmm0, (%rax)
	movq	8(%r10), %rax
	leaq	5760(%rax,%rdi), %rax
	vmovaps	(%rax), %zmm3
	vaddps	640(%rcx), %zmm3, %zmm0
	vmovaps	%zmm0, (%rax)
	movq	8(%r10), %rax
	leaq	5760(%rax,%r8), %rax
	vmovaps	(%rax), %zmm4
	vaddps	704(%rcx), %zmm4, %zmm0
	vmovaps	%zmm0, (%rax)
	movq	8(%r10), %rax
	leaq	6912(%rax,%rdi), %rax
	vmovaps	(%rax), %zmm5
	vaddps	768(%rcx), %zmm5, %zmm0
	vmovaps	%zmm0, (%rax)
	movq	8(%r10), %rax
	leaq	6912(%rax,%r8), %rax
	vmovaps	(%rax), %zmm6
	vaddps	832(%rcx), %zmm6, %zmm0
	vmovaps	%zmm0, (%rax)
	movq	8(%r10), %rax
	leaq	8064(%rax,%rdi), %rax
	subq	$-128, %rdi
	vmovaps	(%rax), %zmm7
	vaddps	896(%rcx), %zmm7, %zmm0
	vmovaps	%zmm0, (%rax)
	movq	8(%r10), %rax
	leaq	8064(%rax,%r8), %rax
	subq	$-128, %r8
	vmovaps	(%rax), %zmm3
	vaddps	960(%rcx), %zmm3, %zmm0
	vmovaps	%zmm0, (%rax)
	movq	%r15, %rsp
	cmpq	$288, %r11
	jne	.L8
	addq	$8, -72(%rbp)
	addq	$2304, %r9
	movq	-72(%rbp), %rax
	addq	$2304, -112(%rbp)
	addq	$2304, -104(%rbp)
	addq	$2304, -96(%rbp)
	addq	$2304, -88(%rbp)
	addq	$4352, -80(%rbp)
	cmpq	%rax, -120(%rbp)
	jg	.L3
	vzeroupper
.L1:
	movq	-56(%rbp), %rax
	subq	%fs:40, %rax
	jne	.L19
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
.L2:
	.cfi_restore_state
	leaq	.LC0(%rip), %rcx
	movl	$53, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC2(%rip), %rdi
	call	__assert_fail@PLT
.L19:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE5706:
	.size	_Z5gemmcILl288ELl544EEvR6matricIXT0_EERS0_IXT_EES4_, .-_Z5gemmcILl288ELl544EEvR6matricIXT0_EERS0_IXT_EES4_
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC3:
	.string	"void gemm(matrix, matrix, matrix)"
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC4:
	.string	"N % bN == 0"
.LC5:
	.string	"K % bK == 0"
	.text
	.p2align 4
	.globl	_Z4gemm6matrixS_S_
	.type	_Z4gemm6matrixS_S_, @function
_Z4gemm6matrixS_S_:
.LFB5697:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movl	$64, %edi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	andq	$-64, %rsp
	subq	$192, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movq	80(%rbp), %rbx
	movq	88(%rbp), %r14
	movq	%fs:40, %rax
	movq	%rax, 184(%rsp)
	xorl	%eax, %eax
	movq	24(%rbp), %r12
	movq	%rbx, %rsi
	movq	%r14, 64(%rsp)
	salq	$4, %rsi
	movq	%rbx, 128(%rsp)
	addq	%rbx, %rsi
	salq	$7, %rsi
	call	aligned_alloc@PLT
	movl	$626688, %esi
	movl	$64, %edi
	movq	$544, 144(%rsp)
	movq	%rax, 136(%rsp)
	call	aligned_alloc@PLT
	leaq	(%rbx,%rbx,8), %rsi
	movl	$64, %edi
	movq	%rbx, 160(%rsp)
	salq	$7, %rsi
	movq	%rax, 152(%rsp)
	call	aligned_alloc@PLT
	movabsq	$1024819115206086176, %rdx
	movq	%rax, 168(%rsp)
	movabsq	$-8198552921648689607, %rax
	imulq	%r14, %rax
	addq	%rdx, %rax
	movabsq	$64051194700380386, %rdx
	rorq	$5, %rax
	cmpq	%rdx, %rax
	ja	.L61
	movabsq	$8680820740569200761, %rdx
	movq	%r12, %rax
	imulq	%rdx
	movq	%rdx, %rax
	movq	%r12, %rdx
	sarq	$63, %rdx
	sarq	$8, %rax
	subq	%rdx, %rax
	movq	%rax, %rdx
	salq	$4, %rdx
	addq	%rdx, %rax
	salq	$5, %rax
	cmpq	%rax, %r12
	jne	.L22
	cmpq	$0, 64(%rsp)
	jle	.L20
	movq	$0, 112(%rsp)
	movq	104(%rbp), %rax
	movq	%r12, %r8
	movq	%rax, 96(%rsp)
	movq	96(%rbp), %rax
	leaq	0(,%rax,4), %r9
	leaq	160(%rsp), %rax
	movq	%rax, 72(%rsp)
	movq	%r9, 56(%rsp)
.L41:
	testq	%rbx, %rbx
	jle	.L24
	leaq	(%rbx,%rbx,8), %r10
	movq	96(%rsp), %rcx
	movq	56(%rsp), %r13
	xorl	%edi, %edi
	movq	168(%rsp), %rdx
	salq	$5, %r10
	movq	%rdx, %rsi
.L26:
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L25:
	vmovaps	(%rcx,%rax,4), %zmm2
	vmovaps	%zmm2, (%rsi,%rax,4)
	addq	$16, %rax
	cmpq	$288, %rax
	jne	.L25
	addq	$288, %rdi
	addq	%r13, %rcx
	addq	$1152, %rsi
	cmpq	%rdi, %r10
	jne	.L26
	testq	%r8, %r8
	jle	.L44
.L43:
	movq	64(%rbp), %rbx
	movq	40(%rbp), %rax
	movq	%r8, 104(%rsp)
	xorl	%r15d, %r15d
	movq	112(%rsp), %rsi
	movq	%r15, %r13
	leaq	128(%rsp), %rdi
	movq	%rax, 88(%rsp)
	movq	%rbx, %rax
	salq	$4, %rax
	addq	%rbx, %rax
	salq	$2, %rbx
	salq	$7, %rax
	movq	%rbx, %r15
	movq	%rax, 80(%rsp)
	movq	72(%rbp), %rax
	leaq	(%rax,%rsi,4), %r12
	movq	32(%rbp), %rax
	leaq	144(%rsp), %rsi
	movq	%rsi, %rbx
	leaq	0(,%rax,4), %r14
.L34:
	movq	128(%rsp), %r10
	testq	%r10, %r10
	jle	.L33
	movq	88(%rsp), %rax
	movq	136(%rsp), %rdx
	leaq	(%rax,%r13,4), %rcx
	movq	%r10, %rax
	salq	$4, %rax
	addq	%rax, %r10
	salq	$7, %r10
	addq	%rdx, %r10
	.p2align 4,,10
	.p2align 3
.L32:
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L31:
	vmovaps	(%rcx,%rax,4), %zmm0
	vmovaps	%zmm0, (%rdx,%rax,4)
	addq	$16, %rax
	cmpq	$544, %rax
	jne	.L31
	addq	$2176, %rdx
	addq	%r14, %rcx
	cmpq	%rdx, %r10
	jne	.L32
.L33:
	movq	144(%rsp), %rax
	testq	%rax, %rax
	jle	.L30
	leaq	(%rax,%rax,8), %r11
	movq	152(%rsp), %rcx
	movq	%r12, %rdx
	xorl	%r10d, %r10d
	salq	$5, %r11
	.p2align 4,,10
	.p2align 3
.L37:
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L36:
	vmovaps	(%rdx,%rax,4), %zmm1
	vmovaps	%zmm1, (%rcx,%rax,4)
	addq	$16, %rax
	cmpq	$288, %rax
	jne	.L36
	addq	$288, %r10
	addq	%r15, %rdx
	addq	$1152, %rcx
	cmpq	%r10, %r11
	jne	.L37
.L30:
	movq	72(%rsp), %rdx
	movq	%rbx, %rsi
	movq	%rdi, 120(%rsp)
	vzeroupper
	call	_Z5gemmcILl288ELl544EEvR6matricIXT0_EERS0_IXT_EES4_
	addq	$544, %r13
	addq	80(%rsp), %r12
	cmpq	104(%rsp), %r13
	movq	120(%rsp), %rdi
	jl	.L34
	movq	160(%rsp), %rbx
	movq	104(%rsp), %r8
	testq	%rbx, %rbx
	jle	.L38
	movq	168(%rsp), %rdx
.L44:
	movq	96(%rsp), %rcx
	movq	56(%rsp), %r13
	xorl	%esi, %esi
.L40:
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L39:
	vmovaps	(%rdx,%rax,4), %zmm3
	vmovaps	%zmm3, (%rcx,%rax,4)
	addq	$16, %rax
	cmpq	$288, %rax
	jne	.L39
	addq	$1, %rsi
	addq	$1152, %rdx
	addq	%r13, %rcx
	cmpq	%rbx, %rsi
	jl	.L40
.L38:
	addq	$288, 112(%rsp)
	movq	112(%rsp), %rax
	addq	$1152, 96(%rsp)
	cmpq	64(%rsp), %rax
	jl	.L41
	vzeroupper
.L20:
	movq	184(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L62
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L24:
	.cfi_restore_state
	testq	%r8, %r8
	jg	.L43
	jmp	.L38
.L61:
	leaq	.LC3(%rip), %rcx
	movl	$90, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
.L62:
	call	__stack_chk_fail@PLT
.L22:
	leaq	.LC3(%rip), %rcx
	movl	$91, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC5(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5697:
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
