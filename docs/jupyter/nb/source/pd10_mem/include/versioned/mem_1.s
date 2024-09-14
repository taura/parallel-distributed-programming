	.file	"mem_1.cc"
	.text
	.p2align 4
	.globl	_Z4scanPll
	.type	_Z4scanPll, @function
_Z4scanPll:
.LFB0:
	.cfi_startproc
	endbr64
	testq	%rsi, %rsi
	jle	.L4
	xorl	%eax, %eax
	xorl	%r8d, %r8d
	.p2align 4,,10
	.p2align 3
.L3:
	addq	$1, %rax
	movq	(%rdi,%r8,8), %r8
	cmpq	%rax, %rsi
	jne	.L3
	movq	%r8, %rax
	ret
	.p2align 4,,10
	.p2align 3
.L4:
	xorl	%r8d, %r8d
	movq	%r8, %rax
	ret
	.cfi_endproc
.LFE0:
	.size	_Z4scanPll, .-_Z4scanPll
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
