bits 64

default rel

global Microkernel

section .data
shuffle_mask: db 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14

section .text

StoreRow:
    mov eax,    0FF00h
    movd    xmm8,   eax
    vpbroadcastw    xmm8,   xmm8
    movdqu  xmm9,   [shuffle_mask]
    vperm2i128  ymm10,  ymm0,   ymm0,   083h
    vperm2i128  ymm11,  ymm1,   ymm1,   083h
    vpslldq xmm12,  xmm0,   1
    vpslldq xmm13,  xmm1,   1
    vpblendvb   xmm0,   xmm10,  xmm12,  xmm8
    vpblendvb   xmm1,   xmm11,  xmm13,  xmm8
    pshufb  xmm0,   xmm9
    pshufb  xmm1,   xmm9
    cmp esi,    32
    jb  l1
    movdqu  xmm2,   [rdi]
    paddb   xmm0,   xmm2
    vmovdqu [rdi],  xmm0
    movdqu  xmm3,   [rdi + 16]
    paddb   xmm1,   xmm3
    vmovdqu [rdi + 16], xmm1
    jmp l6
l1:
    cmp esi,    16
    jb  l2
    movdqu  xmm2,   [rdi]
    paddb   xmm0,   xmm2
    vmovdqu [rdi],  xmm0
    vmovdqa xmm0,   xmm1
    add rdi,    16
    sub esi,    16
l2:
    cmp esi,    8
    jb  l3
    movq    xmm2,   qword [rdi]
    paddb   xmm0,   xmm2
    movq    qword [rdi],    xmm0
    psrldq  xmm0,   8
    add rdi,    8
    sub esi,    8
l3:
    cmp esi,    4
    jb  l4
    movd    xmm2,   dword [rdi]
    paddb   xmm0,   xmm2
    movd    dword [rdi],    xmm0
    psrldq  xmm0,   4
    add rdi,    4
    sub esi,    4
l4:
    cmp esi,    2
    jb  l5
    movd    eax,    xmm0
    add [rdi],  al
    add [rdi + 1],  ah
    psrldq  xmm0,   2
    add rdi,    2
    sub esi,    2
l5:
    cmp esi, 1
    jb  l6
    movd    eax,    xmm0
    add [rdi],  al
l6:
    jmp store_row_return_label

Microkernel:
    push    rbp
    mov rbp,    rsp
    push    r10
    push    r11
    push    r12
    sub rsp,    8 * 32 + 8
    movdqu  [rsp],  xmm6
    movdqu  [rsp + 16], xmm7
    movdqu  [rsp + 2 * 16], xmm8
    movdqu  [rsp + 3 * 16], xmm9
    movdqu  [rsp + 4 * 16], xmm10
    movdqu  [rsp + 5 * 16], xmm11
    movdqu  [rsp + 6 * 16], xmm12
    movdqu  [rsp + 7 * 16], xmm13
    movdqu  [rsp + 8 * 16], xmm14
    movdqu  [rsp + 9 * 16], xmm15
    vpxor   ymm0,   ymm0,   ymm0
    vmovdqu ymm1,   ymm0
    vmovdqu ymm2,   ymm0
    vmovdqu ymm3,   ymm0
    vmovdqu ymm4,   ymm0
    vmovdqu ymm5,   ymm0
    vmovdqu ymm6,   ymm0
    vmovdqu ymm7,   ymm0
    mov eax,    esi
    xor rsi,    rsi
    xor r10d,   r10d
    xor r11d,   r11d
    mov r12d,   dword [rbp + 16]
    cmp r12d,   1
    je  loop_head
    mov esi,    eax
    cmp r12d,   2
    je  loop_head
    lea r10d,   [2 * esi]
    cmp r12d,   3
    je  loop_head
    lea r11d,   [esi + r10d]
    mov eax,    dword [rbp + 32]
; BEGIN LOOP
loop_head:
    vlddqu  ymm8,   [rdx]
    vlddqu  ymm9,   [rdx + 32]
    vpbroadcastb    ymm10,  byte [rdi]
    vpbroadcastb    ymm11,  byte [rdi + rsi]
    vpmullw ymm14,  ymm8,   ymm10
    vpmullw ymm10,  ymm9,   ymm10
    vpmullw ymm15,  ymm8,   ymm11
    vpmullw ymm11,  ymm9,   ymm11
    vpaddw  ymm0,   ymm0,   ymm14
    vpaddw  ymm1,   ymm1,   ymm10
    vpaddw  ymm2,   ymm2,   ymm15
    vpaddw  ymm3,   ymm3,   ymm11
    vpbroadcastb    ymm12,  byte [rdi + r10]
    vpbroadcastb    ymm13,  byte [rdi + r11]
    vpmullw ymm14,  ymm8,   ymm12
    vpmullw ymm12,  ymm9,   ymm12
    vpmullw ymm15,  ymm8,   ymm13
    vpmullw ymm13,  ymm9,   ymm13
    vpaddw  ymm4,   ymm4,   ymm14
    vpaddw  ymm5,   ymm5,   ymm12
    vpaddw  ymm6,   ymm6,   ymm15
    vpaddw  ymm7,   ymm7,   ymm13
    add rdx, 64
    add rdi,    1
    sub eax,    1
    jne loop_head
; END LOOP
    mov r10,    r9
    test    r10,    r10
    jz  no_bias
    vpmovsxbw   ymm8,   [r10]
    vpmovsxbw   ymm9,   [r10 + 16]
    vpaddw  ymm0,   ymm0,   ymm8
    vpaddw  ymm1,   ymm1,   ymm9
    vpaddw  ymm2,   ymm2,   ymm8
    vpaddw  ymm3,   ymm3,   ymm9
    vpaddw  ymm4,   ymm4,   ymm8
    vpaddw  ymm5,   ymm5,   ymm9
    vpaddw  ymm6,   ymm6,   ymm8
    vpaddw  ymm7,   ymm7,   ymm9
no_bias:
    mov r10d,   r8d
    mov r11d,   dword [rbp + 24]
    sub rsp,    32 * 6
    mov rdx, rsp
    vmovdqu [rsp],  ymm2
    vmovdqu [rsp + 32], ymm3
    vmovdqu [rsp + 64], ymm4
    vmovdqu [rsp + 96], ymm5
    vmovdqu [rsp + 128],    ymm6
    vmovdqu [rsp + 160],    ymm7
store_loop_head:
    mov rdi,    rcx
    mov esi,    r11d
    jmp StoreRow
store_row_return_label:
    vmovdqu ymm0,   [rdx]
    vmovdqu ymm1,   [rdx + 32]
    add rdx, 64
    add rcx, r10
    sub r12d,   1
    jne store_loop_head
    add rsp,    32 * 6
epilogue:
    movdqu  xmm6,   [rsp]
    movdqu  xmm7,   [rsp + 16]
    movdqu  xmm8,   [rsp + 2 * 16]
    movdqu  xmm9,   [rsp + 3 * 16]
    movdqu  xmm10,  [rsp + 4 * 16]
    movdqu  xmm11,  [rsp + 5 * 16]
    movdqu  xmm12,  [rsp + 6 * 16]
    movdqu  xmm13,  [rsp + 7 * 16]
    movdqu  xmm14,  [rsp + 8 * 16]
    movdqu  xmm15,  [rsp + 9 * 16]
    add rsp,    8 * 32 + 8
    pop r12
    pop r11
    pop r10
    vzeroupper
    leave
    ret
