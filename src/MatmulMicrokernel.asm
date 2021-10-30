public Microkernel

.data
shuffle_mask db 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14

.code

StoreRow:
    mov eax,    0FF00h
    movd    xmm8,   eax
    vpbroadcastw    xmm8,   xmm8
    movdqu  xmm9,  xmmword ptr [shuffle_mask]
    vperm2i128  ymm10,  ymm0,   ymm0,   083h
    vperm2i128  ymm11,  ymm1,   ymm1,   083h
    vpslldq xmm12,  xmm0,   1
    vpslldq xmm13,  xmm1,   1
    vpblendvb   xmm0,   xmm10,  xmm12,  xmm8
    vpblendvb   xmm1,   xmm11,  xmm13,  xmm8
    pshufb  xmm0,   xmm9
    pshufb  xmm1,   xmm9
    cmp edx,    32
    jb  l1
    movdqu  xmm2,   xmmword ptr [rcx]
    paddb   xmm0,   xmm2
    vmovdqu xmmword ptr [rcx],  xmm0
    movdqu  xmm3,   xmmword ptr [rcx]
    paddb   xmm0,   xmm3
    vmovdqu xmmword ptr [rcx + 16], xmm1
    jmp l6
l1:
    cmp edx,    16
    jb  l2
    movdqu  xmm2,   xmmword ptr [rcx]
    paddb   xmm0,   xmm2
    vmovdqu xmmword ptr [rcx],  xmm0
    vmovdqa xmm0,   xmm1
    add rcx,    16
    sub edx,    16
l2:
    cmp edx,    8
    jb  l3
    movq    xmm2,   qword ptr [rcx]
    paddb   xmm0,   xmm2
    movq    qword ptr [rcx],    xmm0
    psrldq  xmm0,   8
    add rcx,    8
    sub edx,    8
l3:
    cmp edx,    4
    jb  l4
    movd    xmm2,   dword ptr [rcx]
    paddb   xmm0,   xmm2
    movd    dword ptr [rcx],    xmm0
    psrldq  xmm0,   4
    add rcx,    4
    sub edx,    4
l4:
    cmp edx,    2
    jb  l5
    movd    eax,    xmm0
    add [rcx],  al
    add [rcx + 1],  ah
    psrldq  xmm0,   2
    add rcx,    2
    sub edx,    2
l5:
    cmp edx, 1
    jb  l6
    movd    eax,    xmm0
    add [rcx],  al
l6:
    jmp store_row_return_label

Microkernel:
    push    rbp
    mov rbp,    rsp
    push    r12
    sub rsp,    8 * 32 + 8
    movdqu  xmmword ptr [rsp],  xmm6
    movdqu  xmmword ptr [rsp + 16], xmm7
    movdqu  xmmword ptr [rsp + 2 * 16], xmm8
    movdqu  xmmword ptr [rsp + 3 * 16], xmm9
    movdqu  xmmword ptr [rsp + 4 * 16], xmm10
    movdqu  xmmword ptr [rsp + 5 * 16], xmm11
    movdqu  xmmword ptr [rsp + 6 * 16], xmm12
    movdqu  xmmword ptr [rsp + 7 * 16], xmm13
    movdqu  xmmword ptr [rsp + 8 * 16], xmm14
    movdqu  xmmword ptr [rsp + 9 * 16], xmm15
    vpxor   ymm0,   ymm0,   ymm0
    vmovdqu ymm1,   ymm0
    vmovdqu ymm2,   ymm0
    vmovdqu ymm3,   ymm0
    vmovdqu ymm4,   ymm0
    vmovdqu ymm5,   ymm0
    vmovdqu ymm6,   ymm0
    vmovdqu ymm7,   ymm0
    mov eax,    edx
    xor rdx,    rdx
    xor r10d,   r10d
    xor r11d,   r11d
    mov r12d,   dword ptr [rbp + 64]
    cmp r12d,   1
    je  loop_head
    mov edx,    eax
    cmp r12d,   2
    je  loop_head
    lea r10d,   [2 * edx]
    cmp r12d,   3
    je  loop_head
    lea r11d,   [edx + r10d]
    mov eax,    dword ptr [rbp + 80]
; BEGIN LOOP
loop_head:
    vlddqu  ymm8,   ymmword ptr [r8]
    vlddqu  ymm9,   ymmword ptr [r8 + 32]
    vpbroadcastb    ymm10,  byte ptr [rcx]
    vpbroadcastb    ymm11,  byte ptr [rcx + rdx]
    vpmullw ymm14,  ymm8,   ymm10
    vpmullw ymm10,  ymm9,   ymm10
    vpmullw ymm15,  ymm8,   ymm11
    vpmullw ymm11,  ymm9,   ymm11
    vpaddw  ymm0,   ymm0,   ymm14
    vpaddw  ymm1,   ymm1,   ymm10
    vpaddw  ymm2,   ymm2,   ymm15
    vpaddw  ymm3,   ymm3,   ymm11
    vpbroadcastb    ymm12,  byte ptr [rcx + r10]
    vpbroadcastb    ymm13,  byte ptr [rcx + r11]
    vpmullw ymm14,  ymm8,   ymm12
    vpmullw ymm12,  ymm9,   ymm12
    vpmullw ymm15,  ymm8,   ymm13
    vpmullw ymm13,  ymm9,   ymm13
    vpaddw  ymm4,   ymm4,   ymm14
    vpaddw  ymm5,   ymm5,   ymm12
    vpaddw  ymm6,   ymm6,   ymm15
    vpaddw  ymm7,   ymm7,   ymm13
    add r8, 64
    add rcx,    1
    sub eax,    1
    jne loop_head
; END LOOP
    mov r10,    qword ptr [rbp + 56]
    test    r10,    r10
    jz  no_bias
    vpmovsxbw   ymm8,   xmmword ptr [r10]
    vpmovsxbw   ymm9,   xmmword ptr [r10 + 16]
    vpaddw  ymm0,   ymm0,   ymm8
    vpaddw  ymm1,   ymm1,   ymm9
    vpaddw  ymm2,   ymm2,   ymm8
    vpaddw  ymm3,   ymm3,   ymm9
    vpaddw  ymm4,   ymm4,   ymm8
    vpaddw  ymm5,   ymm5,   ymm9
    vpaddw  ymm6,   ymm6,   ymm8
    vpaddw  ymm7,   ymm7,   ymm9
no_bias:
    mov r10d,   dword ptr [rbp + 48]
    mov r11d,   dword ptr [rbp + 72]
    sub rsp,    32 * 6
    mov r8, rsp
    vmovdqu ymmword ptr [rsp],  ymm2
    vmovdqu ymmword ptr [rsp + 32], ymm3
    vmovdqu ymmword ptr [rsp + 64], ymm4
    vmovdqu ymmword ptr [rsp + 96], ymm5
    vmovdqu ymmword ptr [rsp + 128],    ymm6
    vmovdqu ymmword ptr [rsp + 160],    ymm7
store_loop_head:
    mov rcx,    r9
    mov edx,    r11d
    jmp StoreRow
store_row_return_label:
    vmovdqu ymm0,   ymmword ptr [r8]
    vmovdqu ymm1,   ymmword ptr [r8 + 32]
    add r8, 64
    add r9, r10
    sub r12d,   1
    jne store_loop_head
epilogue:
    add rsp,    32 * 6
    movdqu  xmm6,   xmmword ptr [rsp]
    movdqu  xmm7,   xmmword ptr [rsp + 16]
    movdqu  xmm8,   xmmword ptr [rsp + 2 * 16]
    movdqu  xmm9,   xmmword ptr [rsp + 3 * 16]
    movdqu  xmm10,  xmmword ptr [rsp + 4 * 16]
    movdqu  xmm11,  xmmword ptr [rsp + 5 * 16]
    movdqu  xmm12,  xmmword ptr [rsp + 6 * 16]
    movdqu  xmm13,  xmmword ptr [rsp + 7 * 16]
    movdqu  xmm14,  xmmword ptr [rsp + 8 * 16]
    movdqu  xmm15,  xmmword ptr [rsp + 9 * 16]
    add rsp,    8 * 32 + 8
    pop r12
    vzeroupper
    leave
    ret

end