push_callee_saved_general_regs macro
    push   r15
    push   r14
    push   r13
    push   r12
    push   rbx
    push   rsi
    push   rdi
endm

push_callee_saved_xmm_regs macro
    sub    rsp,    160
    movups [rsp],  xmm6
    movups [rsp + 16],     xmm7
    movups [rsp + 32],     xmm8
    movups [rsp + 48],     xmm9
    movups [rsp + 64],     xmm10
    movups [rsp + 80],     xmm11
    movups [rsp + 96],     xmm12
    movups [rsp + 112],    xmm13
    movups [rsp + 128],    xmm14
    movups [rsp + 144],    xmm15
endm

pop_callee_saved_xmm_regs macro
    movups xmm6,   [rsp]
    movups xmm7,   [rsp + 16]
    movups xmm8,   [rsp + 32]
    movups xmm9,   [rsp + 48]
    movups xmm10,  [rsp + 64]
    movups xmm11,  [rsp + 80]
    movups xmm12,  [rsp + 96]
    movups xmm13,  [rsp + 112]
    movups xmm14,  [rsp + 128]
    movups xmm15,  [rsp + 144]
    add    rsp,    160
endm
pop_callee_saved_general_regs macro
    pop    rdi
    pop    rsi
    pop    rbx
    pop    r12
    pop    r13
    pop    r14
    pop    r15
endm