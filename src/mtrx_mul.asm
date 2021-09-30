include macros.inc

extern CreateThread : proc
extern WaitForMultipleObjectsEx : proc
extern SetThreadAffinityMask : proc
extern GetCurrentThread : proc

.const
INFINITE equ 0FFFFFFFFh

.code

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; mtrx_mul_f(src_matrix1, src_matrix1_h, src_matrix1_w, src_matrix2, ;;
;;             src_matrix2_w, dst_matrix, buffer)                     ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
mtrx_mul_f proc
    push   rbp
    mov    rbp,    rsp
    push_callee_saved_general_regs
    mov    ebx,    [rbp + 48]
    mov    [rbp + 16],     rcx
    mov    [rbp + 24],     edx
    mov    [rbp + 32],     r8d
    mov    r8d,    ebx
    mov    edx,    [rbp + 32]
    mov    rcx,    r9
    mov    r9,     [rbp + 64]
    sub    rsp,    32
    call   fill_buffer_f_
    add    rsp,    32
    mov    eax,    [rbp + 24]
    mul    qword ptr [rbp + 32]
    mul    qword ptr rbx
    cmp    rax,    512 * 512 * 512
    jae    async
    push   qword ptr [rbp + 56]
    push   rbx
    sub    rsp,    32
    mov    rcx,    [rbp + 16]
    mov    edx,    [rbp + 24]
    mov    r8d,    [rbp + 32]
    mov    r9,     [rbp + 64]
    call   mtrx_mul_f_
    add    rsp,    48
    jmp    exit
async:
    mov    rdi,    [rbp + 56]
    mov    rsi,    [rbp + 16]
    mov    eax,    [rbp + 24]
    shl    eax,    1
    mov    ebx,    7
    xor    edx,    edx
    div    ebx
    mov    ebx,    eax
    mov    eax,    [rbp + 32]
    mul    rbx
    mov    r11,    rax
    mov    eax,    [rbp + 48]
    mul    rbx
    mov    r12,    rax
    sub    rsp,    24
    mov    r10,    rsp
    mov    r13,    3
    mov    r15,    3
loop_head_create_threads:
    push   rdi
    push   qword ptr [rbp + 48]
    push   qword ptr [rbp + 64]
    push   qword ptr [rbp + 32]
    push   rbx
    push   rsi
    xor    rcx,    rcx
    xor    rdx,    rdx
    mov    r8,     mtrx_mul_f_async_
    mov    r9,     rsp
    push   r10
    push   r11
    push   0
    push   0
    sub    rsp,    32
    call   CreateThread
    add    rsp,    48
    pop    r11
    pop    r10
    mov    [r10],  rax
    add    r10,    8
    push   r10
    push   r11
    mov    rcx,    rax
    mov    edx,    r15d
    sub    rsp,    32
    call   SetThreadAffinityMask
    add    rsp,    32
    shl    r15d,   2
    pop    r11
    pop    r10
    lea    rsi,    [rsi + 4 * r11]
    lea    rdi,    [rdi + 4 * r12]
    sub    r13,    1
    jnz    loop_head_create_threads
    ; set cur thread affinity
    push   r10
    push   r11
    call   GetCurrentThread
    mov    rcx,    rax
    mov    edx,    r15d
    sub    rsp,    32
    call   SetThreadAffinityMask
    add    rsp,    32
    shl    r15d,   2
    pop    r11
    pop    r10
    ;
    mov    eax,    [rbp + 24]
    sub    eax,    ebx
    sub    eax,    ebx
    sub    eax,    ebx
    push   r10
    mov    rcx,    rsi
    mov    edx,    eax
    mov    r8d,    [rbp + 32]
    mov    r9,     [rbp + 64]
    push   rdi
    push   qword ptr [rbp + 48]
    sub    rsp,    32
    call   mtrx_mul_f_
    add    rsp,    48
    pop    r10
    mov    rcx,    3
    lea    rdx,    [r10 - 24]
    mov    r8,     1
    mov    r9,     INFINITE
    push   0
    sub    rsp,    32
    call   WaitForMultipleObjectsEx
    add    rsp,    40 + 3 * 48 + 24
exit:
    mov    rax,    [rbp + 56]
    pop_callee_saved_general_regs
    leave
    ret
mtrx_mul_f endp

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; void fill_buffer_f_(src_matrix, height, width, buffer) ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
fill_buffer_f_ proc
    push   rbp
    mov    rbp,    rsp
    push_callee_saved_general_regs
    mov    edi,    r8d
    test   edi,    -16
    jz     fill_buffer_8
loop_head:
    mov    rsi,    rcx
    mov    ebx,    edx
fill_buffer_16_loop_head:
    vmovupd        ymm0,   [rsi]
    vmovupd        ymm1,   [rsi + 32]
    vmovups        [r9],   ymm0
    vmovups        [r9 + 32],      ymm1
    lea    rsi,    [rsi + 4 * r8]
    add    r9,     64
    sub    ebx,    1
    jnz    fill_buffer_16_loop_head
    add    rcx,    64
    sub    edi,    16
    test   edi,    -16
    jnz    loop_head
fill_buffer_8:
    test   edi,    -8
    jz     fill_buffer_4
    mov    rsi,    rcx
    mov    ebx,    edx
    sub    edi,    8
    add    rcx,    32
fill_buffer_8_loop_head:
    vmovupd        ymm0,   [rsi]
    vmovups        [r9],   ymm0
    lea    rsi,    [rsi + 4 * r8]
    add    r9,     32
    sub    ebx,    1
    jnz    fill_buffer_8_loop_head
fill_buffer_4:
    test   edi,    -4
    jz     fill_buffer_2
    mov    rsi,    rcx
    mov    ebx,    edx
    sub    edi,    4
    add    rcx,    16
fill_buffer_4_loop_head:
    vmovupd        xmm0,   [rsi]
    vmovups        [r9],   xmm0
    lea    rsi,    [rsi + 4 * r8]
    add    r9,     16
    sub    ebx,    1
    jnz    fill_buffer_4_loop_head
fill_buffer_2:
    test    edi,    -2
    jz     fill_buffer_1
    mov    rsi,    rcx
    mov    ebx,    edx
    sub    edi,    2
    add    rcx,    8
fill_buffer_2_loop_head:
    vmovsd xmm0,   qword ptr [rsi]
    vmovsd qword ptr [r9], xmm0
    lea    rsi,    [rsi + 4 * r8]
    add    r9,     8
    sub    ebx,    1
    jnz    fill_buffer_2_loop_head
fill_buffer_1:
    test   edi,    edi
    jz     exit
    mov    rsi,    rcx
    mov    ebx,    edx
    sub    edi,    1
    add    rcx,    4
fill_buffer_1_loop_head:
    vmovss xmm0,   dword ptr [rsi]
    vmovss dword ptr [r9], xmm0
    lea    rsi,    [rsi + 4 * r8]
    add    r9,     4
    sub    ebx,    1
    jnz    fill_buffer_1_loop_head
exit:
    pop_callee_saved_general_regs
    leave
    ret
fill_buffer_f_ endp

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; mtrx_mul_f_async_(params) ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
mtrx_mul_f_async_ proc
    push   rbp
    mov    rbp,    rsp
    mov    edx,    [rcx + 8]
    mov    r8d,    [rcx + 16]
    mov    r9,     [rcx + 24]
    push   [rcx + 40]
    push   [rcx + 32]
    mov    rcx,    [rcx]
    sub    rsp,    32
    call   mtrx_mul_f_
    add    rsp,    48
    xor    rax,    rax
    leave
    ret
mtrx_mul_f_async_ endp

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; mtrx_mul_f_(src_matrix1, src_matrix1_h, src_matrix1_w, src_buffer2,  ;;
;;             src_matrix2_w, dst_matrix)                               ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
mtrx_mul_f_ proc
    push   rbp
    mov    rbp,    rsp
    push_callee_saved_general_regs
    push_callee_saved_xmm_regs
    mov    [rbp + 24],     edx
    mov    edx,    r8d
    mov    r8,     rcx
    mov    ecx,    [rbp + 48]
    mov    r10d,   edx
    shl    r10,    2
    mov    r11d,   ecx
    shl    r11,    2
    mov    r12,    [rbp + 56]
    ; R8 - src_matrix1
    ; [RBP + 24] - src_matrix1_h
    ; EDX - src_matrix1_w
    ; R10 - 4 * src_matrix1_w (invariant)
    ; R9 - src_buffer2
    ; ECX - src_matrix2_w
    ; R11 - 4 * src_matrix2_w (invariant)
    ; R12 - dst_matrix
    cmp    ecx,    16
    jb     kernels_Yx8
loop_head_kernels_Yx16:
    mov    edi,    [rbp + 24]
    mov    r15,    r12
    mov    rax,    r8
    cmp    edi,    6
    jb     kernel_4x16
loop_head_kernels_6x16:
    mov    r14,    r9
    mov    r13,    rax
    mov    ebx,    edx
    lea    rax,    [rax + 4 * r10]
    call   compute_kernel_6x16_f_
    lea    rax,    [rax + 2 * r10]
    sub    edi,    6
    cmp    edi,    6
    jae    loop_head_kernels_6x16
kernel_4x16:
    test   edi,    4
    jz     kernel_2x16
    mov    r14,    r9
    mov    r13,    rax
    mov    ebx,    edx
    lea    rax,    [rax + 4 * r10]
    call   compute_kernel_4x16_f_
kernel_2x16:
    test   edi,    2
    jz     kernel_1x16
    mov    r14,    r9
    mov    r13,    rax
    mov    ebx,    edx
    lea    rax,    [rax + 2 * r10]
    call   compute_kernel_2x16_f_
kernel_1x16:
    test   edi,    1
    jz     next_iteration
    mov    r14,    r9
    mov    r13,    rax
    mov    ebx,    edx
    call   compute_kernel_1x16_f_
next_iteration:
    mov    r9,     r14
    add    r12,    64
    sub    ecx,    16
    cmp    ecx,    16
    jae    loop_head_kernels_Yx16
kernels_Yx8:
    cmp    ecx,    8
    jb     kernels_Yx4
    mov    r15,    r12
    mov    rax,    r8
    mov    edi,    [rbp + 24]
    cmp    edi,    6
    jb     kernel_4x8
loop_head_kernels_6x8:
    mov    r14,    r9
    mov    r13,    rax
    mov    ebx,    edx
    lea    rax,    [rax + 4 * r10]
    call   compute_kernel_6x8_f_
    lea    rax,    [rax + 2 * r10]
    sub    edi,    6
    cmp    edi,    6
    jae    loop_head_kernels_6x8
kernel_4x8:
    test   edi,    4
    jz     kernel_2x8
    mov    r14,    r9
    mov    r13,    rax
    mov    ebx,    edx
    lea    rax,    [rax + 4 * r10]
    call   compute_kernel_4x8_f_
kernel_2x8:
    test   edi,    2
    jz     kernel_1x8
    mov    r14,    r9
    mov    r13,    rax
    mov    ebx,    edx
    lea    rax,    [rax + 2 * r10]
    call   compute_kernel_2x8_f_
kernel_1x8:
    test   edi,    1
    jz     kernels_Yx4
    mov    r14,    r9
    mov    r13,    rax
    mov    ebx,    edx
    call   compute_kernel_1x8_f_
    mov    r9,     r14
    add    r12,    32
    sub    ecx,    8
kernels_Yx4:
    cmp    ecx,    4
    jb     kernels_Yx2
    mov    r15,    r12
    mov    rax,    r8
    mov    edi,    [rbp + 24]
    cmp    edi,    6
    jb     kernel_4x4
loop_head_kernels_6x4:
    mov    r14,    r9
    mov    r13,    rax
    mov    ebx,    edx
    lea    rax,    [rax + 4 * r10]
    call   compute_kernel_6x4_f_
    lea    rax,    [rax + 2 * r10]
    sub    edi,    6
    cmp    edi,    6
    jae    loop_head_kernels_6x4
kernel_4x4:
    test   edi,    4
    jz     kernel_2x4
    mov    r14,    r9
    mov    r13,    rax
    mov    ebx,    edx
    lea    rax,    [rax + 4 * r10]
    call   compute_kernel_4x4_f_
kernel_2x4:
    test   edi,    2
    jz     kernel_1x4
    mov    r14,    r9
    mov    r13,    rax
    mov    ebx,    edx
    lea    rax,    [rax + 2 * r10]
    call   compute_kernel_2x4_f_
kernel_1x4:
    test   edi,    1
    jz     exit
    mov    r14,    r9
    mov    r13,    rax
    mov    ebx,    edx
    call   compute_kernel_1x4_f_
    mov    r9,     r14
    add    r12,    16
    sub    ecx,    4
kernels_Yx2:
    cmp    ecx,    2
    jb     kernels_Yx1
    mov    r15,    r12
    mov    rax,    r8
    mov    edi,    [rbp + 24]
    cmp    edi,    6
    jb     kernel_4x2
loop_head_kernels_6x2:
    mov    r14,    r9
    mov    r13,    rax
    mov    ebx,    edx
    lea    rax,    [rax + 4 * r10]
    call   compute_kernel_6x2_f_
    lea    rax,    [rax + 2 * r10]
    sub    edi,    6
    cmp    edi,    6
    jae    loop_head_kernels_6x2
kernel_4x2:
    test   edi,    4
    jz     kernel_2x2
    mov    r14,    r9
    mov    r13,    rax
    mov    ebx,    edx
    lea    rax,    [rax + 4 * r10]
    call   compute_kernel_4x2_f_
kernel_2x2:
    test   edi,    2
    jz     kernel_1x2
    mov    r14,    r9
    mov    r13,    rax
    mov    ebx,    edx
    lea    rax,    [rax + 2 * r10]
    call   compute_kernel_2x2_f_
kernel_1x2:
    test   edi,    1
    jz     exit
    mov    r14,    r9
    mov    r13,    rax
    mov    ebx,    edx
    call   compute_kernel_1x2_f_
    mov    r9,     r14
    add    r12,    8
    sub    ecx,    2
kernels_Yx1:
    test   ecx,    ecx
    jz     exit
    mov    r15,    r12
    mov    rax,    r8
    mov    edi,    [rbp + 24]
    cmp    edi,    6
    jb     kernel_4x1
loop_head_kernels_6x1:
    mov    r14,    r9
    mov    r13,    rax
    mov    ebx,    edx
    lea    rax,    [rax + 4 * r10]
    call   compute_kernel_6x1_f_
    lea    rax,    [rax + 2 * r10]
    sub    edi,    6
    cmp    edi,    6
    jae    loop_head_kernels_6x1
kernel_4x1:
    test   edi,    2
    jz     kernel_2x1
    mov    r14,    r9
    mov    r13,    rax
    mov    ebx,    edx
    lea    rax,    [rax + 4 * r10]
    call   compute_kernel_4x1_f_
kernel_2x1:
    test   edi,    2
    jz     kernel_1x1
    mov    r14,    r9
    mov    r13,    rax
    mov    ebx,    edx
    lea    rax,    [rax + 2 * r10]
    call   compute_kernel_2x1_f_
kernel_1x1:
    test   edi,    1
    jz     exit
    mov    r14,    r9
    mov    r13,    rax
    mov    ebx,    edx
    call   compute_kernel_1x1_f_
exit:
    pop_callee_saved_xmm_regs
    pop_callee_saved_general_regs
    leave
    ret
mtrx_mul_f_ endp

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; compute_kernel_6x16_f_(src_matrix1 = R13, src_matrix1_w = EBX, src_matrix1_w_real = R10, ;;
;;                     src_matrix2 = R14, src_matrix2_w_real = R11, dst_matrix = R15)       ;;
;; Uses RSI. Returns R15 = ptr to next kernel in dst_matrix                                 ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
compute_kernel_6x16_f_ proc
    mov    rsi,    r13
    vxorpd ymm0,   ymm0,   ymm0
    vxorpd ymm1,   ymm1,   ymm1
    vxorpd ymm2,   ymm2,   ymm2
    vxorpd ymm3,   ymm3,   ymm3
    vxorpd ymm4,   ymm4,   ymm4
    vxorpd ymm5,   ymm5,   ymm5
    vxorpd ymm6,   ymm6,   ymm6
    vxorpd ymm7,   ymm7,   ymm7
    vxorpd ymm8,   ymm8,   ymm8
    vxorpd ymm9,   ymm9,   ymm9
    vxorpd ymm10,  ymm10,  ymm10
    vxorpd ymm11,  ymm11,  ymm11
loop_head:
    add    rsi,    4
    vmovups        ymm13,  [r14]
    vmovups        ymm14,  [r14 + 32]
    vbroadcastss   ymm12,  dword ptr [r13]
    vbroadcastss   ymm15,  dword ptr [r13 + r10]
    vfmadd231ps    ymm0,   ymm12,  ymm13
    vfmadd231ps    ymm1,   ymm12,  ymm14
    vfmadd231ps    ymm2,   ymm15,  ymm13
    vfmadd231ps    ymm3,   ymm15,  ymm14
    lea    r13,    [r13 + 2 * r10]
    vbroadcastss   ymm12,  dword ptr [r13]
    vbroadcastss   ymm15,  dword ptr [r13 + r10]
    vfmadd231ps    ymm4,   ymm12,  ymm13
    vfmadd231ps    ymm5,   ymm12,  ymm14
    vfmadd231ps    ymm6,   ymm15,  ymm13
    vfmadd231ps    ymm7,   ymm15,  ymm14
    lea    r13,    [r13 + 2 * r10]
    vbroadcastss   ymm12,  dword ptr [r13]
    vbroadcastss   ymm15,  dword ptr [r13 + r10]
    vfmadd231ps    ymm8,   ymm12,  ymm13
    vfmadd231ps    ymm9,   ymm12,  ymm14
    vfmadd231ps    ymm10,  ymm15,  ymm13
    vfmadd231ps    ymm11,  ymm15,  ymm14
    lea    r14,    [r14 + 64]
    mov    r13,    rsi
    sub    ebx,    1
    jnz    loop_head
    vmovupd        [r15],  ymm0
    vmovupd        [r15 + 32],     ymm1
    lea    r15,    [r15 + r11]
    vmovupd        [r15],  ymm2
    vmovupd        [r15 + 32],     ymm3
    lea    r15,    [r15 + r11]
    vmovupd        [r15],  ymm4
    vmovupd        [r15 + 32],     ymm5
    lea    r15,    [r15 + r11]
    vmovupd        [r15],  ymm6
    vmovupd        [r15 + 32],     ymm7
    lea    r15,    [r15 + r11]
    vmovupd        [r15],  ymm8
    vmovupd        [r15 + 32],     ymm9
    lea    r15,    [r15 + r11]
    vmovupd        [r15],  ymm10
    vmovupd        [r15 + 32],     ymm11
    add    r15,    r11
    ret
compute_kernel_6x16_f_ endp

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; compute_kernel_4x16_f_(src_matrix1 = R13, src_matrix1_w = EBX, src_matrix1_w_real = R10, ;;
;;                     src_matrix2 = R14, src_matrix2_w_real = R11, dst_matrix = R15)       ;;
;; Uses RSI. Returns R15 = ptr to next kernel in dst_matrix                                 ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
compute_kernel_4x16_f_ proc
    mov    rsi,    r13
    vxorpd ymm0,   ymm0,   ymm0
    vxorpd ymm1,   ymm1,   ymm1
    vxorpd ymm2,   ymm2,   ymm2
    vxorpd ymm3,   ymm3,   ymm3
    vxorpd ymm4,   ymm4,   ymm4
    vxorpd ymm5,   ymm5,   ymm5
    vxorpd ymm6,   ymm6,   ymm6
    vxorpd ymm7,   ymm7,   ymm7
loop_head:
    add    rsi,    4
    vmovups        ymm9,   [r14]
    vmovups        ymm10,  [r14 + 32]
    vbroadcastss   ymm8,   dword ptr [r13]
    vbroadcastss   ymm11,  dword ptr [r13 + r10]
    lea    r13,    [r13 + 2 * r10]
    vbroadcastss   ymm12,  dword ptr [r13]
    vbroadcastss   ymm13,  dword ptr [r13 + r10]
    vfmadd231ps    ymm0,   ymm8,   ymm9
    vfmadd231ps    ymm1,   ymm8,   ymm10
    vfmadd231ps    ymm2,   ymm11,  ymm9
    vfmadd231ps    ymm3,   ymm11,  ymm10
    vfmadd231ps    ymm4,   ymm12,  ymm9
    vfmadd231ps    ymm5,   ymm12,  ymm10
    vfmadd231ps    ymm6,   ymm13,  ymm9
    vfmadd231ps    ymm7,   ymm13,  ymm10
    lea    r14,    [r14 + 64]
    mov    r13,    rsi
    sub    ebx,    1
    jnz    loop_head
    vmovupd        [r15],  ymm0
    vmovupd        [r15 + 32],     ymm1
    lea    r15,    [r15 + r11]
    vmovupd        [r15],  ymm2
    vmovupd        [r15 + 32],     ymm3
    lea    r15,    [r15 + r11]
    vmovupd        [r15],  ymm4
    vmovupd        [r15 + 32],     ymm5
    lea    r15,    [r15 + r11]
    vmovupd        [r15],  ymm6
    vmovupd        [r15 + 32],     ymm7
    lea    r15,    [r15 + r11]
    ret
compute_kernel_4x16_f_ endp

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; compute_kernel_2x16_f_(src_matrix1 = R13, src_matrix1_w = EBX, src_matrix1_w_real = R10, ;;
;;                     src_matrix2 = R14, src_matrix2_w_real = R11, dst_matrix = R15)       ;;
;; Returns R15 = ptr to next kernel in dst_matrix                                           ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
compute_kernel_2x16_f_ proc
    vxorpd ymm0,   ymm0,   ymm0
    vxorpd ymm1,   ymm1,   ymm1
    vxorpd ymm2,   ymm2,   ymm2
    vxorpd ymm3,   ymm3,   ymm3
loop_head:
    vmovups        ymm5,   [r14]
    vmovups        ymm6,   [r14 + 32]
    vbroadcastss   ymm4,   dword ptr [r13]
    vbroadcastss   ymm7,   dword ptr [r13 + r10]
    vfmadd231ps    ymm1,   ymm4,  ymm6
    vfmadd231ps    ymm0,   ymm4,  ymm5
    vfmadd231ps    ymm2,   ymm7,  ymm5
    vfmadd231ps    ymm3,   ymm7,  ymm6
    lea    r14,    [r14 + 64]
    add    r13,    4
    sub    ebx,    1
    jne    loop_head
    vmovupd        [r15],  ymm0
    vmovupd        [r15 + 32],    ymm1
    lea    r15,    [r15 + r11]
    vmovupd        [r15], ymm2
    vmovupd        [r15 + 32],    ymm3
    lea    r15,    [r15 + r11]
    ret
compute_kernel_2x16_f_ endp

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; compute_kernel_1x16_f_(src_matrix1 = R13, src_matrix1_w = EBX, src_matrix1_w_real = R10, ;;
;;                     src_matrix2 = R14, src_matrix2_w_real = R11, dst_matrix = R15)       ;;
;; Returns R15 = ptr to next kernel in dst_matrix                                           ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
compute_kernel_1x16_f_ proc
    vxorpd ymm0,   ymm0,   ymm0
    vxorpd ymm1,   ymm1,   ymm1
loop_head:
    vbroadcastss   ymm8,   dword ptr [r13]
    vmovups        ymm9,   [r14]
    vmovups        ymm10,  [r14 + 32]
    vfmadd231ps    ymm0,   ymm8,   ymm9
    vfmadd231ps    ymm1,   ymm8,   ymm10
    lea    r14,    [r14 + 64]
    add    r13,    4
    sub    ebx,    1
    jnz    loop_head
    vmovupd        [r15],  ymm0
    vmovupd        [r15 + 32],     ymm1
    ret
compute_kernel_1x16_f_ endp

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; compute_kernel_6x8_f_(src_matrix1 = R13, src_matrix1_w = EBX, src_matrix1_w_real = R10, ;;
;;                     src_matrix2 = R14, src_matrix2_w_real = R11, dst_matrix = R15)      ;;
;; Uses RSI. Returns R15 = ptr to next kernel in dst_matrix                                ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
compute_kernel_6x8_f_ proc
    mov    rsi,    r13
    vxorpd ymm0,   ymm0,   ymm0
    vxorpd ymm1,   ymm1,   ymm1
    vxorpd ymm2,   ymm2,   ymm2
    vxorpd ymm3,   ymm3,   ymm3
    vxorpd ymm4,   ymm4,   ymm4
    vxorpd ymm5,   ymm5,   ymm5
loop_head:
    add    rsi,    4
    vmovups        ymm8,   [r14]
    vbroadcastss   ymm9,   dword ptr [r13]
    vbroadcastss   ymm10,  dword ptr [r13 + r10]
    lea    r13,    [r13 + 2 * r10]
    vbroadcastss   ymm11,  dword ptr [r13]
    vbroadcastss   ymm12,  dword ptr [r13 + r10]
    lea    r13,    [r13 + 2 * r10]
    vbroadcastss   ymm13,  dword ptr [r13]
    vbroadcastss   ymm14,  dword ptr [r13 + r10]
    vfmadd231ps    ymm0,   ymm9,  ymm8
    vfmadd231ps    ymm1,   ymm10, ymm8
    vfmadd231ps    ymm2,   ymm11, ymm8
    vfmadd231ps    ymm3,   ymm12, ymm8
    vfmadd231ps    ymm4,   ymm13, ymm8
    vfmadd231ps    ymm5,   ymm14, ymm8
    mov    r13,    rsi
    add    r14,    32
    sub    ebx,    1
    jne    loop_head
    vmovupd        [r15], ymm0
    lea    r15,    [r15 + r11]
    vmovupd        [r15], ymm1
    lea    r15,    [r15 + r11]
    vmovupd        [r15], ymm2
    lea    r15,    [r15 + r11]
    vmovupd        [r15], ymm3
    lea    r15,    [r15 + r11]
    vmovupd        [r15], ymm4
    lea    r15,    [r15 + r11]
    vmovupd        [r15], ymm5
    add    r15,    r11
    ret
compute_kernel_6x8_f_ endp

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; compute_kernel_4x8_f_(src_matrix1 = R13, src_matrix1_w = EBX, src_matrix1_w_real = R10, ;;
;;                     src_matrix2 = R14, src_matrix2_w_real = R11, dst_matrix = R15)      ;;
;; Uses RSI. Returns R15 = ptr to next kernel in dst_matrix                                ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
compute_kernel_4x8_f_ proc
    mov    rsi,    r13
    vxorpd ymm0,   ymm0,   ymm0
    vxorpd ymm1,   ymm1,   ymm1
    vxorpd ymm2,   ymm2,   ymm2
    vxorpd ymm3,   ymm3,   ymm3
loop_head:
    add    rsi,    4
    vmovups        ymm8,   [r14]
    vbroadcastss   ymm9,   dword ptr [r13]
    lea    r13,    [r13 + r10]
    vbroadcastss   ymm10,  dword ptr [r13]
    vbroadcastss   ymm11,  dword ptr [r13 + r10]
    vbroadcastss   ymm12,  dword ptr [r13 + 2 * r10]
    vfmadd231ps    ymm0,   ymm9,  ymm8
    vfmadd231ps    ymm1,   ymm10, ymm8
    vfmadd231ps    ymm2,   ymm11, ymm8
    vfmadd231ps    ymm3,   ymm12, ymm8
    lea    r14,    [r14 + 32]
    mov    r13,    rsi
    sub    ebx,    1
    jne    loop_head
    vmovupd        [r15], ymm0
    lea    r15,    [r15 + r11]
    vmovupd        [r15], ymm1
    lea    r15,    [r15 + r11]
    vmovupd        [r15], ymm2
    lea    r15,    [r15 + r11]
    vmovupd        [r15], ymm3
    lea    r15,    [r15 + r11]
    ret
compute_kernel_4x8_f_ endp

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; compute_kernel_2x8_f_(src_matrix1 = R13, src_matrix1_w = EBX, src_matrix1_w_real = R10, ;;
;;                     src_matrix2 = R14, src_matrix2_w_real = R11, dst_matrix = R15)      ;;
;; Returns R15 = ptr to next kernel in dst_matrix                                          ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
compute_kernel_2x8_f_ proc
    vxorpd ymm0,   ymm0,   ymm0
    vxorpd ymm1,   ymm1,   ymm1
loop_head:
    vmovups        ymm8,   [r14]
    vbroadcastss   ymm9,   dword ptr [r13]
    vbroadcastss   ymm10,  dword ptr [r13 + r10]
    vfmadd231ps    ymm0,   ymm9,   ymm8
    vfmadd231ps    ymm1,   ymm10,  ymm8
    lea    r14,    [r14 + 32]
    add    r13,    4
    sub    ebx,    1
    jne    loop_head
    vmovupd        [r15], ymm0
    lea    r15,    [r15 + r11]
    vmovupd        [r15], ymm1
    lea    r15,    [r15 + r11]
    ret
compute_kernel_2x8_f_ endp

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; compute_kernel_1x8_f_(src_matrix1 = R13, src_matrix1_w = EBX, src_matrix1_w_real = R10, ;;
;;                     src_matrix2 = R14, src_matrix2_w_real = R11, dst_matrix = R15)      ;;
;; Returns R15 = ptr to next kernel in dst_matrix                                          ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
compute_kernel_1x8_f_ proc
    vxorpd ymm0,   ymm0,   ymm0
loop_head:
    vmovups        ymm9,   [r14]
    vbroadcastss   ymm8,   dword ptr [r13]
    vfmadd231ps    ymm0,   ymm8,  ymm9
    lea    r14,    [r14 + 32]
    add    r13,    4
    sub    ebx,    1
    jne    loop_head
    vmovupd        [r15], ymm0
    ret
compute_kernel_1x8_f_ endp

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; compute_kernel_6x4_f_(src_matrix1 = R13, src_matrix1_w = EBX, src_matrix1_w_real = R10, ;;
;;                     src_matrix2 = R14, src_matrix2_w_real = R11, dst_matrix = R15)      ;;
;; Uses RSI. Returns R15 = ptr to next kernel in dst_matrix                                ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
compute_kernel_6x4_f_ proc
    mov    rsi,    r13
    vxorpd xmm0,   xmm0,   xmm0
    vxorpd xmm1,   xmm1,   xmm1
    vxorpd xmm2,   xmm2,   xmm2
    vxorpd xmm3,   xmm3,   xmm3
    vxorpd xmm4,   xmm4,   xmm4
    vxorpd xmm5,   xmm5,   xmm5
loop_head:
    add    rsi,    4
    vmovups        xmm8,   [r14]
    vbroadcastss   xmm9,   dword ptr [r13]
    vbroadcastss   xmm10,  dword ptr [r13 + r10]
    lea    r13,    [r13 + 2 * r10]
    vbroadcastss   xmm11,  dword ptr [r13]
    vbroadcastss   xmm12,  dword ptr [r13 + r10]
    lea    r13,    [r13 + 2 * r10]
    vbroadcastss   ymm13,  dword ptr [r13]
    vbroadcastss   xmm14,  dword ptr [r13 + r10]
    vfmadd231ps    xmm0,   xmm9,  xmm8
    vfmadd231ps    xmm1,   xmm10, xmm8
    vfmadd231ps    xmm2,   xmm11, xmm8
    vfmadd231ps    xmm3,   xmm12, xmm8
    vfmadd231ps    xmm4,   xmm13, xmm8
    vfmadd231ps    xmm5,   xmm14, xmm8
    mov    r13,    rsi
    add    r14,    16
    sub    ebx,    1
    jne    loop_head
    vmovupd        [r15], xmm0
    lea    r15,    [r15 + r11]
    vmovupd        [r15], xmm1
    lea    r15,    [r15 + r11]
    vmovupd        [r15], xmm2
    lea    r15,    [r15 + r11]
    vmovupd        [r15], xmm3
    lea    r15,    [r15 + r11]
    vmovupd        [r15], xmm4
    lea    r15,    [r15 + r11]
    vmovupd        [r15], xmm5
    add    r15,    r11
    ret
compute_kernel_6x4_f_ endp

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; compute_kernel_4x4_f_(src_matrix1 = R13, src_matrix1_w = EBX, src_matrix1_w_real = R10, ;;
;;                     src_matrix2 = R14, src_matrix2_w_real = R11, dst_matrix = R15)      ;;
;; Uses RSI. Returns R15 = ptr to next kernel in dst_matrix                                ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
compute_kernel_4x4_f_ proc
    mov    rsi,    r13
    vxorpd xmm0,   xmm0,   xmm0
    vxorpd xmm1,   xmm1,   xmm1
    vxorpd xmm2,   xmm2,   xmm2
    vxorpd xmm3,   xmm3,   xmm3
loop_head:
    add    rsi,    4
    vmovups        xmm8,   [r14]
    vbroadcastss   xmm9,   dword ptr [r13]
    lea    r13,    [r13 + r10]
    vbroadcastss   xmm10,  dword ptr [r13]
    vbroadcastss   xmm11,  dword ptr [r13 + r10]
    vbroadcastss   xmm12,  dword ptr [r13 + 2 * r10]
    vfmadd231ps    xmm0,   xmm9,  xmm8
    vfmadd231ps    xmm1,   xmm10, xmm8
    vfmadd231ps    xmm2,   xmm11, xmm8
    vfmadd231ps    xmm3,   xmm12, xmm8
    lea    r14,    [r14 + 16]
    mov    r13,    rsi
    sub    ebx,    1
    jne    loop_head
    vmovups        [r15], xmm0
    lea    r15,    [r15 + r11]
    vmovups        [r15], xmm1
    lea    r15,    [r15 + r11]
    vmovups        [r15], xmm2
    lea    r15,    [r15 + r11]
    vmovups        [r15], xmm3
    lea    r15,    [r15 + r11]
    ret
compute_kernel_4x4_f_ endp

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; compute_kernel_2x4_f_(src_matrix1 = R13, src_matrix1_w = EBX, src_matrix1_w_real = R10, ;;
;;                     src_matrix2 = R14, src_matrix2_w_real = R11, dst_matrix = R15)      ;;
;; Returns R15 = ptr to next kernel in dst_matrix                                          ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
compute_kernel_2x4_f_ proc
    vxorpd xmm0,   xmm0,   xmm0
    vxorpd xmm1,   xmm1,   xmm1
loop_head:
    vmovups        xmm8,   [r14]
    vbroadcastss   xmm9,   dword ptr [r13]
    vbroadcastss   xmm10,  dword ptr [r13 + r10]
    vfmadd231ps    xmm0,   xmm9,   xmm8
    vfmadd231ps    xmm1,   xmm10,  xmm8
    lea    r14,    [r14 + 16]
    add    r13,    4
    sub    ebx,    1
    jne    loop_head
    vmovupd        [r15], xmm0
    lea    r15,    [r15 + r11]
    vmovupd        [r15], xmm1
    lea    r15,    [r15 + r11]
    ret
compute_kernel_2x4_f_ endp

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; compute_kernel_1x4_f_(src_matrix1 = R13, src_matrix1_w = EBX, src_matrix1_w_real = R10, ;;
;;                     src_matrix2 = R14, src_matrix2_w_real = R11, dst_matrix = R15)      ;;
;; Returns R15 = ptr to next kernel in dst_matrix                                          ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
compute_kernel_1x4_f_ proc
    vxorpd xmm0,   xmm0,   xmm0
loop_head:
    vmovups        xmm9,   [r14]
    vbroadcastss   xmm8,   dword ptr [r13]
    vfmadd231ps    xmm0,   xmm8,  xmm9
    lea    r14,    [r14 + 16]
    add    r13,    4
    sub    ebx,    1
    jne    loop_head
    vmovupd        [r15], xmm0
    ret
compute_kernel_1x4_f_ endp

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; compute_kernel_6x2_f_(src_matrix1 = R13, src_matrix1_w = EBX, src_matrix1_w_real = R10, ;;
;;                     src_matrix2 = R14, src_matrix2_w_real = R11, dst_matrix = R15)      ;;
;; Uses RSI. Returns R15 = ptr to next kernel in dst_matrix                                ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
compute_kernel_6x2_f_ proc
    mov    rsi,    r13
    vxorpd xmm0,   xmm0,   xmm0
    vxorpd xmm1,   xmm1,   xmm1
    vxorpd xmm2,   xmm2,   xmm2
    vxorpd xmm3,   xmm3,   xmm3
    vxorpd xmm4,   xmm4,   xmm4
    vxorpd xmm5,   xmm5,   xmm5
    vxorpd xmm6,   xmm6,   xmm6
    vxorpd xmm7,   xmm7,   xmm7
    vxorpd xmm8,   xmm8,   xmm8
    vxorpd xmm9,   xmm9,   xmm9
    vxorpd xmm10,  xmm10,  xmm10
    vxorpd xmm11,  xmm11,  xmm11
loop_head:
    add    rsi,    4
    vmovss xmm13,  dword ptr [r14]
    vmovss xmm14,  dword ptr [r14 + 4]
    vmovss xmm12,  dword ptr [r13]
    vmovss xmm15,  dword ptr [r13 + r10]
    vfmadd231ss    xmm0,   xmm12,  xmm13
    vfmadd231ss    xmm1,   xmm12,  xmm14
    vfmadd231ss    xmm2,   xmm15,  xmm13
    vfmadd231ss    xmm3,   xmm15,  xmm14
    lea    r13,    [r13 + 2 * r10]
    vmovss xmm12,  dword ptr [r13]
    vmovss xmm15,  dword ptr [r13 + r10]
    vfmadd231ss    xmm4,   xmm12,  xmm13
    vfmadd231ss    xmm5,   xmm12,  xmm14
    vfmadd231ss    xmm6,   xmm15,  xmm13
    vfmadd231ss    xmm7,   xmm15,  xmm14
    lea    r13,    [r13 + 2 * r10]
    vmovss xmm12,  dword ptr [r13]
    vmovss xmm15,  dword ptr [r13 + r10]
    vfmadd231ss    xmm8,   xmm12,  xmm13
    vfmadd231ss    xmm9,   xmm12,  xmm14
    vfmadd231ss    xmm10,  xmm15,  xmm13
    vfmadd231ss    xmm11,  xmm15,  xmm14
    mov    r13,    rsi
    add    r14,    8
    sub    ebx,    1
    jne    loop_head
    vmovss dword ptr [r15],        xmm0
    vmovss dword ptr [r15 + 4],    xmm1
    lea    r15,    [r15 + r11]
    vmovss dword ptr [r15],        xmm2
    vmovss dword ptr [r15 + 4],    xmm3
    lea    r15,    [r15 + r11]
    vmovss dword ptr [r15],        xmm4
    vmovss dword ptr [r15 + 4],    xmm5
    lea    r15,    [r15 + r11]
    vmovss dword ptr [r15],        xmm6
    vmovss dword ptr [r15 + 4],    xmm7
    lea    r15,    [r15 + r11]
    vmovss dword ptr [r15],        xmm8
    vmovss dword ptr [r15 + 4],    xmm9
    lea    r15,    [r15 + r11]
    vmovss dword ptr [r15],        xmm10
    vmovss dword ptr [r15 + 4],    xmm11
    add    r15,    r11
    ret
compute_kernel_6x2_f_ endp

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; compute_kernel_4x2_f_(src_matrix1 = R13, src_matrix1_w = EBX, src_matrix1_w_real = R10, ;;
;;                     src_matrix2 = R14, src_matrix2_w_real = R11, dst_matrix = R15)      ;;
;; Uses RSI. Returns R15 = ptr to next kernel in dst_matrix                                ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
compute_kernel_4x2_f_ proc
    mov    rsi,    r13
    vxorpd xmm0,   xmm0,   xmm0
    vxorpd xmm1,   xmm1,   xmm1
    vxorpd xmm2,   xmm2,   xmm2
    vxorpd xmm3,   xmm3,   xmm3
    vxorpd xmm4,   xmm4,   xmm4
    vxorpd xmm5,   xmm5,   xmm5
    vxorpd xmm6,   xmm6,   xmm6
    vxorpd xmm7,   xmm7,   xmm7
loop_head:
    add    rsi,    4
    vmovss xmm9,   dword ptr [r14]
    vmovss xmm10,  dword ptr [r14 + 4]
    vmovss xmm8,   dword ptr [r13]
    vmovss xmm11,  dword ptr [r13 + r10]
    lea    r13,    [r13 + 2 * r10]
    vmovss xmm12,  dword ptr [r13]
    vmovss xmm13,  dword ptr [r13 + r10]
    vfmadd231ss    xmm0,   xmm8,   xmm9
    vfmadd231ss    xmm1,   xmm8,   xmm10
    vfmadd231ss    xmm2,   xmm11,  xmm9
    vfmadd231ss    xmm3,   xmm11,  xmm10
    vfmadd231ss    xmm4,   xmm12,  xmm9
    vfmadd231ss    xmm5,   xmm12,  xmm10
    vfmadd231ss    xmm6,   xmm13,  xmm9
    vfmadd231ss    xmm7,   xmm13,  xmm10
    lea    r14,    [r14 + 8]
    mov    r13,    rsi
    sub    ebx,    1
    jne    loop_head
    vmovss dword ptr [r15],        xmm0
    vmovss dword ptr [r15 + 4],    xmm1
    lea    r15,    [r15 + r11]
    vmovss dword ptr [r15],        xmm2
    vmovss dword ptr [r15 + 4],    xmm3
    lea    r15,    [r15 + r11]
    vmovss dword ptr [r15],        xmm4
    vmovss dword ptr [r15 + 4],    xmm5
    lea    r15,    [r15 + r11]
    vmovss dword ptr [r15],        xmm6
    vmovss dword ptr [r15 + 4],    xmm7
    lea    r15,    [r15 + r11]
    ret
compute_kernel_4x2_f_ endp

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; compute_kernel_2x2_f_(src_matrix1 = R13, src_matrix1_w = EBX, src_matrix1_w_real = R10, ;;
;;                     src_matrix2 = R14, src_matrix2_w_real = R11, dst_matrix = R15)      ;;
;; Returns R15 = ptr to next kernel in dst_matrix                                          ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
compute_kernel_2x2_f_ proc
    vxorpd xmm0,   xmm0,   xmm0
    vxorpd xmm1,   xmm1,   xmm1
    vxorpd xmm2,   xmm2,   xmm2
    vxorpd xmm3,   xmm3,   xmm3
loop_head:
    vmovss xmm5,   dword ptr [r14]
    vmovss xmm6,   dword ptr [r14 + 4]
    vmovss xmm4,   dword ptr [r13]
    vmovss xmm7,   dword ptr [r13 + r10]
    vfmadd231ss    xmm1,   xmm4,  xmm6
    vfmadd231ss    xmm0,   xmm4,  xmm5
    vfmadd231ss    xmm2,   xmm7,  xmm5
    vfmadd231ss    xmm3,   xmm7,  xmm6
    lea    r14,    [r14 + 8]
    add    r13,    4
    sub    ebx,    1
    jne    loop_head
    vmovss dword ptr [r15],        xmm0
    vmovss dword ptr [r15 + 4],    xmm1
    lea    r15,    [r15 + r11]
    vmovss dword ptr [r15],        xmm2
    vmovss dword ptr [r15 + 4],    xmm3
    lea    r15,    [r15 + r11]
    ret
compute_kernel_2x2_f_ endp

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; compute_kernel_1x2_f_(src_matrix1 = R13, src_matrix1_w = EBX, src_matrix1_w_real = R10, ;;
;;                     src_matrix2 = R14, src_matrix2_w_real = R11, dst_matrix = R15)      ;;
;; Returns R15 = ptr to next kernel in dst_matrix                                          ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
compute_kernel_1x2_f_ proc
    vxorpd xmm0,   xmm0,   xmm0
    vxorpd xmm1,   xmm1,   xmm1
loop_head:
    vmovss xmm8,   dword ptr [r13]
    vmovss xmm9,   dword ptr [r14]
    vmovss xmm10,  dword ptr [r14 + 4]
    vfmadd231ss    xmm0,   xmm8,   xmm9
    vfmadd231ss    xmm1,   xmm8,   xmm10
    lea    r14,    [r14 + 8]
    add    r13,    4
    sub    ebx,    1
    jnz    loop_head
    vmovss dword ptr [r15],        xmm0
    vmovss dword ptr [r15 + 4],    xmm1
    ret
compute_kernel_1x2_f_ endp


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; compute_kernel_6x8_f_(src_matrix1 = R13, src_matrix1_w = EBX, src_matrix1_w_real = R10, ;;
;;                     src_matrix2 = R14, src_matrix2_w_real = R11, dst_matrix = R15)      ;;
;; Uses RSI. Returns R15 = ptr to next kernel in dst_matrix                                ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
compute_kernel_6x1_f_ proc
    mov    rsi,    r13
    vxorpd xmm0,   xmm0,   xmm0
    vxorpd xmm1,   xmm1,   xmm1
    vxorpd xmm2,   xmm2,   xmm2
    vxorpd xmm3,   xmm3,   xmm3
    vxorpd xmm4,   xmm4,   xmm4
    vxorpd xmm5,   xmm5,   xmm5
loop_head:
    add    rsi,    4
    vmovss xmm8,   dword ptr [r14]
    vmovss xmm9,   dword ptr [r13]
    vmovss xmm10,  dword ptr [r13 + r10]
    lea    r13,    [r13 + 2 * r10]
    vmovss xmm11,  dword ptr [r13]
    vmovss xmm12,  dword ptr [r13 + r10]
    lea    r13,    [r13 + 2 * r10]
    vmovss xmm13,  dword ptr [r13]
    vmovss xmm14,  dword ptr [r13 + r10]
    vfmadd231ss    xmm0,   xmm9,  xmm8
    vfmadd231ss    xmm1,   xmm10, xmm8
    vfmadd231ss    xmm2,   xmm11, xmm8
    vfmadd231ss    xmm3,   xmm12, xmm8
    vfmadd231ss    xmm4,   xmm13, xmm8
    vfmadd231ss    xmm5,   xmm14, xmm8
    mov    r13,    rsi
    add    r14,    4
    sub    ebx,    1
    jne    loop_head
    vmovss dword ptr [r15],        xmm0
    lea    r15,    [r15 + r11]
    vmovss dword ptr [r15],        xmm1
    lea    r15,    [r15 + r11]
    vmovss dword ptr [r15],        xmm2
    lea    r15,    [r15 + r11]
    vmovss dword ptr [r15],        xmm3
    lea    r15,    [r15 + r11]
    vmovss dword ptr [r15],        xmm4
    lea    r15,    [r15 + r11]
    vmovss dword ptr [r15],        xmm5
    add    r15,    r11
    ret
compute_kernel_6x1_f_ endp

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; compute_kernel_4x1_f_(src_matrix1 = R13, src_matrix1_w = EBX, src_matrix1_w_real = R10, ;;
;;                     src_matrix2 = R14, src_matrix2_w_real = R11, dst_matrix = R15)      ;;
;; Uses RSI. Returns R15 = ptr to next kernel in dst_matrix                                ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
compute_kernel_4x1_f_ proc
    mov    rsi,    r13
    vxorpd xmm0,   xmm0,   xmm0
    vxorpd xmm1,   xmm1,   xmm1
    vxorpd xmm2,   xmm2,   xmm2
    vxorpd xmm3,   xmm3,   xmm3
loop_head:
    add    rsi,    4
    vmovss xmm8,   dword ptr [r14]
    vmovss xmm9,   dword ptr [r13]
    lea    r13,    [r13 + r10]
    vmovss xmm10,  dword ptr [r13]
    vmovss xmm11,  dword ptr [r13 + r10]
    vmovss xmm12,  dword ptr [r13 + 2 * r10]
    vfmadd231ss    xmm0,   xmm9,  xmm8
    vfmadd231ss    xmm1,   xmm10, xmm8
    vfmadd231ss    xmm2,   xmm11, xmm8
    vfmadd231ss    xmm3,   xmm12, xmm8
    lea    r14,    [r14 + 4]
    mov    r13,    rsi
    sub    ebx,    1
    jne    loop_head
    vmovss dword ptr [r15],        xmm0
    lea    r15,    [r15 + r11]
    vmovss dword ptr [r15],        xmm1
    lea    r15,    [r15 + r11]
    vmovss dword ptr [r15],        xmm2
    lea    r15,    [r15 + r11]
    vmovss dword ptr [r15],        xmm3
    lea    r15,    [r15 + r11]
    ret
compute_kernel_4x1_f_ endp

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; compute_kernel_2x1_f_(src_matrix1 = R13, src_matrix1_w = EBX, src_matrix1_w_real = R10, ;;
;;                     src_matrix2 = R14, src_matrix2_w_real = R11, dst_matrix = R15)      ;;
;; Returns R15 = ptr to next kernel in dst_matrix                                          ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
compute_kernel_2x1_f_ proc
    vxorpd xmm0,   xmm0,   xmm0
    vxorpd xmm1,   xmm1,   xmm1
loop_head:
    vmovss xmm8,   dword ptr [r14]
    vmovss xmm9,   dword ptr [r13]
    vmovss xmm10,  dword ptr [r13 + r10]
    vfmadd231ss    xmm0,   xmm9,   xmm8
    vfmadd231ss    xmm1,   xmm10,  xmm8
    lea    r14,    [r14 + 4]
    add    r13,    4
    sub    ebx,    1
    jne    loop_head
    vmovss dword ptr [r15],        xmm0
    lea    r15,    [r15 + r11]
    vmovss dword ptr [r15],        xmm1
    lea    r15,    [r15 + r11]
    ret
compute_kernel_2x1_f_ endp

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; compute_kernel_1x1_f_(src_matrix1 = R13, src_matrix1_w = EBX, src_matrix1_w_real = R10, ;;
;;                     src_matrix2 = R14, src_matrix2_w_real = R11, dst_matrix = R15)      ;;
;; Returns R15 = ptr to next kernel in dst_matrix                                          ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
compute_kernel_1x1_f_ proc
    vxorpd xmm0,   xmm0,   xmm0
loop_head:
    vmovss xmm9,   dword ptr [r14]
    vmovss xmm8,   dword ptr [r13]
    vfmadd231ss    xmm0,   xmm8,  xmm9
    lea    r14,    [r14 + 4]
    add    r13,    4
    sub    ebx,    1
    jne    loop_head
    vmovss dword ptr [r15],        xmm0
    ret
compute_kernel_1x1_f_ endp

end