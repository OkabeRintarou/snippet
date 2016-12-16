# rep 指令本身不执行什么操作,它用于按照特定次数重复执行字符串指令,
# 重复次数由 ECX 寄存器中的值控制.

.section .data
value1:
    .ascii "This is a test string.\n"
.section .bss
    .lcomm output,23

.section .text
.global _start

_start:
    nop
    leal value1,%esi
    leal output,%edi
    movl $23,%ecx
    cld
    rep  movsb

    mov $1,%eax
    mov $0,%ebx
    int $0x80

# 单步调试时,当执行完 rep movsb 指令后使用 x/s &output GDB 查看&output 处的值等于 "This is a test string.\n"
