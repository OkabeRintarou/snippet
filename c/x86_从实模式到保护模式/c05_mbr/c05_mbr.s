org 0x7c00
mov ax, 0xb800
mov es, ax

mov byte [es:0x00], 'H'
mov byte [es:0x01], 0x07
mov byte [es:0x02], 'e'
mov byte [es:0x03], 0x07
mov byte [es:0x04], 'l'
mov byte [es:0x05], 0x07
mov byte [es:0x06], 'l'
mov byte [es:0x07], 0x07
mov byte [es:0x08], 'o'
mov byte [es:0x09], 0x07
mov byte [es:0x0a], ' '
mov byte [es:0x0b], 0x07
mov byte [es:0x0c], 'W'
mov byte [es:0x0d], 0x07
mov byte [es:0x0e], 'o'
mov byte [es:0x0f], 0x07
mov byte [es:0x10], 'r'
mov byte [es:0x11], 0x07
mov byte [es:0x12], 'l'
mov byte [es:0x13], 0x07
mov byte [es:0x14], 'd'
mov byte [es:0x15], 0x07
mov byte [es:0x16], '!'
mov byte [es:0x17], 0x07

jmp $

times 510-($-$$) db 0
db 0x55
db 0xAA
