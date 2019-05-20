mov ax, 0x07c0
mov ds, ax
mov ax, 0xb800
mov es, ax

cld
mov si, message_start
mov di, 0
mov cx, message_end - message_start
rep movsb

mov byte [es:0x00], 'H'
mov byte [es:0x01], 0x07

jmp $

message_start:
db 'H', 0x07, 'e', 0x07, 'l', 0x07, 'l', 0x07, 'o', 0x07, ' ', 0x07,\
	 'W', 0x07, 'o', 0x07, 'r', 0x07, 'l', 0x07, 'd', 0x07, '!', 0x07
message_end:

times 510-($-$$) db 0
db 0x55
db 0xAA
