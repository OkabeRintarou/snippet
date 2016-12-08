include ../Makefile.header

rep1:rep1.s
	$(AS) $(ASFLAGS) -o rep1.o rep1.s
	$(LD) -o rep1 rep1.o
clean:
	-rm rep1 rep1.o
