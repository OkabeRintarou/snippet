all: example1 example1.beam

example1.beam:example1.erl
	erlc example1.erl

example1: erl_comm.o example1.o example1_driver.o
	gcc -o example1 erl_comm.o example1.o example1_driver.o

erl_comm.o:erl_comm.c
	gcc -c -o $@ $<

example1.o:example1.c
	gcc -c -o $@ $<

example1_driver.o:example1_driver.c
	gcc -c -o $@ $<

.PHONY:clean
clean:
	@rm example1 *.o
