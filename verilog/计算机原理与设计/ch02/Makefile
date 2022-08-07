BINS= gates7_structural_test gates7_dataflow_test
all: $(BINS)

VERILOG=iverilog

gates7_structural_test: gates7_structural_test.v
	$(VERILOG) -o $@ $<

gates7_dataflow_test: gates7_dataflow_test.v
	$(VERILOG) -o $@ $<

.PHONY:clean
clean:
	rm $(BINS) *.vcd
