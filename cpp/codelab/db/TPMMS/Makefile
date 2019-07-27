all:sort gen_record
sort:sort.cpp record.cpp timer.cpp
	g++ -O2 -std=c++14 -o sort sort.cpp timer.cpp record.cpp -lgflags -lglog

gen_record:gen_record.cpp record.cpp
	g++ -O2 -std=c++14 -o gen_record gen_record.cpp record.cpp -lgflags

.PHONY:clean
clean:
	rm -rf gen_record sort
