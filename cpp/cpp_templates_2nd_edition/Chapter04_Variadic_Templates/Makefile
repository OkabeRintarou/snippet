targets=varprint1 varprint2 foldtraverse addspace varusing

all:$(targets)

varprint1:varprint1.cpp
	clang++ -std=c++11 -o $@ $<

varprint2:varprint2.cpp
	clang++ -std=c++11 -o $@ $<

foldtraverse:foldtraverse.cpp
	clang++ -std=c++17 -o $@ $<

addspace:addspace.cpp	
	clang++ -std=c++17 -o $@ $<

varusing:varusing.cpp
	clang++ -std=c++17 -o $@ $<

.PHONY:clean
clean:
	-rm $(targets)
