all:Main.class run

Main.class:Main.c Main.java
	javac Main.java
	javah -jni Main
	gcc -shared -fpic -o libMain.so -I${JAVA_HOME}/include \
	-I${JAVA_HOME}/include/linux Main.c

run:Main.class
	java -Djava.library.path=. Main
.PHONY:clean
clean:
	-rm Main.class libMain.so Main.h
