JAVA=java
JAVAC=javac

all:Main.class Foo.class KlassKicker.class

Foo.class:Foo.java
	${JAVAC} $<
Main.class:Main.java
	${JAVAC} $<
KlassKicker.class:KlassKicker.java
	${JAVAC} -cp .:${JAVA_HOME}/lib/sa-jdi.jar $<

.PHONY:clean run
clean:
	-rm *.class
run:
	java -cp .:${JAVA_HOME}/lib/sa-jdi.jar KlassKicker
