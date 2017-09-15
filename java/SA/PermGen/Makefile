JAVAC=javac

all:Foo.class TestPrintPSPermGen.class

Foo.class:Foo.java
	${JAVAC} $<
TestPrintPSPermGen.class:TestPrintPSPermGen.java
	${JAVAC} -cp .:${JAVA_HOME}/lib/sa-jdi.jar $<

.PHONY:clean
clean:
	-rm Foo.class TestPrintPSPermGen.class
.PHONY:run
run:
	java -server -XX:+UseParallelGC -XX:OnOutOfMemoryError='java -cp ${JAVA_HOME}/lib/sa-jdi.jar:. TestPrintPSPermGen %p >foo.txt' -Xms1g -Xmx1g Foo
