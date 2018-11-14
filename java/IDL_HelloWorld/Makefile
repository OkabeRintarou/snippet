all:HelloApp HelloServer.class HelloClient.class 

HelloApp:Hello.idl
	idlj -fall Hello.idl

HelloServer.class:HelloServer.java
	javac HelloServer.java
HelloClient.class:HelloClient.java
	javac HelloClient.java

.PHONY:clean
clean:
	-rm -rf HelloApp
	-rm *.class
