all:

HeapOOM:HeapOOM.class
	java -Xms20m -Xmx20m -XX:+HeapDumpOnOutOfMemoryError $@
JavaVMStackSOF:JavaVMStackSOF.class
	java -Xss256k $@
JavaVMStackOOM:JavaVMStackOOM.class
	java -Xss2M $@
RuntimeConstantPoolOOM:RuntimeConstantPoolOOM.class
	java -XX:PermSize=10M -XX:MaxPermSize=10M $@
DirectMemoryOOM:DirectMemoryOOM.class
	java -Xmx20M -XX:MaxDirectMemorySize=10M $@
.PHONY:clean
clean:
	-rm *.class
