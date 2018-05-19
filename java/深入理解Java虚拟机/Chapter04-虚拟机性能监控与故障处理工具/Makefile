all:


Bar:Bar.class
	java -XX:+UnlockDiagnosticVMOptions -XX:+PrintAssembly -Xcomp -XX:CompileCommand=dontinline,*Bar.sum -XX:CompileCommand=compileonly,*Bar.sum $@

MonitoringTest:MonitoringTest.class
	java -Xms100m -Xmx100M -XX:+UseSerialGC $@
