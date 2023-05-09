BINS=importer exporter
all:$(BINS)

importer:importer.cc util.cc
	g++ -I/usr/include/libdrm -o $@ importer.cc util.cc -ldrm -ldrm_amdgpu -lglfw -lGL -lGLEW

exporter:exporter.cc util.cc
	g++ -I/usr/include/libdrm -o $@ exporter.cc util.cc -ldrm -ldrm_amdgpu

clean:
	rm $(BINS)
