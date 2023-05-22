BINS=draw_triangle vert.spv frag.spv


all:$(BINS)
draw_triangle:draw_triangle.cc
	g++ -g -std=c++20 -o $@ $< -lvulkan -lglfw

vert.spv:shaders/shader.vert
	glslangValidator -S vert --target-env vulkan1.0 $< -o $@

frag.spv:shaders/shader.frag
	glslangValidator -S frag --target-env vulkan1.0 $< -o $@

clean:
	rm $(BINS)
