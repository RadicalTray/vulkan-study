CFLAGS = -std=c++17 -O2
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

build/VulkanTest: main.cpp
	mkdir build
	g++ $(CFLAGS) -o build/VulkanTest main.cpp $(LDFLAGS)

.PHONY: test clean
test: build/VulkanTest
	./build/VulkanTest

clean:
	rm -r build
