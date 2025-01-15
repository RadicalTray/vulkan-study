TARGET_EXEC := VulkanTest

BUILD_DIR := build
SRC_DIRS := src

SRCS := $(shell find $(SRC_DIRS) -name '*.cpp' -or -name '*.c' -or -name '*.s')
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

SHADERS_DIR := shaders
SHADERS := $(shell find $(SHADERS_DIR) -name 'shader.*')
TARGET_SHADERS := $(SHADERS:$(SHADERS_DIR)/shader.%=$(BUILD_DIR)/shaders/%.spv)

CXX := clang++

INC_DIRS := $(shell find $(SRC_DIRS) -type d) /usr/include/stb tinyobjloader
INC_FLAGS := $(addprefix -I,$(INC_DIRS))
CPPFLAGS := $(INC_FLAGS) -MMD -MP -std=c++23 -Wall -g

LDFLAGS := $(shell pkg-config --cflags --libs glfw3 vulkan)

all: $(BUILD_DIR)/$(TARGET_EXEC) shaders

$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.c.o: %.c
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

.PHONY: shaders
shaders: $(TARGET_SHADERS)

$(BUILD_DIR)/shaders/%.spv: $(SHADERS_DIR)/shader.%
	mkdir -p $(dir $@)
	glslc $< -o $@

.PHONY: test
test: all
	./build/VulkanTest

.PHONY: clean
clean:
	rm -r $(BUILD_DIR)

-include $(DEPS)
