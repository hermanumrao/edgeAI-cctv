# Makefile for camera_streaming_project

# Paths
SRC_DIR := src
BUILD_DIR := build

# Compiler
CXX := g++

# Source and target
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
TARGET := $(BUILD_DIR)/camera_stream

# Flags
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS := $(shell pkg-config --libs opencv4)

CXXFLAGS_DEBUG := -g -Wall -pthread $(OPENCV_CFLAGS)
CXXFLAGS_OPT := -O3 -Wall -pthread $(OPENCV_CFLAGS)
CXXFLAGS_ARM := -O3 -march=armv8-a -mtune=cortex-a53 -Wall -pthread $(OPENCV_CFLAGS)

LDFLAGS := $(OPENCV_LIBS)

# Create build directory if not exists
$(shell mkdir -p $(BUILD_DIR))

.PHONY: all debug release arm clean

all: release

debug:
	$(CXX) $(SRCS) -o $(TARGET)_debug $(CXXFLAGS_DEBUG) $(LDFLAGS)

release:
	$(CXX) $(SRCS) -o $(TARGET) $(CXXFLAGS_OPT) $(LDFLAGS)

arm:
	aarch64-linux-gnu-g++ $(SRCS) -o $(TARGET)_arm $(CXXFLAGS_ARM) $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

