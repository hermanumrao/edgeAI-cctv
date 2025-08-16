# ==== Config ====
CXX      := g++
CXXFLAGS := -std=c++17 -Wall -O2 -pthread
PKGCONF  := $(shell pkg-config --cflags --libs opencv4)
INCLUDES := -Iinclude
SRC_DIR  := src
BUILD_DIR:= build
TARGET   := camdash

# ==== Source files ====
SRC := $(wildcard $(SRC_DIR)/*.cpp)
OBJ := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRC))
DEP := $(OBJ:.o=.d)

# ==== Default target ====
all: $(TARGET)

# ==== Build target ====
$(TARGET): $(OBJ)
	@echo "Linking $@"
	$(CXX) $(OBJ) $(CXXFLAGS) $(PKGCONF) -o $@

# ==== Compile rule ====
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	@echo "Compiling $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

# ==== Clean ====
clean:
	@echo "Cleaning..."
	@rm -rf $(BUILD_DIR) $(TARGET)

# ==== Include dependencies ====
-include $(DEP)

.PHONY: all clean

