CXX = mpic++
CXXFLAGS = -O2 -Wall -std=c++17 -fopenmp

# object and bin files
OBJ_DIR=./obj
BIN_DIR=./bin
$(shell mkdir -p $(OBJ_DIR))
$(shell mkdir -p $(BIN_DIR))

TARGET = generate_points
OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/random_points.o $(OBJ_DIR)/input_parser.o $(OBJ_DIR)/block_info.o $(OBJ_DIR)/distance_calc.o 

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

$(OBJ_DIR)/main.o: main.cpp random_points.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/random_points.o: random_points.cpp random_points.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/input_parser.o: input_parser.cpp random_points.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/block_info.o: block_info.cpp block_info.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/distance_calc.o: distance_calc.cpp distance_calc.h
	$(CXX) $(CXXFLAGS) -Wsign-compare -c $< -o $@

clean:
	rm -f $(OBJ_DIR)/*.o $(TARGET)

.PHONY: clean
