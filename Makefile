#----------------complier configurations-------------------------
# _SUPPORT_VECCHIA_?=TRUE
_USE_MAGMA_?=TRUE
#specify cuda directory
_CUDA_ROOT_=$(CUDA_HOME)
# _CUDA_ARCH_ ?= 70
CUDA_ROOT=$(_CUDA_ROOT_)

#----------------compliers -------------------------
# Default to Cray system unless specified otherwise
SYSTEM ?= OPENMPI
# SYSTEM options: CRAY, OPENMPI

ifeq ($(SYSTEM),CRAY)
  CXX = CC
else ifeq ($(SYSTEM),OPENMPI)
  CXX = mpic++
else
  $(error Unknown system type: $(SYSTEM). Use CRAY or OPENMPI)
endif
#NVCC=$(_CUDA_ROOT_)/bin/nvcc
NVCC=nvcc

NVOPTS = -ccbin $(CXX) --compiler-options -uns --extended-lambda -allow-unsupported-compiler
CXXFLAGS = -O2 -Wall -std=c++17 -fopenmp -Wsign-compare -Wno-sign-compare -Wunknown-pragmas -Wunused-variable

ifeq ($(SYSTEM),CRAY)
  CXXFLAGS += -lstdc++fs
endif

ifdef _DEBUG_
  CXXFLAGS += -g -Xcompiler -rdynamic
  NVOPTS += -G -g -lineinfo
else
  CXXFLAGS += -O3
  NVOPTS += -O3
endif

ifdef _USE_MAGMA_
  CXXFLAGS += -DUSE_MAGMA
  _MAGMA_ROOT_?=$(HOME)/dev/magma-2.7.2
  NVOPTS += -DUSE_MAGMA
endif

#---------------- directory -------------------------
# object and bin files
OBJ_DIR=./obj
BIN_DIR=./bin
SRC_DIR=./src
$(shell mkdir -p $(OBJ_DIR))
$(shell mkdir -p $(BIN_DIR))


INCLUDES=
INCLUDES+= -I.
INCLUDES+= -I./include
INCLUDES+= -I${CUDA_ROOT}/include
INCLUDES+= -I${NLOPT_ROOT}/include
# INCLUDES+= -I${GSL_ROOT}/include // used for matern kernel, bessel function

ifeq ($(SYSTEM),CRAY)
  LIB_PATH=
  LIB_PATH+= -L${CUDA_ROOT}/targets/sbsa-linux/lib  #GH 200 system
  LIB_PATH+= -L${NLOPT_ROOT}/lib64
else
  LIB_PATH=
  LIB_PATH+= -L${CUDA_ROOT}/lib64  # A100 system
  LIB_PATH+= -L${NLOPT_ROOT}/lib # A100 system
endif
# LIB_PATH+= -L${GSL_ROOT}/lib  // used for matern kernel, bessel function

ifdef _USE_MAGMA_
	LIB_PATH+= -L${_MAGMA_ROOT_}/lib
	INCLUDES+= -I$(_MAGMA_ROOT_)/include
endif

# libraries to link against
LIB= -lm 
LIB+= -lnlopt  
# LIB+= -lgsl // used for matern kernel, bessel function
ifdef _USE_MAGMA_
	LIB+= -lmagma -lcusparse
endif
LIB+= -lcudart -lcublas
LIB+= -lgomp
LIB+= -lstdc++

#---------------- make -------------------------

TARGET = $(BIN_DIR)/dbv
OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/random_points.o  $(OBJ_DIR)/block_info.o $(OBJ_DIR)/distance_calc.o $(OBJ_DIR)/gpu_operations.o $(OBJ_DIR)/gpu_covariance.o $(OBJ_DIR)/vecchia_helper.o $(OBJ_DIR)/prediction.o

# Add input_parser.h as a dependency for all object files
DEPS=
DEPS+=include/input_parser.h
DEPS+=include/gpu_covariance.h
DEPS+=include/vecchia_helper.h
DEPS+=include/prediction.h
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(OBJS) $(LIB) $(LIB_PATH)

$(OBJ_DIR)/main.o: main.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVOPTS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJ_DIR)/*.o $(TARGET)

.PHONY: clean
