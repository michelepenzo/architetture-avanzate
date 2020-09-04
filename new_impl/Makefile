# Makefile based on nvcc version

EXE 			= run_transposer_app
LOCAL_INCLUDE	= ./include
LOCAL_SRC		= ./src
LOCAL_BUILD 	= build
GENCODE_FLAGS   = -m64 -gencode arch=compute_30,code=sm_30 -lcudart -lcusparse -lcublas

NVCC = ${NVCC_VERSION}

# $@ is the target (eg: main.o)
# $< is the source (eg: main.cu)
$(EXE): $(LOCAL_BUILD)/main.o
# CUDA server
ifeq ($(NVCC), 10)
	nvcc -lcusparse -o $(EXE) $(LOCAL_BUILD)/main.o
endif

# CUDA incud
ifeq ($(NVCC), 9)
	nvcc -ccbin g++-6 $(GENCODE_FLAGS) -o $(EXE) $(LOCAL_BUILD)/main.o
endif

$(LOCAL_BUILD)/main.o: $(LOCAL_SRC)/main.cu
# CUDA server
ifeq ($(NVCC), 10)
	nvcc -lcusparse -I$(LOCAL_INCLUDE) -c $< -o $@
endif

# CUDA incud
ifeq ($(NVCC), 9)
	nvcc -ccbin g++-6 $(GENCODE_FLAGS) -I$(LOCAL_INCLUDE) -c $< -o $@
endif


# The .PHONY rule keeps make from doing something with a file named clean.
.PHONY: clean test

clean:
	rm -f $(EXE) $(LOCAL_BUILD)/main.o

test:
	@echo "Starting test application...\n\n"
	@./$(EXE)