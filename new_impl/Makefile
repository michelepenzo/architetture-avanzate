EXE 			= run_transposer_app
CUDA_INCLUDE 	= /usr/local/cuda/include
CUDA_LIB64		= /usr/local/cuda/lib64
CUDA_LIBRARIES 	= -lcudart -lcusparse -fopenmp
LOCAL_INCLUDE	= ./include
LOCAL_SRC		= ./src
LOCAL_BUILD 	= build

# $@ is the target (eg: main.o)
# $< is the source (eg: main.cu)

$(EXE): $(LOCAL_BUILD)/main.o
	g++ -o $(EXE) $(LOCAL_BUILD)/main.o -L$(CUDA_LIB64) $(CUDA_LIBRARIES)

$(LOCAL_BUILD)/main.o: $(LOCAL_SRC)/main.cu
	nvcc -w -std=c++11 -I$(CUDA_INCLUDE) -I$(LOCAL_INCLUDE) -c $< -o $@


# The .PHONY rule keeps make from doing something with a file named clean.
.PHONY: clean test

clean:
	rm -f $(EXE) $(LOCAL_BUILD)/main.o

test:
	@echo "Starting test application...\n\n"
	@./$(EXE)