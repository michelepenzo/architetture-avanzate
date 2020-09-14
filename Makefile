EXE 			= run_transposer_app
LOCAL_INCLUDE	= ./include
LOCAL_SRC		= ./src
LOCAL_BUILD 	= build

# personalize code considering CUDA version 9 or 11
NVCC_VERSION    = ${shell nvcc -V | grep -oh "release [0-9]*\." | grep -oh "[0-9]*"}
GENCODE_CUDA9   = -ccbin g++-6 -m64 -gencode arch=compute_30,code=sm_30 -lcudart -lcusparse -lcublas
GENCODE_CUDA11  = -lcudart -lcusparse -lcublas
ifeq ($(NVCC_VERSION), 9)
	GENCODE_FLAGS = -Xcudafe "--diag_suppress=code_is_unreachable" $(GENCODE_CUDA9)
else
	GENCODE_FLAGS = -Xcudafe "--diag_suppress=code_is_unreachable" $(GENCODE_CUDA11)
endif

# This is the whole executable
$(EXE): $(LOCAL_BUILD)/main.o $(LOCAL_BUILD)/ScanTransposer.o $(LOCAL_BUILD)/MergeTransposer.o $(LOCAL_BUILD)/fort.o $(LOCAL_BUILD)/prefix_scan.o
	nvcc $(GENCODE_FLAGS) -o $(EXE) $(LOCAL_BUILD)/main.o $(LOCAL_BUILD)/ScanTransposer.o $(LOCAL_BUILD)/fort.o $(LOCAL_BUILD)/prefix_scan.o

# Just main.cu. $@ is the target (eg: main.o) and $< is the source (eg: main.cu)
$(LOCAL_BUILD)/main.o: $(LOCAL_SRC)/main.cu
	nvcc $(GENCODE_FLAGS) -I$(LOCAL_INCLUDE) -c $< -o $@

$(LOCAL_BUILD)/ScanTransposer.o: $(LOCAL_SRC)/transposers/ScanTransposer.cu
	nvcc $(GENCODE_FLAGS) -I$(LOCAL_INCLUDE) -c $< -o $@

$(LOCAL_BUILD)/MergeTransposer.o: $(LOCAL_SRC)/transposers/MergeTransposer.cu
	nvcc $(GENCODE_FLAGS) -I$(LOCAL_INCLUDE) -c $< -o $@

$(LOCAL_BUILD)/prefix_scan.o: $(LOCAL_SRC)/cuda_utils/prefix_scan.cu
	nvcc $(GENCODE_FLAGS) -I$(LOCAL_INCLUDE) -c $< -o $@

$(LOCAL_BUILD)/fort.o: $(LOCAL_SRC)/libfort/fort.c
	gcc -I$(LOCAL_INCLUDE) -c $< -o $@



# The .PHONY rule keeps make from doing something with a file named clean.
.PHONY: clean test

clean:
	@rm -f $(EXE) $(LOCAL_BUILD)/main.o $(LOCAL_BUILD)/MergeTransposer.o $(LOCAL_BUILD)/ScanTransposer.o $(LOCAL_BUILD)/fort.o $(LOCAL_BUILD)/prefix_scan.o

test:
	@echo "Starting test application...\n\n"
	@./$(EXE)
