EXE 			= run_transposer_app
LOCAL_INCLUDE	= ./include
LOCAL_SRC		= ./src
LOCAL_BUILD 	= build

# personalize code considering CUDA version 9 or 11
NVCC_VERSION    = ${shell nvcc -V | grep -oh "release [0-9]*\." | grep -oh "[0-9]*"}
GENCODE_CUDA9   = -ccbin g++-6 -m64 -gencode arch=compute_30,code=sm_30 -lcudart -lcusparse -lcublas
GENCODE_CUDA11  = -lcudart -lcusparse -lcublas
ifeq ($(NVCC_VERSION), 9)
	GENCODE_FLAGS = -Xcudafe "--diag_suppress=code_is_unreachable --diag_suppress=expr_has_no_effect" $(GENCODE_CUDA9)
else
	GENCODE_FLAGS = -Xcudafe "--diag_suppress=code_is_unreachable --diag_suppress=expr_has_no_effect" $(GENCODE_CUDA11)
endif

# This is the whole executable
$(EXE): $(LOCAL_BUILD)/main.o $(LOCAL_BUILD)/transposer.o $(LOCAL_BUILD)/sort.o $(LOCAL_BUILD)/segsort.o $(LOCAL_BUILD)/segmerge_step.o $(LOCAL_BUILD)/segmerge_sm_step.o
	nvcc $(GENCODE_FLAGS) -o $(EXE) $(LOCAL_BUILD)/main.o $(LOCAL_BUILD)/sort.o $(LOCAL_BUILD)/transposer.o $(LOCAL_BUILD)/segsort.o $(LOCAL_BUILD)/segmerge_step.o $(LOCAL_BUILD)/segmerge_sm_step.o

$(LOCAL_BUILD)/main.o: $(LOCAL_SRC)/main.cu
	nvcc $(GENCODE_FLAGS) -I$(LOCAL_INCLUDE) -c $< -o $@

$(LOCAL_BUILD)/transposer.o: $(LOCAL_SRC)/transposer.cu
	nvcc $(GENCODE_FLAGS) -I$(LOCAL_INCLUDE) -c $< -o $@

$(LOCAL_BUILD)/segsort.o: $(LOCAL_SRC)/segsort.cu
	nvcc $(GENCODE_FLAGS) -I$(LOCAL_INCLUDE) -c $< -o $@

$(LOCAL_BUILD)/segmerge_step.o: $(LOCAL_SRC)/segmerge_step.cu
	nvcc $(GENCODE_FLAGS) -I$(LOCAL_INCLUDE) -c $< -o $@

$(LOCAL_BUILD)/segmerge_sm_step.o: $(LOCAL_SRC)/segmerge_sm_step.cu
	nvcc $(GENCODE_FLAGS) -I$(LOCAL_INCLUDE) -c $< -o $@

$(LOCAL_BUILD)/sort.o: $(LOCAL_SRC)/sort.cu
	nvcc $(GENCODE_FLAGS) -I$(LOCAL_INCLUDE) -c $< -o $@



# The .PHONY rule keeps make from doing something with a file named clean.
.PHONY: clean test

clean:
	@rm -f $(EXE) $(LOCAL_BUILD)/*

test:
	@echo "Starting test application...\n\n"
	@./$(EXE)
