EXE 			= app
EXE_TEST    	= test_app
LOCAL_INCLUDE	= ./include
LOCAL_SRC		= ./src
LOCAL_TEST		= ./test
MAT_DIR			= /matrices/
TARGET_CSV      = timing_analysis.csv
LOCAL_BUILD 	= build
OBJECT_LIST     = $(LOCAL_BUILD)/transposers.o $(LOCAL_BUILD)/matrix.o $(LOCAL_BUILD)/indexes_to_pointers.o $(LOCAL_BUILD)/pointers_to_indexes.o $(LOCAL_BUILD)/scan.o $(LOCAL_BUILD)/sort.o $(LOCAL_BUILD)/segsort.o 

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
$(EXE): $(LOCAL_BUILD)/main.o $(LOCAL_BUILD)/test_main.o $(OBJECT_LIST)
	nvcc $(GENCODE_FLAGS) $(LOCAL_BUILD)/main.o      $(OBJECT_LIST) -o $(EXE)
	nvcc $(GENCODE_FLAGS) $(LOCAL_BUILD)/test_main.o $(OBJECT_LIST) -o $(EXE_TEST)

$(LOCAL_BUILD)/test_main.o: $(LOCAL_TEST)/test_main.cu
	nvcc $(GENCODE_FLAGS) -I$(LOCAL_INCLUDE) -I$(LOCAL_TEST) -c $< -o $@

$(LOCAL_BUILD)/main.o: $(LOCAL_SRC)/main.cu
	nvcc $(GENCODE_FLAGS) -I$(LOCAL_INCLUDE) -c $< -o $@

$(LOCAL_BUILD)/matrix.o: $(LOCAL_SRC)/matrix.cu
	nvcc $(GENCODE_FLAGS) -I$(LOCAL_INCLUDE) -c $< -o $@

$(LOCAL_BUILD)/indexes_to_pointers.o: $(LOCAL_SRC)/indexes_to_pointers.cu
	nvcc $(GENCODE_FLAGS) -I$(LOCAL_INCLUDE) -c $< -o $@

$(LOCAL_BUILD)/pointers_to_indexes.o: $(LOCAL_SRC)/pointers_to_indexes.cu
	nvcc $(GENCODE_FLAGS) -I$(LOCAL_INCLUDE) -c $< -o $@

$(LOCAL_BUILD)/segsort.o: $(LOCAL_SRC)/segsort.cu
	nvcc $(GENCODE_FLAGS) -I$(LOCAL_INCLUDE) -c $< -o $@

$(LOCAL_BUILD)/sort.o: $(LOCAL_SRC)/sort.cu
	nvcc $(GENCODE_FLAGS) -I$(LOCAL_INCLUDE) -c $< -o $@

$(LOCAL_BUILD)/scan.o: $(LOCAL_SRC)/scan.cu
	nvcc $(GENCODE_FLAGS) -I$(LOCAL_INCLUDE) -c $< -o $@

$(LOCAL_BUILD)/transposers.o: $(LOCAL_SRC)/transposers.cu
	nvcc $(GENCODE_FLAGS) -I$(LOCAL_INCLUDE) -c $< -o $@


# The .PHONY rule keeps make from doing something with a file named clean.
.PHONY: clean test run

clean:
	@rm -f $(EXE) $(EXE_TEST) $(LOCAL_BUILD)/*

run:
	@rm -f $(TARGET_CSV)
	@echo "NAME; M; N; NNZ; SERIAL TIME; SCANTRANS TIME; MERGETRANS TIME; CUSPARSE1 TIME; CUSPARSE2 TIME\n" >> $(TARGET_CSV)
	@./$(EXE) $(MAT_DIR)test1.mtx    	>> $(TARGET_CSV)
	#@./$(EXE) $(MAT_DIR)ASIC_680k.mtx 	>> $(TARGET_CSV)
	#@./$(EXE) $(MAT_DIR)memchip.mtx 	>> $(TARGET_CSV) 
	#@./$(EXE) $(MAT_DIR)FullChip.mtx 	>> $(TARGET_CSV)
	#@./$(EXE) $(MAT_DIR)cage14.mtx 	>> $(TARGET_CSV) 
	#@./$(EXE) $(MAT_DIR)cant.mtx 		>> $(TARGET_CSV)
	#@./$(EXE) $(MAT_DIR)circuit5M.mtx 	>> $(TARGET_CSV)
	#@./$(EXE) $(MAT_DIR)para-4.mtx 	>> $(TARGET_CSV)
	#@./$(EXE) $(MAT_DIR)rajat21.mtx 	>> $(TARGET_CSV)
	#@./$(EXE) $(MAT_DIR)rajat29.mtx 	>> $(TARGET_CSV)
	#@./$(EXE) $(MAT_DIR)sme3Dc.mtx 	>> $(TARGET_CSV)
	#@./$(EXE) $(MAT_DIR)stomach.mtx 	>> $(TARGET_CSV)
	#@./$(EXE) $(MAT_DIR)torso1.mtx 	>> $(TARGET_CSV)
	#@./$(EXE) $(MAT_DIR)venkat01.mtx 	>> $(TARGET_CSV)
	#@./$(EXE) $(MAT_DIR)web-Google.mtx >> $(TARGET_CSV)
	#@./$(EXE) $(MAT_DIR)webbase-1M.mtx >> $(TARGET_CSV)  

test:
	@./$(EXE_TEST)
