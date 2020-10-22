# Architetture Avanzate - AA 2019-2020

## Sparse Matrix Transposition for GPUs

## Usage

The following steps are for successfully build and run the project:

1. Clone this repository or copy the [code/] folder into a device with CUDA compiler installed:
```
git clone https://github.com/michelepenzo/architetture-avanzate.git
cd code/
```

2. Create a *build* folder:
```
mkdir build
```

3. Run make and start compilation:

```
make
```

4. The compilation process produces two executable files: 
	- run *app* to start the CUDA implementation:
	```
	make run
	```

	- run *test_app* to start the tests:
	```
	make test
	```

The Cuda implementation (*app*) runs multiple tests in sequence with different parameters in input to produce a log file: *timing_analysis.csv*.


## Sources

The implementation of can be found in [/code](./code) directory. 


## Documentation

The report of __Sparse Matrix Transposition for GPUs__ can be found in [/doc/report_aa.pdf](./doc/report_aa.pdf). 