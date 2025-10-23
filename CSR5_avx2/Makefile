#compilers
CC=g++

#GLOBAL_PARAMETERS
VALUE_TYPE = double
NUM_RUN = 100

#ENVIRONMENT_PARAMETERS
#OPRIONS=-xCORE-AVX2 -opt-prefetch=3 #icc/icpc option
OPRIONS=-mavx -mavx2 -march=native -ffast-math#c++ option
# Split PAPI include and library flags. Place library flags after objects/sources when linking.
PAPI_INC=-I/home/min/a/das160/papi-install/include
PAPI_LIB=-L/home/min/a/das160/papi-install/lib -lpapi
#backup
#$(CC) -xCORE-AVX2 -opt-prefetch=3 -Wno-deprecated-writable-strings -fopenmp -O3 main.cpp -o spmv -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN)
avx2:
	$(CC) $(OPRIONS) -O3 main.cpp -o spmv_time -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN)
	# Note: put library flags (PAPI_LIB) after the source file so the linker can resolve symbols
	$(CC) $(OPRIONS) -O3 main2.cpp -o spmv_branch -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN) $(PAPI_LIB) $(PAPI_INC)
