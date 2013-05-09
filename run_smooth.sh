for i in 512 1024 2000 4000; do
		echo "#define X_WIDTH ${i}" > defines.h
		echo "#define Y_WIDTH ${i}" >> defines.h
		echo "#define BLOCK_WIDTH 1" >> defines.h
		echo "#define NUM_THREADS 1" >> defines.h
		echo "#define SMOOTHING 1" >> defines.h

		make clean &> /dev/null
		make &> /dev/null
		./cpu
done
