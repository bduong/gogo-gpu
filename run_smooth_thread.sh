for k in 2 4 8 16; do
	echo "Threads: ${k}"
	for i in 512 1024 2000 4000 8000; do
			echo "#define X_WIDTH ${i}" > defines.h
			echo "#define Y_WIDTH ${i}" >> defines.h
			echo "#define BLOCK_WIDTH 1" >> defines.h
			echo "#define NUM_THREADS ${k}" >> defines.h
			echo "#define THREAD_SMOOTHING 1" >> defines.h

			make clean &> /dev/null
			make cpu &> /dev/null
			./cpu
	done
done
