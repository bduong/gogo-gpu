for i in 512 1024 2000 4000 8000; do
		echo "#define SM_ARR_LEN ${i}" > defines.h
		echo "#define FACTOR 1" >> defines.h
		
		rm -fr gpu
		nvcc -o gpu project.cu -I $CUDA_HOME/samples/common/inc &> /dev/null
		./gpu
done
