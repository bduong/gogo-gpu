CC 		:= gcc
CPU_FLAGS 	:= -lpthread 

cpu: cpu.c
	$(CC) -o $@ $< $(CPU_FLAGS)

clean:
	rm -fr cpu
