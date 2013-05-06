CC 		:= gcc
CPU_FLAGS 	:= -lpthread 

cpu: cpu.c
	$(CC) -o $@ $< $(CPU_FLAGS) -I$(CURDIR)

clean:
	rm -fr cpu
