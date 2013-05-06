CC 		:= gcc
platform=$(shell uname)

ifeq ($(platform),Darwin)
CPU_FLAGS 	:= -lpthread 
else
CPU_FLAGS       := -lpthread -lrt
endif

cpu: cpu.c
	$(CC) -o $@ $< $(CPU_FLAGS) -I$(CURDIR)

cpu_thread: cpu_test_thread.c
	$(CC) -o $@ $< $(CPU_FLAGS) -I$(CURDIR)

clean:
	rm -fr cpu cpu_thread
