OBJS	= main.o mpi.o game_of_life.o
SOURCE	= main.c mpi.c game_of_life.c
HEADER	= mpi.h game_of_life.h
OUT	= game_of_life
CC	 = mpicc
FLAGS	 = -g -c -Wall -O3 
LFLAGS	 = 

all: $(OBJS)
	$(CC) -g $(OBJS) -o $(OUT) $(LFLAGS) -lm

main.o: main.c
	$(CC) $(FLAGS) main.c 

mpi.o: mpi.c
	$(CC) $(FLAGS) mpi.c

game_of_life.o: game_of_life.c
	$(CC) $(FLAGS) game_of_life.c

clean:
	rm -f $(OBJS) $(OUT)
