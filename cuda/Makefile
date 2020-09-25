OBJS	= main.o game_of_life.o
SOURCE	= main.c game_of_life.c
HEADER	= game_of_life.h
OUT	= game_of_life
CC	 = gcc
FLAGS	 = -g -c -Wall
LFLAGS	 = 

all: $(OBJS)
	$(CC) -g $(OBJS) -o $(OUT) $(LFLAGS)

main.o: main.c
	$(CC) $(FLAGS) main.c 

game_of_life.o: game_of_life.c
	$(CC) $(FLAGS) game_of_life.c 


clean:
	rm -f $(OBJS) $(OUT)
