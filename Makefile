#-------------- basic settings ---------------
SRC    = histo.cpp
TARGET = histo
CFLAGS = -Wall -O3 `pkg-config --libs opencv`
LDFLAGS = `pkg-config --cflags opencv`
CC = g++

#-------------- compile and link ---------------
$(TARGET): Makefile $(SRC)
	$(CC) -ggdb -std=c++11 $(LDFLAGS) -o $(TARGET) $(SRC) $(CFLAGS)

#-------------- others ------------------------
.PHONY: clean
clean:
	rm $(TARGET)
