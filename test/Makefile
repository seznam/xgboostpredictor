# Makefile for gtest

G++ = g++
G++_FLAGS = -c -Wall -I ../src -std=c++17
LD_FLAGS = -lgtest -lpthread -lgmock

OBJECTS = main.o xgboostpredictor_test.o
TARGET = xgboostpredictor_test

all: $(TARGET)

$(TARGET): $(OBJECTS)
	 $(G++) -o $(TARGET) $(OBJECTS) $(LD_FLAGS)

%.o : %.cc
	$(G++) $(G++_FLAGS) $<

test: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJECTS)
