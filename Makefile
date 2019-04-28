PROJECT  =  netvlad

CXX  :=  g++
CC  :=  gcc

#linking libraries of OpenCV
LDFLAGES   =  -lopencv_highgui -lopencv_imgproc -lopencv_core
#linking libraries of DNNDK
LDFLAGES  +=  -lhineon -ln2cube -lpthread

INC += -I. `pkg-config --cflags opencv`
LIBS += `pkg-config --libs opencv`

#LIBDIRS = -L/usr/local/lib

CUR_DIR = $(shell pwd)
SRC     = $(CUR_DIR)/src
BUILD   = $(CUR_DIR)/build
MODEL   = $(CUR_DIR)/model
VPATH   = $(SRC)

CFLAGES := -g -mcpu=cortex-a53 -Wall -Wpointer-arith -std=c++11 -ffast-math

all:$(BUILD) $(PROJECT)

$(PROJECT):main.o
	$(CXX) $(LIBDIRS) $(CFLAGES) $(addprefix $(BUILD)/, $^) $(MODEL)/dpu_netvlad.elf -o $@ $(LDFLAGES) $(LIBS)

%.o : %.cc
	$(CXX) -c  $(INC) $(CFLAGES) $< -o $(BUILD)/$@  $(LIBS)

%.o : %.s
	$(CC) -c  $(INC) $(CFLAGES) $< -o $(BUILD)/$@  $(LIBS)

clean:
	$(RM) $(BUILD)/*.o
	$(RM) netvlad
