CC = gcc
CFLAGS = -shared -fPIC

TARGET = libMDPCBP.so
SOURCE = mdpcbp.c

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)