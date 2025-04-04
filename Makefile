# HIP compiler
HIPCC = hipcc

# Target binary and source file
TARGET = vectorAdd_dmabuf
SRC = vectorAdd_dmabuf.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCC) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
