import random
import sys

buffer = ""
arraySizeInput = str(sys.argv[1])
arraySize = int(arraySizeInput) * int(arraySizeInput)
for i in range(0, arraySize):
    if random.choice([True, False]):
        buffer += "1"
    else:
        buffer += "0"

name = str(sys.argv[2])
f = open(name, "w")
f.write(buffer)
f.close()
