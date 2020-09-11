import sys

size = int(sys.argv[1])
f = open(sys.argv[2], "r")
buffer = ""
buffer += f.read()
f.close()
newBuffer = ""
c = 0
for i in buffer:
    c += 1
    if i == "0":
        # newBuffer += "\u2B1B "
        newBuffer += "\u2B1C "
    else:
        # newBuffer += "\u2B1C "
        newBuffer += "\u2B1B "

    if c % int(size) == 0:
        newBuffer += "\n"

f = open(sys.argv[3], "w")
f.write(newBuffer)
f.close()
