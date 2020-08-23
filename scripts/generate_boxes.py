size = 32
f = open("/home/msi/projects/CLionProjects/game-of-life/mpi/input/32x32.txt", "r")
buffer = ""
buffer += f.read()
f.close()
print(buffer)
newBuffer = ""
c = 0
for i in buffer:
    if c % int(size) == 0:
        newBuffer += "\n"
    if i == "0":
        newBuffer += "\u2B1C "
    else:
        newBuffer += "\u2B1B "
    c += 1
print(newBuffer)
f = open("/home/msi/projects/CLionProjects/game-of-life/mpi/input/32x32-boxes.txt", "w")
f.write(newBuffer)
f.close()
