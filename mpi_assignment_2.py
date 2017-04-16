'''
Saurabh Mahajan
sm6921@nyu.edu
Each process sends a number to the next process.
The first process accepts a number and sends to the second process.
The second process multiples the rank with the number and forwards to the third process.
This continues and the last process sends the result to the first process to be displayed.
'''
import numpy
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank() # get the index of a process
size = comm.Get_size() # get the total number of processes
num = numpy.zeros(1)

if rank == 0: # if its process 0, accept an integer less than 100
    num[0] = 100
    while num[0] >= 100:
        try: # handle exceptions for non-integer values
            input_num = input("Starting number: ")
            num[0] = int(input_num)
            if num[0] >= 100:
                print("Error. Number greater than 100")
        except:
            print("Error. Input not an integer")
    comm.Send(num, dest = 1)
    comm.Recv(num, source = size - 1) # receive result from the last process
    print(num[0])
else: # for all processes after process 0, rank is multiplied to the number and sent to the next process
    comm.Recv(num, source = rank - 1)
    num *= rank
    # if this is the last process, send the result to process 0 otherwise to the next process
    if rank == (size-1):
        comm.Send(num, dest = 0)
    else:
        comm.Send(num, dest = rank + 1)
