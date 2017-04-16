'''
Saurabh Mahajan
sm6921@nyu.edu
Print “Hello” with even ranked processes and “Goodbye” for odd ranked ones.
'''

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank() # get the index of a process.

if rank % 2: # if the rank of a particular process is odd.
    print("Goodbye from process", rank)
else: # if rank is even.
    print("Hello from process", rank)
