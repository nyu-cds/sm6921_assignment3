'''
Saurabh Mahajan
sm6921
This program generates a large list of unsorted numbers.
It divides range of numbers among processes to sort and then combines them.
'''

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank() # Get the index of a process
size = comm.Get_size() # Get the total number of processes

def generate_numbers(n): # Generate a dataset of random numbers
	nums = np.random.randint(0, n, n)
	return nums

def split_data(nums, size): # Splits the dataset into number of sublists to be sent to different processes
	data_range = np.asarray(range(max(nums) + 1))
	divisions = np.array_split(data_range, size)
	divided_lists = []
	for i in range(size):
		divided_lists.append([n for n in nums if (n in divisions[i])])
	return divided_lists

def parallel_sort(N): # Sending divided list into various processors and finally combining them to get sorted result
	divided_lists = None
	if rank == 0:
		divided_lists = split_data(generate_numbers(N), size)
	scatter_lists = comm.scatter(divided_lists, root=0) # Scatter divided lists into different processors
	integrate_lists = comm.gather(np.sort(scatter_lists), root=0)

	if rank == 0:
		sorted_list = np.concatenate(integrate_lists) # Combine all sorted lists
	return sorted_list

if __name__ == '__main__':
	N = 10
	print(parallel_sort(N))
