'''
    Author: Saurabh Mahajan
    Netid: sm6921
'''

from itertools import permutations

# function to print all strings of length n containing k zeros.
def zbits(n, k): 
    binary = '0' * k + '1' * (n-k)
    binary_set = set()
    # loop to create of all permutations of the binary digits.
    for i in permutations(binary, n):
    	s = ''.join(i)
    	if s not in binary_set:
	    	binary_set.add(s)
	    	print(s)
    return(binary_set)

# assertions to test the function
assert zbits(4, 3) == {'0100', '0001', '0010', '1000'}
assert zbits(4, 1) == {'0111', '1011', '1101', '1110'}
assert zbits(5, 4) == {'00001', '00100', '01000', '10000', '00010'}