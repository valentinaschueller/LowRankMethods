using LowRankMethods

B_test = [0 0.5 0 0; -0.5 0 0.5 0; 0 -0.5 0 0.5; 0 0 -1 1]
@assert time_matrix(4, 1) == B_test
A_test = [-2.0 1 0 0; 1 -2 1 0; 0 1 -2 1; 0 0 1 -2]
@assert space_matrix(4, 1) == A_test

@assert eigensolver(10, 10) ≈ solve_sylvester(10, 10)