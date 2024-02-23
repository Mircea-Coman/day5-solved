import numpy as np
from scipy.linalg import solve, eig, inv, det, norm

A = np.array([
    [1, -2, 3],
    [4, 5, 6],
    [7, 1, 9]
])
print(f"Matrix A: \n{np.array2string(A)}")

b = np.array([[1, 2, 3]])
print(f"Matrix b: {np.array2string(b)}")

x = solve(A, b.T)
print(f"Solution of Ax=b: {np.array2string(x.T)}")

b_calc = A @ x
print(f"Ax=: {np.array2string(b_calc.T)}")


print("\nRandom Stuff")

b_rand = np.random.rand(3, 3)
print(f"Random matrix b: {np.array2string(b_rand.T)}")

x_rand = solve(A, b_rand)
print(f"Solution of Ax=b: {np.array2string(x_rand.T)}")

b_rand_calc = A @ x_rand
print(f"Ax= {np.array2string(b_rand_calc.T)}")

eig_result = eig(A)
eig_check = eig_result[1] @ np.diag(eig_result[0]) @ inv(eig_result[1])
print(f'\nEigenvalues: {np.array2string(eig_result[0])}')
print(f'Eigenvectors: \n{np.array2string(eig_result[1])}')
print(f'Diagonal Check: U^{-1} A_d U = \n{np.array2string(eig_check)}'  )

inv_A = inv(A)
inv_check = inv(A)@A
print(f"Inverse of A: \n{np.array2string(inv_A)}")
print(f"invA * A = \n{np.array2string(inv_check)}")


det_A = det(A)
print(f"Determinant of A: \n{det_A}")


print(f"Norm of order 1 of A: {norm(A, ord = 1)}")
print(f"Norm of order 2 of A: {norm(A, ord = 2)}")
print(f"Norm of order inf of A: {norm(A, ord = np.inf)}")
