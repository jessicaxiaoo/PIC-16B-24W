#!/usr/bin/env python
# coding: utf-8
---
title: "HW4"
author: "Jessica Xiao"
date: "2024-3-1"
categories: [week 7, homework]
---
# # HW 4: Heat Diffusion
# 
# In the homework, we will be conducting a simulation of two-dimensional heat diffusion in various ways.
# First, let's define our initial conditions and import the necessary libraries.
# 
# ```python
# N = 101
# epsilon = 0.2
# iterations = 2700
# plot_interval = 300
# 
# import numpy as np
# from matplotlib import pyplot as plt
# # construct initial condition: 1 unit of heat at midpoint. 
# u0 = np.zeros((N, N))
# u0[int(N/2), int(N/2)] = 1.0
# plt.imshow(u0)
# ```
# Here, we have the initial condition as in the 1D case: putting 1 unit of heat at the midpoint.
# ![output](hw4-initialplot.png)
# 
# ## With matrix multiplication
# 
# First, let’s use matrix-vector multiplication to simulate the heat diffusion in the 2D space. The vector here is created by flattening the current solution. 
# 
# ```python
# def advance_time_matvecmul(A, u, epsilon):
#     """Advances the simulation by one timestep, via matrix-vector multiplication
#     Args:
#         A: The 2d finite difference matrix, N^2 x N^2. 
#         u: N x N grid state at timestep k.
#         epsilon: stability constant.
# 
#     Returns:
#         N x N Grid state at timestep k+1.
#     """
#     N = u.shape[0]
#     u = u + epsilon * (A @ u.flatten()).reshape((N, N))
#     return u
# ```
# 
# This function takes the value N as the argument and returns the corresponding matrix A.
# ```python
# def get_A(N):
#     n = N * N
#     diagonals = [-4 * np.ones(n), np.ones(n-1), np.ones(n-1), np.ones(n-N), np.ones(n-N)]
#     diagonals[1][(N-1)::N] = 0
#     diagonals[2][(N-1)::N] = 0
#     A = np.diag(diagonals[0]) + np.diag(diagonals[1], 1) + np.diag(diagonals[2], -1) + np.diag(diagonals[3], N) + np.diag(diagonals[4], -N)
#     return A
# ```
# 
# After defining advance_time_matvecmul and get_A, we can run the simulation to visualize the diffusion of heat every 300 iterations using a heatmap.
# ```python
# start_time = time.time() #start time
# # Initialize intermediate solutions array for visualization
# intermediate_solutions = []
# 
# # Construct finite difference matrix
# A = get_A(N)
# 
# # Run simulation
# current_solution = u0.copy()
# for i in range(iterations):
#     current_solution = advance_time_matvecmul(A, current_solution, epsilon)
#     if (i + 1) % plot_interval == 0:
#         intermediate_solutions.append(current_solution)
# 
# end_time = time.time() # end time excluding time for visualization
#         
# # Visualize diffusion at specified intervals
# plt.figure(figsize=(12, 12))
# for i, solution in enumerate(intermediate_solutions):
#     plt.subplot(3, 3, i+1)
#     plt.imshow(solution, cmap='hot')
#     plt.title(f"Iteration {(i+1)*plot_interval}")
#     plt.axis('off')
# 
# plt.tight_layout()
# plt.show()
# elapsed_time = end_time - start_time
# print(elapsed_time) #44.71136999130249
# ```
# We can see that using matrix multiplication is quite slow, and takes 44.71136999130249 seconds to generate the heat diffusion simulation below.
# ![output](hw4-output.png)
# 
# 
# ## Sparse matrix in JAX
# 
# ## Direct operation with numpy
# A much efficient way to generate this simulation is through direct operation with numpy. Here, we can define advance_time_numpy in heat_equation.py and run the simulation in intervals similar to the other methods.
# 
# ```python
# def advance_time_numpy(u, epsilon):
#     """Advances the solution by one timestep using numpy vectorized operations.
#     
#     Args:
#         u: N x N grid state at timestep k.
#         epsilon: stability constant.
# 
#     Returns:
#         N x N grid state at timestep k+1.
#     """
#     # Pad zeros to form an (N+2) x (N+2) array
#     padded_u = np.pad(u, 1, mode='constant')
# 
#     # Compute the Laplacian using np.roll()
#     laplacian = (
#         np.roll(padded_u, 1, axis=0) + np.roll(padded_u, -1, axis=0) +
#         np.roll(padded_u, 1, axis=1) + np.roll(padded_u, -1, axis=1) -
#         4 * padded_u
#     )
# 
#     # Update the solution using the heat equation
#     new_u = u + epsilon * laplacian[1:-1, 1:-1]
# 
#     return new_u
# ```
# 
# ```python
# start_time = time.time()
# 
# # Construct initial condition
# u0 = np.zeros((N, N))
# u0[int(N/2), int(N/2)] = 1.0
# 
# # Initialize intermediate solutions array for visualization
# intermediate_solutions_numpy = []
# 
# # Run simulation with numpy
# current_solution = u0.copy()
# for i in range(iterations):
#     current_solution = advance_time_numpy(current_solution, epsilon)
#     if (i + 1) % plot_interval == 0:
#         intermediate_solutions_numpy.append(current_solution)
# 
# end_time = time.time()
# 
# # Visualize diffusion at specified intervals
# plt.figure(figsize=(12, 12))
# for i, solution in enumerate(intermediate_solutions_numpy):
#     plt.subplot(3, 3, i+1)
#     plt.imshow(solution, cmap='hot')
#     plt.title(f"Iteration {(i+1)*plot_interval}")
#     plt.axis('off')
# 
# plt.tight_layout()
# plt.show() 
# elapsed_time = end_time - start_time
# print(elapsed_time) #0.18757390975952148
# ```
# We can see that using direct operation with numpy is significantly faster, only taking 0.18757390975952148 seconds to run the simulation.
# ![output](hw4-output.png)
# 
# 
# ## With JAX
# Finally, we can use JAX and define advance_time_jax without using a sparse matrix.
# 
# ```python
# @jit
# def advance_time_jax(u, epsilon):
#     """Advances the solution by one timestep using JAX and JIT compilation
#     Args:
#         u: N x N grid state at timestep k.
#         epsilon: stability constant.
# 
#     Returns:
#         N x N Grid state at timestep k+1.
#     """
#     # Extract the size of the grid
#     N = u.shape[0]
#     
#     # Create a padded version of 'u' to simplify boundary computations
#     padded_u = jnp.pad(u, 1, mode='constant')
#     
#     # Calculate the updates in a vectorized manner
#     update_value = epsilon * (padded_u[:-2, 1:-1] + padded_u[2:, 1:-1] + padded_u[1:-1, :-2] + padded_u[1:-1, 2:] - 4 * u)
#     u_new = u + update_value
#     
#     return u_new
# ```
# 
# 
# ```python
# start_time = time.time()
# 
# # Construct initial condition
# u0 = np.zeros((N, N))
# u0[int(N/2), int(N/2)] = 1.0
# 
# # Initialize intermediate solutions array for visualization
# intermediate_solutions_jax = []
# 
# # Run simulation with JAX
# current_solution = u0.copy()
# for i in range(iterations):
#     current_solution = advance_time_jax(current_solution, epsilon)
#     if (i + 1) % plot_interval == 0:
#         intermediate_solutions_jax.append(current_solution)
# 
# end_time = time.time()
# 
# # Visualize diffusion at specified intervals
# plt.figure(figsize=(12, 12))
# for i, solution in enumerate(intermediate_solutions_jax):
#     plt.subplot(3, 3, i+1)
#     plt.imshow(solution, cmap='hot')
#     plt.title(f"Iteration {(i+1)*plot_interval}")
#     plt.axis('off')
# 
# plt.tight_layout()
# plt.show() 
# elapsed_time = end_time - start_time
# print(elapsed_time) #0.07134509086608887
# ```
# Using JAX only takes us 0.07134509086608887 seconds to generate the simulation. This method is the fastest, and has clean and short code that is easy to follow.
# ![output](hw4-output.png)
# 
# 
# ## Full code for heat_equation.py
# The full implementation for heat_equation.py can be seen below.
# 
# ```python
# N = 101
# epsilon = 0.2
# 
# import numpy as np
# from matplotlib import pyplot as plt
# import jax.numpy as jnp
# import jax
# from jax import jit, ops
# from jax import lax
# from jax.experimental.sparse import bcoo
# import jax.scipy.sparse as sps
# 
# # construct initial condition: 1 unit of heat at midpoint. 
# u0 = np.zeros((N, N))
# u0[int(N/2), int(N/2)] = 1.0
# plt.imshow(u0)
# 
# def advance_time_matvecmul(A, u, epsilon):
#     """Advances the simulation by one timestep, via matrix-vector multiplication
#     Args:
#         A: The 2d finite difference matrix, N^2 x N^2. 
#         u: N x N grid state at timestep k.
#         epsilon: stability constant.
# 
#     Returns:
#         N x N Grid state at timestep k+1.
#     """
#     N = u.shape[0]
#     u = u + epsilon * (A @ u.flatten()).reshape((N, N))
#     return u
# 
# # @jit
# def advance_time_matvecmul_sparse(A, u, epsilon):
#     """Advances the simulation by one timestep, via matrix-vector multiplication
#     Args:
#         A: The 2d finite difference matrix, N^2 x N^2. 
#         u: N x N grid state at timestep k.
#         epsilon: stability constant.
# 
#     Returns:
#         N x N Grid state at timestep k+1.
#     """
#     N = u.shape[0]
#     
#     # Convert u to a vector
#     u_vec = u.flatten()
#     
#     # Sparse matrix-vector multiplication
#     result_vec = bcoo.bcoo_multiply_dense(A, u_vec)
#     
#     # Reshape the result vector to N x N grid
#     result_grid = result_vec.reshape((N, N))
#     
#     # Update the grid state
#     u_new = u + epsilon * result_grid
#     return u_new
# 
# def get_A(N):
#     n = N * N
#     diagonals = [-4 * np.ones(n), np.ones(n-1), np.ones(n-1), np.ones(n-N), np.ones(n-N)]
#     diagonals[1][(N-1)::N] = 0
#     diagonals[2][(N-1)::N] = 0
#     A = np.diag(diagonals[0]) + np.diag(diagonals[1], 1) + np.diag(diagonals[2], -1) + np.diag(diagonals[3], N) + np.diag(diagonals[4], -N)
#     return A
# 
# 
# def get_sparse_A(N):
#     """Constructs the finite difference matrix A for 2D heat equation in a sparse format using JAX's experimental sparse module.
#     
#     Args:
#         N: Size of the grid.
#         
#     Returns:
#         A_sp_matrix: The sparse representation of finite difference matrix in COO format.
#     """
#     n = N * N
# 
#     # Initialize data for diagonals
#     data = [-4 * jnp.ones(n), jnp.ones(n-1), jnp.ones(n-1), jnp.ones(n-N), jnp.ones(n-N)]
# 
#     # Modify diagonals
#     data[1] = jnp.where(jnp.arange(1, n) % N == 0, 0, data[1])
#     data[2] = jnp.where(jnp.arange(0, n-1) % N == 0, 0, data[2])
# 
#    # rows = jnp.tile(jnp.arange(N), N)
#    # cols = jnp.repeat(jnp.arange(N), N)
# 
#     # Create sparse matrix in BCOO format
#    # offsets = [-N, -1, 0, 1, N]
#     A_sp_matrix = bcoo.BCOO(data, shape=(n, n), indices_sorted=True)
# 
#     return A_sp_matrix
# '''
# def get_sparse_A(N):
#     """Constructs the finite difference matrix A for 2D heat equation in a sparse format.
#     
#     Args:
#         N: Size of the grid.
#         
#     Returns:
#         A_sp_matrix: The sparse representation of finite difference matrix.
#     """
#     n = N * N
#     diagonals = [-4 * jnp.ones(n), jnp.ones(n-1), jnp.ones(n-1), jnp.ones(n-N), jnp.ones(n-N)]
#     diagonals[1] = jnp.where(jnp.arange(1, n) % N == 0, 0, diagonals[1])
#     diagonals[2] = jnp.where(jnp.arange(0, n-1) % N == 0, 0, diagonals[2])
#     return A_sp_matrix
# '''
# 
# # part 3
# def advance_time_numpy(u, epsilon):
#     """Advances the solution by one timestep using numpy vectorized operations.
#     
#     Args:
#         u: N x N grid state at timestep k.
#         epsilon: stability constant.
# 
#     Returns:
#         N x N grid state at timestep k+1.
#     """
#     # Pad zeros to form an (N+2) x (N+2) array
#     padded_u = np.pad(u, 1, mode='constant')
# 
#     # Compute the Laplacian using np.roll()
#     laplacian = (
#         np.roll(padded_u, 1, axis=0) + np.roll(padded_u, -1, axis=0) +
#         np.roll(padded_u, 1, axis=1) + np.roll(padded_u, -1, axis=1) -
#         4 * padded_u
#     )
# 
#     # Update the solution using the heat equation
#     new_u = u + epsilon * laplacian[1:-1, 1:-1]
# 
#     return new_u
# 
# @jit
# def advance_time_jax(u, epsilon):
#     """Advances the solution by one timestep using JAX and JIT compilation
#     Args:
#         u: N x N grid state at timestep k.
#         epsilon: stability constant.
# 
#     Returns:
#         N x N Grid state at timestep k+1.
#     """
#     # Extract the size of the grid
#     N = u.shape[0]
#     
#     # Create a padded version of 'u' to simplify boundary computations
#     padded_u = jnp.pad(u, 1, mode='constant')
#     
#     # Calculate the updates in a vectorized manner
#     update_value = epsilon * (padded_u[:-2, 1:-1] + padded_u[2:, 1:-1] + padded_u[1:-1, :-2] + padded_u[1:-1, 2:] - 4 * u)
#     u_new = u + update_value
#     
#     return u_new
# 
# 
# # JIT compile the function
# #advance_time_jax_jit = jit(advance_time_jax)
# ```

# In[ ]:




