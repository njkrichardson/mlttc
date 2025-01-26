from typing import Callable 

import matplotlib.pyplot as plt
import jax.numpy as np 
import jax 

def newton(f: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, **kwargs) -> np.ndarray: 
    gradient: Callable[[np.ndarray], np.ndarray] = jax.grad(f) if x0.size == 1 else jax.jacobian(f)
    x: np.ndarray = x0 

    for _ in range(kwargs.get("max_iterations", 100)): 
        if x.size > 1: 
            x = x - np.linalg.solve(gradient(x), f(x))
            pass
        else: 
            x: np.ndarray = x - f(x)/gradient(x)
        if (np.abs(f(x)) < kwargs.get("root_convergence_tol", 1e-5)).all(): 
            return x 
    return x 

def main(): 
    def f(x: np.ndarray) -> np.ndarray: 
        return (x-1)**2 - 1.
        
    def f1(x: np.ndarray) -> np.ndarray: 
        return np.array([
            [3.*x[0] -np.cos(x[1] * x[2]) - 3/2], 
            [4.*(x[0]**2) -625. * (x[1]**2) + 2*x[2] - 1.], 
            [20.*x[2] + np.exp(-x[0] * x[1]) + 9.], 
        ]).ravel()


    x0: np.ndarray = np.array(1.1)
    root: np.ndarray = newton(f, x0)

    x: np.ndarray = np.linspace(-.5, 2.5, 100)
    plt.figure() 
    plt.title("Newton-Rhapson Method")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, f(x), c="tab:blue")
    plt.scatter(root, f(root), c="tab:red", label="root")
    plt.legend()
    plt.show()

    x0: np.ndarray = np.ones(3)
    root = newton(f1, x0)

if __name__ == "__main__":
    main()