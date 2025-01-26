import matplotlib.pyplot as plt
import numpy as np 

def main(): 
    def f(x: np.ndarray) -> np.ndarray: 
        return np.exp(x) - (1 / x)

    def g(x: np.ndarray) -> np.ndarray: 
        return np.exp(-x)

    max_iterations: int = 100 
    CONVERGENCE_TOL: float = 1e-05 

    root: np.ndarray = np.array([1.])

    for _ in range(max_iterations): 
        root = g(root) 
        if np.abs(f(root)) < CONVERGENCE_TOL: 
            break

    x: np.ndarray = np.linspace(0, 3, 100)
    plt.figure() 
    plt.title("Fixed Point Iteration")
    plt.xlabel("x")
    plt.plot(x, f(x), label="f(x)")
    plt.scatter(root, f(root), color="red", label="Root")
    plt.legend()
    plt.show()



    

if __name__ == "__main__":
    main()