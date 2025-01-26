from typing import Callable 

import matplotlib.pyplot as plt 
import numpy as np 

ROOT_CONVERGENCE_TOL: float = 1e-5

def bisect(f: Callable[[np.ndarray], np.ndarray], a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray: 
    if np.sign(f(a)) == np.sign(f(b)): 
        raise ValueError("f(a) and f(b) must have different signs.")

    for i in range(kwargs.get("max_iterations", 100)): 
        m = (a+b)/2
        if np.abs(f(m)) < ROOT_CONVERGENCE_TOL: 
            return m 
        if np.sign(f(a)) != np.sign(f(m)): 
            b = m 
        else: 
            a = m

    return m 
    

def main(): 
    def f(x: np.ndarray) -> np.ndarray: 
        return (x-1)**2 - 1. 

    a: np.ndarray = np.array([1.1])
    b: np.ndarray = np.array([2.1])
    root: np.ndarray = bisect(f, a, b)

    x: np.ndarray = np.linspace(-.5, 2.5, 100)
    plt.figure() 
    plt.title("Bisection Method")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, f(x), c="tab:blue")
    plt.scatter(root, f(root), c="tab:red", label="root")
    plt.legend()
    plt.show()

if __name__=="__main__": 
    main() 