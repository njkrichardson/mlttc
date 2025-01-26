from typing import Callable 

import matplotlib.pyplot as plt 
import numpy as np 

ROOT_CONVERGENCE_TOL: float = 1e-5

def secant(f: Callable[[np.ndarray], np.ndarray], a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray: 

    for i in range(kwargs.get("max_iterations", 100)): 
        x  = a - (b-a) * f(a) / (f(b)-f(a))
        if np.abs(f(x)) < ROOT_CONVERGENCE_TOL: 
            print(f"converged after {i+1} iterations")
            return x

        if (f(a) * f(b)) < 0: 
            if np.sign(f(a)) == np.sign(f(x)): 
                a = x 
            else: 
                b = x 
        else: 
            if (np.abs(f(a)) > np.abs(f(b))): 
                a = x 
            else: 
                b = x 
    
    return x 

    

def main(): 
    def f(x: np.ndarray) -> np.ndarray: 
        return (x-1)**2 - 1. 

    a: np.ndarray = np.array([1.1])
    b: np.ndarray = np.array([2.1])
    root: np.ndarray = secant(f, a, b)

    x: np.ndarray = np.linspace(-.5, 2.5, 100)
    plt.figure() 
    plt.title("Secant Method")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, f(x), c="tab:blue")
    plt.scatter(root, f(root), c="tab:red", label="root")
    plt.legend()
    plt.show()

if __name__=="__main__": 
    main() 