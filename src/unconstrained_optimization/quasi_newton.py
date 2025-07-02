from typing import Optional

import jax.numpy as np
from jax import grad 

quasi_newton_methods: tuple[str] = (
    "bfgs", 
    "dfs"
)

def quasi_newton(f: callable, x0: np.ndarray, method: Optional[str]="bfgs", **kwargs) -> np.ndarray: 
    if method.lower() not in quasi_newton_methods: 
        raise ValueError(f"Method must be one of: {[method for method in quasi_newton_methods]} (got {method}))")
    elif method.lower() == "bfgs": 
        return bfgs(f, x0, **kwargs)

def bfgs(f: callable, x0: np.ndarray, **kwargs) -> np.ndarray: 
    raise NotImplementedError

def dfs(f: callable, x0: np.ndarray, **kwargs) -> np.ndarray: 
    raise NotImplementedError

def dfs(f: callable, x0: np.ndarray, **kwargs) -> np.ndarray: 
    raise NotImplementedError


def main(): 
    def f(x: np.ndarray) -> np.ndarray: 
        return 100.*(x[1] - x[0]**2)**2 + (1 - x[0])**2

    x0: np.ndarray = np.array([-0.5, 1.5])
    error: callable = lambda x: f(x) 
    gradient: callable = grad(f) 

    print(f"Initial error: {error(x0):.4f}\tInitial gradient norm: {np.linalg.norm(gradient(x0)):.4f}")


if __name__=="__main__": 
    main() 