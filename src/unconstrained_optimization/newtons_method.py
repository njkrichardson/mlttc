from typing import Callable 

import jax.numpy as np 
import jax 

jax.config.update("jax_enable_x64", True)

def newton(f: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, **kwargs) -> np.ndarray:
    J: callable = jax.jacobian(f) 
    H: callable = jax.hessian(f) 
    x: np.ndarray = x0
    path: list[np.ndarray] = [x] 

    for i in range(kwargs.get("max_iterations", 100)):
        x -= np.linalg.solve(H(x), J(x))
        path.append(x) 
        if (np.abs(f(x)) < kwargs.get("root_convergence_tol", 1e-5)).all():
            print(f"Converged in {i+1} iterations")
            return x, np.array(path)
    print("Didn't converge!")
    return x, np.array(path)


def main(): 
    def f(x: np.ndarray) -> np.ndarray:
        return np.array([
            3. * x[0] - (x[1] * x[2])**2 - 3./2., 
            4. * x[0]**2 - 625. * x[1]**2 + 2 * x[1] - 1.,
            np.exp(-x[0] * x[1]) + 20. * x[2] + 9.,
        ])
    def objective(x: np.ndarray) -> float:
        f_x: np.ndarray = f(x) 
        return f_x.T @ f_x

    x0: np.ndarray = np.zeros(3)
    root, path = newton(objective, x0)
    for i, x in enumerate(path): 
        print(f"x_{i}: {x}")

    print(f"Root: {root}")
    print(f"Error: {objective(root)}")

if __name__ == "__main__":
    main()