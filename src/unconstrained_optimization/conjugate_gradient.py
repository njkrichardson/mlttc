import matplotlib.pyplot as plt 
import jax.numpy as np 
from jax import grad, jacfwd, jacrev

def minimize_1d(f: callable, x0=None, **kwargs): 
    x = x0 if x0 is not None else np.zeros(1) 
    gradient: callable = lambda x: grad(f)(x).item()
    hessian: callable = lambda x: jacfwd(jacrev(f))(x).item()

    for _ in range(kwargs.get("max_iterations", 10)): 
        if np.abs(gradient(x)) < 1e-02: 
            break 
        else: 
            x -= gradient(x) / hessian(x) 


    return x 

def conjugate_gradient(f: callable, x0: np.ndarray, **kwargs) -> np.ndarray: 
    gradient: callable = grad(f) 
    current_error: np.ndarray = f(x0)
    x: np.ndarray = x0 
    g_x: np.ndarray = gradient(x)
    search_direction: np.ndarray = -g_x

    for i in range(kwargs.get("max_iterations", x0.size)):
        delta: np.ndarray = minimize_1d(lambda d: f(d*search_direction))
        x: np.ndarray = x + delta*search_direction
        current_error: np.ndarray = f(x)

        g_xn: np.ndarray = gradient(x) 
        beta: np.ndarray = -(np.dot(g_xn, g_xn)) / (np.dot(g_x, g_x))
        search_direction: np.ndarray = -g_xn - beta*search_direction
        g_x = g_xn

        if kwargs.get("debug", False): 
            print(f"iteration [{i:02d}/{kwargs.get('max_iterations', x0.size):02d}] x: [{x[0].item():.3e}, {x[1].item():.3e}, {x[2].item():.3e}] error: {current_error.item():.6e}")

    return x 


def main(): 
    def f(x: np.ndarray) -> np.ndarray: 
        z = np.array([
            3.*x[0] - (x[1]*x[2])**2 - 3./2., 
            4.*x[0]**2 - 500*x[1]**2 + 2.*x[1] - 1., 
            np.exp(-x[0]*x[1]) + 20.*x[2] + 9., 
        ])
        return np.dot(z, z)

    x0: np.ndarray = np.zeros(3)
    error: np.ndarray = f(x0)
    x: np.ndarray = conjugate_gradient(f, x0, debug=True)



if __name__ == "__main__":
    main()