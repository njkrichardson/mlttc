import jax.numpy as np 
import jax 

def gradient_descent(f: callable, x0: np.ndarray, **kwargs) -> np.ndarray: 
    gradient: callable = jax.jit(jax.grad(f))
    x: np.ndarray = x0
    path = [x] 
    s = kwargs.get("sigma", np.array(1e-06))
    for i in range(kwargs.get("max_iterations", 1000)): 
        g_n: np.ndarray = gradient(x) 
        g_sn: np.ndarray = gradient(x + s*g_n)
        g_n_sqnorm: np.ndarray = g_n.T @ g_n
        step_size: np.ndarray = (s * g_n_sqnorm) / (g_n_sqnorm - (g_sn.T @ g_n))

        x = x - step_size * g_n
        path.append(x)
        if (np.abs(g_n) < kwargs.get("gradient_convergence_tol", 1e-3)).all(): 
            print(f"Converged after {i+1} iterations.")
            return x, np.array(path)

    print("Didn't converge")
    return x, np.array(path)



def main(): 
    def f(x: np.ndarray) -> np.ndarray: 
        return np.array([
            [3.*x[0] -np.cos(x[1] * x[2]) - 3/2], 
            [4.*(x[0]**2) -625. * (x[1]**2) + 2*x[2] - 1.], 
            [20.*x[2] + np.exp(-x[0] * x[1]) + 9.], 
        ]).ravel()

    def objective(x: np.ndarray) -> np.ndarray: 
        f_x = f(x)
        return f_x.T @ f_x

    x0: np.ndarray = np.array([1., 1.])
    x, path = gradient_descent(objective, x0)
    print(f"x: {x}")
    print(f"Error: {objective(x)}")

if __name__ == "__main__":
    main()