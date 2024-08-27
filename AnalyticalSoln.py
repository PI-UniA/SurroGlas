import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

class AnalyticalSoln():
    
    def T(t_: float, constants:dict):
        """Calculate the temperature at time t,
        Eq. A.1"""
        a = constants["a"]
        b = constants["b"]
        c = constants["c"]

        return c / (np.log(a*t_ + b))

    def phi(t_:float, constants:dict):
        """Calculate the shift function,
        Eq. A.2"""
        Tb = constants["Tb"]
        a = constants["a"]
        c = constants["c"]
        b = constants["b"]
        H = constants["H"]

        return np.exp(H*((1/Tb) - (1/AnalyticalSoln.T(t_, constants))))

    def E(t_: float, constants:dict):
        """Calculate the relaxation modulus,
        Eq. A.3"""
        E0 = constants["E0"]
        lambda_ = constants["lambda_"]

        return E0*np.exp(-t_/lambda_)

    def xi(t_: float, constants:dict):
        """Integrate the scaled time by quadrature integration,
        Eq. 3 """
        scaled_time, _ = quad(lambda t_: AnalyticalSoln.phi(t_, constants), 0, t_)
        return scaled_time

    def epsilon(t_: float, constants:dict):
        """Calculate the strain infinitesimal strain increment,
        Eq. A.4"""
        k = constants["k"]
        a = constants["a"]
        c = constants["c"]
        b = constants["b"]
        H = constants["H"]

        if t_ < 0:
            return 0
        else:
            return k * (a*t_ + b)**(-H/c)

    def stress(t_: float, constants:dict):
        """Integrate the uniaxial load by quadrature integration,
        Eq. A.4, 5 """
        def integrand_s(t_prime):
            xi_c = AnalyticalSoln.xi(t_, constants)
            xi_prime = AnalyticalSoln.xi(t_prime, constants)
            xi_diff = xi_c - xi_prime     
            return AnalyticalSoln.E(xi_diff, constants) * AnalyticalSoln.epsilon(t_prime, constants)

        sigma, _ = quad(integrand_s, 0, t_)
        return sigma

    def sigma_analytical(t_: float, constants:dict):

        """Calculate the stress infinitesimal strain increment as alternative solution,
        Eq. A.6"""

        Tb = constants["Tb"]
        a = constants["a"]
        c = constants["c"]
        b = constants["b"]
        H = constants["H"]
        E0 = constants["E0"]
        lambda_ = constants["lambda_"]
        k = constants["k"]

        A1 = (-1 / (lambda_ * E0 * k)) * np.exp((H * (H - 2 * c)) / (Tb * (H - c)))
        A2 = c / (a * lambda_ * Tb * (H - c))
        A3 = -H / c
        A4 = -np.exp(-H * c / (Tb * (H - c)))
        A5 = A2 * np.exp(H / Tb)

        return ((np.exp((A5*b*Tb*((a*t_)+b)**A3) - (A5*Tb*b**(A3+1)) + (A5*t_*a*Tb*((a*t_)+b)**A3) - (A2*H*a*lambda_))) + A4)/(A1*10**6)
        # the term is divided by 10**6 in order to convert the units from Pa into MPa 



