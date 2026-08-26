# this file is to define physical system and level-set function

import numpy as np
from skfem import LinearForm
import numpy as np

class PhysicalProblemDefine:
    def __init__(self):
        pass

    def level_set(self, xy):
        raise NotImplementedError
    
    def level_set_cut(self, xy):
        raise NotImplementedError

    def trim_condition(self, ls_vals): # inside is <0
        return np.any(ls_vals > 0, axis=1)

    def source(self, x, y):
        return 0.0

    def get_val(self, x, y):
        raise NotImplementedError

    def get_grad(self, xy):
        raise NotImplementedError

    def exact_proj(self, p):
        return self.get_val(p[0], p[1])

    def get_source_form(self):
        @LinearForm
        def f(v, w):
            x, y = w.x
            return self.source(x, y) * v
        return f
    
class CircleLaplacian(PhysicalProblemDefine):
    def __init__(self, radius=0.2,center=(0.5, 0.5), U=1.0):
        super().__init__()
        self.xc, self.yc = center
        self.radius = radius
        self.U = U

    # ----------------------------------
    # Geometry
    # ----------------------------------
    def level_set(self, xy):
        x = xy[:, 0]
        y = xy[:, 1]
        return np.sqrt((x - self.xc)**2 + (y - self.yc)**2) - self.radius

    # ----------------------------------
    # used for cutFEM to trim the shaped void
    # ----------------------------------
    def level_set_cut(self, xy):
        return -(self.radius - np.sqrt((xy[0] - 0.5)**2 + (xy[1] - 0.5)**2))

    # ----------------------------------
    # Exact solution
    # ----------------------------------
    def get_val(self, x, y):
        dx = x - self.xc
        dy = y - self.yc
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)

        return self.U * r * (1 + (self.radius**2 / r**2)) * np.cos(theta)
    
    # ----------------------------------
    # Exact grad solution
    # ----------------------------------
    def get_grad(self, xy):
        x, y = xy
        dx = x - self.xc
        dy = y - self.yc
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)

        U = self.U
        R = self.radius

        dphi_dr = U * (1 - R**2 / r**2) * np.cos(theta)
        dphi_dtheta = -U * (r + R**2 / r) * np.sin(theta)

        dr_dx = dx / r
        dr_dy = dy / r
        dtheta_dx = -dy / r**2
        dtheta_dy = dx / r**2

        du = dphi_dr * dr_dx + dphi_dtheta * dtheta_dx
        dv = dphi_dr * dr_dy + dphi_dtheta * dtheta_dy

        return du, dv


class CircleLaplacian2mate(PhysicalProblemDefine):
    def __init__(self, radius=0.309,center=(0.5, 0.5), U=1.0, k1=5., k2 = 1.0):
        super().__init__()
        self.xc, self.yc = center
        self.radius = radius
        self.U = U
        self.k1 = k1 #inner
        self.k2 = k2 #outer
        self.a = 0.309 #inner radius
        self.b = 0.5 #outer radius

    # ----------------------------------
    # Geometry
    # ----------------------------------
    def level_set(self, xy):
        x = xy[:, 0]
        y = xy[:, 1]
        return np.sqrt((x - self.xc)**2 + (y - self.yc)**2) - self.radius

    # ----------------------------------
    # used for cutFEM to trim the shaped void
    # ----------------------------------
    def level_set_cut(self, xy):
        return -(self.radius - np.sqrt((xy[0] - 0.5)**2 + (xy[1] - 0.5)**2))

    # ----------------------------------
    # Exact solution
    # ----------------------------------

    def get_val(self, x, y):
        dx, dy = x - 0.5, y - 0.5
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        
        k1, k2 = self.k1, self.k2
        a, b = self.a, self.b
        
        C = -1 / k2 * (1 / (((k1 + k2) / (a**2 * (k2 - k1))) - (1 / b**2)))
        B = (C / a**2) * (k1 + k2) / (k2 - k1)
        A = B + (C / a**2)
        
        for j in r:
            for i in j:
                if i <= a:
                    res = A * i * np.cos(theta)
                elif i <= b:
                    res = (B * i+ C / i) * np.cos(theta)
                else:
                    res = 0.0 
            
        return res
    
    # ----------------------------------
    # Exact grad solution
    # ----------------------------------
    # def get_grad(self, xy):
    #     x, y = xy
    #     dx = x - self.xc
    #     dy = y - self.yc
    #     r = np.sqrt(dx**2 + dy**2)
    #     theta = np.arctan2(dy, dx)

    #     U = self.U
    #     R = self.radius

    #     dphi_dr = U * (1 - R**2 / r**2) * np.cos(theta)
    #     dphi_dtheta = -U * (r + R**2 / r) * np.sin(theta)

    #     dr_dx = dx / r
    #     dr_dy = dy / r
    #     dtheta_dx = -dy / r**2
    #     dtheta_dy = dx / r**2

    #     du = dphi_dr * dr_dx + dphi_dtheta * dtheta_dx
    #     dv = dphi_dr * dr_dy + dphi_dtheta * dtheta_dy

    #     return du, dv

class StarfishPoisson(PhysicalProblemDefine):
    def level_set(self, xy): #input xy shape (N,2)
        x = xy[:, 0]
        y = xy[:, 1]
        dx, dy = x - 0.5, y - 0.5
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        return r - (0.2 + 0.1 * np.sin(5 * theta))

    def level_set_cut(self, xy): #input xy shape (2,)
        dx = xy[0] - 0.5
        dy = xy[1] - 0.5
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
       
        return r - (0.2 + 0.1 * np.sin(5 * theta))


    def source(self, x, y):
        dx, dy = x - 0.5, y - 0.5
        theta = np.arctan2(dy, dx)
        r = np.sqrt(dx**2 + dy**2)
        omega, alpha, r0 = 5, 0.05, 0.2 # alpha 0.1 to 0.05

        t1 = alpha * np.sin(theta * omega)
        t2 = alpha * (np.cos(theta * omega))**2
        t3 = t1 - r + r0

        return -(alpha * omega**2 * (2 * t3**2 * t2 + t2 - (t3 * np.sin(theta * omega))) +
                 r * (-t1 + 2 * r * t3**2 + 2 * r - r0)) * (2 / (r**2)) * np.exp(t3**2)
    # ----------------------------------
    # Exact solution
    # ----------------------------------
    def get_val(self, x, y):
        dx, dy = x - 0.5, y - 0.5
        theta = np.arctan2(dy, dx)
        r = np.sqrt(dx**2 + dy**2)
        omega, alpha, r0 = 5, 0.05, 0.2
        return np.exp(((r0 + alpha * np.sin(theta * omega)) - r)**2)
    
    # ----------------------------------
    # Exact grad solution
    # ----------------------------------
    def get_grad(self, xy):
        x, y = xy
        dx, dy = x - 0.5, y - 0.5
        theta = np.arctan2(dy, dx)
        r = np.sqrt(dx**2 + dy**2)
        omega, alpha, r0 = 5, 0.05, 0.2

        g = (r0 + alpha * np.sin(omega * theta)) - r
        exp_term = np.exp(g**2)

        denom = dx**2 + dy**2
        dtheta_dx = -dy / denom
        dtheta_dy = dx / denom

        dr_dx = dx / r
        dr_dy = dy / r

        dg_dx = alpha * omega * np.cos(omega * theta) * dtheta_dx - dr_dx
        dg_dy = alpha * omega * np.cos(omega * theta) * dtheta_dy - dr_dy

        df_dx = 2 * g * exp_term * dg_dx
        df_dy = 2 * g * exp_term * dg_dy

        return df_dx, df_dy


