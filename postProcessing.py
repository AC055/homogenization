# this file is to plot all the figures, save the file as .vtk format and plot the 2d figure (built-in function in skfem)
# the generated file will be saved inside the defined folder in th edefined path

import numpy as np
import matplotlib.pyplot as plt
from skfem.visuals.matplotlib import plot
import os
import numpy as np
import matplotlib.pyplot as plt

class ResultVisualizer:
    def __init__(self, save_dir="results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    # -----------------------------
    # Plot convergence line
    # -----------------------------
    def _compute_slope(self, x, y):
        log_x = np.log(x)
        log_y = np.log(y)
        slope, intercept = np.polyfit(log_x, log_y, 1)
        fit_y = np.exp(intercept) * (x**slope)
        return slope, fit_y

    def plot_convergence(self, plot_data, title="Convergence Analysis", show_slope=True, save_name=None):
        plt.figure(figsize=(8, 6))

        for data in plot_data:
            dof = np.array(data["dof"])
            err = np.array(data["error"])
            label = data.get("label", "Data")
            color = data.get("color", None)

            plt.loglog(dof, err, 'o', color=color, label=label, linewidth=2, markersize=7)

            if show_slope and len(dof) > 1:
                slope, fit_y = self._compute_slope(dof, err)
                plt.loglog(dof, fit_y, ':', color=color, alpha=0.7,
                           label=f"{label} slope={slope:.2f}")

        plt.xlabel("Degrees of Freedom (DOFs)")
        plt.ylabel("Error")
        plt.title(title)
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()

        if save_name:
            plt.savefig(os.path.join(self.save_dir, f"{save_name}.png"), dpi=300)
        plt.show()


    # -----------------------------
    # output result to paraview as forlat .vtk
    # -----------------------------
    def save_to_vtk(self, mesh, filename, point_data=None, cell_data=None, subfolder=None):
        base = os.path.splitext(os.path.basename(filename))[0]

        if subfolder:
            folder = os.path.join(self.save_dir, subfolder)
        else:
            folder = self.save_dir

        os.makedirs(folder, exist_ok=True)

        vtk_path = os.path.join(folder, f"{base}.vtk")

        mesh.save(
            vtk_path,
            point_data=point_data if point_data else {},
            cell_data=cell_data if cell_data else {}
        )

        print(f"VTK saved: {vtk_path}")

    # -----------------------------
    # Plot 2d visual figure
    # -----------------------------
    def visual_plot(self, basis, ux,uy, uex, alpha,
                    
                    title='Result',
                    show=False,
                    filename=None,
                    save_vtk=True): #removed uex from the input arguments
        
        # Save VTK if requested
        if save_vtk and filename:
            self.save_to_vtk(
                basis.mesh,
                filename,
                point_data={
                    'temperature_x': ux,
                    'temperature_y': uy,
                    'u_exact': uex
                    # 'error_abs': np.abs(u - uex)
                }, 
                cell_data={
                "alpha": [alpha],}

            )

        if show:
            # Numerical solution
            plot(basis, ux, shading='gouraud', colorbar=True)
            # plt.title(f'Numerical: {title}')
            plt.title('f=0.3')

            # Numerical solution
            plot(basis, uy, shading='gouraud', colorbar=True)
            # plt.title(f'Numerical: {title}')
            plt.title('f=0.3')

            # # Exact solution
            # plot(basis, uex, shading='gouraud', colorbar=True)
            # plt.title(f'f=0.3, exact')

            plt.show()

    # -----------------------------
    # Skewness map
    # -----------------------------
    def skewness_map(self, mesh, filename=None):

        if not filename:
            return

        ref_F = np.array([
            [1., 0.5],
            [0., np.sqrt(3)/2.]
        ])

        eigen_val_list = []
        node_xy = mesh.p.T
        cell_list = mesh.t.T

        ref_Finv = np.linalg.inv(ref_F)

        for cell in cell_list:
            ele_node = node_xy[cell]

            ele_F = np.stack(
                (ele_node[1] - ele_node[0],
                 ele_node[2] - ele_node[0]),
                axis=1
            )

            F = ele_F @ ref_Finv
            C = F.T @ F

            val = np.sqrt(np.linalg.eigvals(C))
            ratio = max(val) / min(val)

            eigen_val_list.append(ratio)

        # Save using shared function
        self.save_to_vtk(
            mesh,
            filename,
            cell_data={'skewness': [eigen_val_list]},
            subfolder="paraview_map"
        )


    def plot_level_set(self, xy, t2v, iso_zero, lsVal, title, analytical_curve=None):
        import matplotlib.pyplot as plt
        import matplotlib.tri as tri

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect('equal')

        triangulation = tri.Triangulation(
            xy[:, 0],
            xy[:, 1],
            t2v
        )

        tpc = ax.tripcolor(
            triangulation,
            lsVal,
            shading='flat',
            edgecolors='k'
        )

        fig.colorbar(tpc)

        ax.tricontour(
            triangulation,
            lsVal,
            levels=[0],
            colors='red',
            linewidths=1
        )
        

        ax.plot(iso_zero[:,0], iso_zero[:,1], 'o', color='orange')
        ax.set_title(title)

        plt.show()

    # def plot_hausdorff_convergence(self, hausdorff_data, title="Hausdorff Distance Convergence", save_name=None):
    #     """
    #     Plot Hausdorff distance vs mesh cell size (h) on a log-log scale.
 
    #     Parameters
    #     ----------
    #     hausdorff_data : list of dict
    #         Each entry must contain:
    #           - 'h'    : characteristic mesh size (e.g. 1/sqrt(N_dof) or actual h)
    #           - 'dist' : Hausdorff distance value
    #           - 'label': (optional) series label
    #           - 'color': (optional) line colour
    #     title : str
    #     save_name : str or None  – if given, saves a .png to self.save_dir
    #     """
    #     fig, ax = plt.subplots(figsize=(8, 6))
 
    #     # Group by label so multiple series can be overlaid
    #     from itertools import groupby
    #     for data in hausdorff_data:
    #         h    = np.array(data["h"])
    #         dist = np.array(data["dist"])
    #         label = data.get("label", "XMesh")
    #         color = data.get("color", None)
 
    #         order = np.argsort(h)
    #         h, dist = h[order], dist[order]
 
    #         ax.loglog(h, dist, 'o-', color=color, label=label, linewidth=2, markersize=7)
 
    #         if len(h) > 1:
    #             slope, fit_y = self._compute_slope(h, dist)
    #             ax.loglog(h, fit_y, ':', color=color, alpha=0.7,
    #                       label=f"{label} slope={slope:.2f}")
 
    #     ax.set_xlabel("Mesh cell size $h$")
    #     ax.set_ylabel("Hausdorff distance")
    #     ax.set_title(title)
    #     ax.grid(True, which="both", ls="-", alpha=0.2)
    #     ax.legend()
 
    #     if save_name:
    #         fig.savefig(os.path.join(self.save_dir, f"{save_name}.png"), dpi=300)
    #     plt.show()
    #     return fig



# # def compute_HausdorffDist(xy, t2v, iso_zero, lsVal):
# #     from scipy.spatial import cKDTree

# #     tree = cKDTree(xy)
# #     dist, _ = tree.query(iso_zero)
# #     hausdorff_dist = np.max(dist)

# #     return hausdorff_dist


# def compute_HausdorffDist(iso_zero, analytical_pts):
#     """
#     Symmetric Hausdorff distance between two point clouds using
#     scipy.spatial.distance.directed_hausdorff.
 
#     Parameters
#     ----------
#     iso_zero       : (M, 2) array – iso-zero intersection points from XMesh
#     analytical_pts : (K, 2) array – densely sampled points on the exact boundary
 
#     Returns
#     -------
#     float : symmetric Hausdorff distance
#     """
#     from scipy.spatial.distance import directed_hausdorff
 
#     d_na = directed_hausdorff(iso_zero, analytical_pts)[0]   # numerical → analytical
#     d_an = directed_hausdorff(analytical_pts, iso_zero)[0]   # analytical → numerical
 
#     return float(max(d_na, d_an))


# def sample_analytical_boundary(problem, n_pts=2000):
#     """
#     Densely sample the exact zero-level-set boundary of *problem* by
#     evaluating the level-set on a fine regular grid and collecting the
#     contour points.  Falls back to a parametric circle/starfish sampler
#     when the class name is recognised.
 
#     Parameters
#     ----------
#     problem : PhysicalProblemDefine subclass
#     n_pts   : approximate number of sample points
 
#     Returns
#     -------
#     pts : (K, 2) ndarray
#     """
#     # --- parametric samplers for known geometries --------------------------
#     class_name = type(problem).__name__
 
#     if class_name == "CircleLaplacian":
#         theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
#         x = problem.xc + problem.radius * np.cos(theta)
#         y = problem.yc + problem.radius * np.sin(theta)
#         return np.stack([x, y], axis=1)
 
#     if class_name == "StarfishPoisson":
#         theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
#         r = 0.2 + 0.1 * np.sin(5 * theta)
#         x = 0.5 + r * np.cos(theta)
#         y = 0.5 + r * np.sin(theta)
#         return np.stack([x, y], axis=1)
 
#     # --- generic: marching-squares on level-set field ----------------------
#     from matplotlib.contour import QuadContourSet
#     grid = int(np.sqrt(n_pts) * 4)
#     xs = np.linspace(0, 1, grid)
#     ys = np.linspace(0, 1, grid)
#     XX, YY = np.meshgrid(xs, ys)
#     xy_grid = np.stack([XX.ravel(), YY.ravel()], axis=1)
#     ls_vals  = problem.level_set(xy_grid).reshape(grid, grid)
 
#     import matplotlib
#     matplotlib.use('Agg')          # non-interactive backend for contour extraction
#     fig_tmp, ax_tmp = plt.subplots()
#     cs = ax_tmp.contour(XX, YY, ls_vals, levels=[0])
#     pts_list = []
#     for collection in cs.collections:
#         for path in collection.get_paths():
#             pts_list.append(path.vertices)
#     plt.close(fig_tmp)
 
#     if pts_list:
#         return np.concatenate(pts_list, axis=0)
#     raise RuntimeError("Could not extract analytical boundary for this problem.")


# def compute_CondiNum(A, sigma = 1e-8):
#     from scipy.sparse.linalg import eigsh

#     lmax = eigsh(A, k=1)[0]
#     # lmin1 = eigsh(A, k=1, which='SM')[0] 
#     lmin = eigsh(A, k=1, sigma=sigma)[0] 
#     if lmin == 0:
#         return np.inf
#     cond = lmax / lmin
#     return cond[0]

    

# def compute_CondiNum1(A):
#     """
#     Estimates the reciprocal condition number (rcond) of a sparse matrix A.
#     """
#     import numpy as np
#     import scipy.sparse.linalg as spla
#     # 1. Calculate the 1-norm of the original sparse matrix A
#     norm_A = spla.norm(A, ord=1)
    
#     if norm_A == 0:
#         return 0.0
    
#     # 2. Perform the LU factorization using SuperLU
#     lu = spla.splu(A)
    
#     # 3. Wrap the solve method in a LinearOperator representing A^-1
#     n = A.shape[0]
#     A_inv_op = spla.LinearOperator(shape=(n, n), matvec=lu.solve, rmatvec=lambda x: lu.solve(x, trans='T'))
    
#     # 4. Estimate the 1-norm of A^-1 using Hager's algorithm (via onenormest)
#     norm_A_inv = spla.onenormest(A_inv_op)
    
#     # 5. Compute the condition number and rcond
#     cond = norm_A * norm_A_inv
#     # rcond = 1.0 / cond
    
#     return cond


# if __name__ == "__main__":
#     # Example usage of compute_CondiNum
#     from scipy.sparse import csr_matrix

#     # Create a sample sparse matrix (e.g., a tridiagonal matrix)
#     n = 1000
#     A = csr_matrix(np.diag(2 * np.ones(n)) + np.diag(-1 * np.ones(n - 1), k=1) + np.diag(-1 * np.ones(n - 1), k=-1))

#     condi_num = compute_CondiNum(A)
#     print(f"Estimated condition number: {condi_num}")

#     condi_num1= compute_CondiNum1(A)
#     print(f"Estimated condition number11: {condi_num1:.2e}")
#     nnn