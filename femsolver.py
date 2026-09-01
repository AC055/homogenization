# this file is to activate the solvers (direct fem, XMesh and cut fem) as well as the error function to obtain the results
from loadLevelSet import levelSetAdapter

import os
import skfem as fem
from skfem.models.poisson import dot,grad
from skfem.helpers import ddot, sym_grad, eye, trace
#from scipy.sparse.linalg import spsolve
from skfem import asm, BilinearForm, Functional
import numpy as np
import meshio
# from fictDom import fictitiousDomainMesher #cutFEM
from postProcessing import ResultVisualizer
from skfem.models.elasticity import lame_parameters, linear_elasticity


# ==========================================
#   Error computation
# ==========================================
def compute_error(mesh, femSolution, exactSolution, cutFEM=False, type='L2', grad=None):
    e = fem.ElementTriP1()
    basis = fem.Basis(mesh, e)

    if type == 'L2':
        @fem.Functional
        def errorL2Sqrd(w):
            xy = w.x
            uh = w['uh']
            u = exactSolution.get_val(xy[0], xy[1])  
            return (uh - u) ** 2

        return np.sqrt(errorL2Sqrd.assemble(basis, uh=basis.interpolate(femSolution)))

    elif type == 'H1':
        @fem.Functional
        def errorH1Sqrd(w):
            xy = w.x
            grad_u = exactSolution.get_grad(xy)
            if cutFEM and grad is not None:
                grad_uh = grad
            else:
                grad_uh = w['uh'].grad
            return np.linalg.norm(grad_uh - grad_u, axis=0) ** 2

        return np.sqrt(errorH1Sqrd.assemble(basis, uh=basis.interpolate(femSolution)))

    elif type == 'MaxH1':
        if cutFEM and grad is not None:
            grad_uh = grad
        else:
            grad_uh = basis.interpolate(femSolution).grad

        quad_pts = basis.mapping.F(basis.quadrature[0])
        xy = quad_pts[0], quad_pts[1]
        grad_uex = exactSolution.get_grad(xy)

        quad_pts_diff = np.abs(grad_uh - grad_uex)
        per_element_diff = np.sum(quad_pts_diff, axis=2) / len(basis.quadrature[1])
        per_element_diff = np.sqrt(per_element_diff[0]**2 + per_element_diff[1]**2)
        return quad_pts_diff.max()
    else:
        raise ValueError("Unknown error type")


# ==========================================
#   Mesh loader
# ==========================================

def load_meshio(filename):
    mesh = meshio.read(filename)
    nodes = mesh.points[:, :2].T
    if 'triangle' in mesh.cells_dict:
        elements = mesh.cells_dict['triangle'].T
    else:
        elements = next(block.data for block in mesh.cells if block.type == "triangle").T
    return nodes, elements


# ==========================================
#   FEM solver
# ==========================================
@BilinearForm
def mtx_A(u, v, w):
    # two material case
    return w.alpha*dot(grad(u), grad(v))  
    # return dot(grad(u), grad(v))

@Functional
def average_flux_x(w):
    grad_u = w['u'].grad
    return w.alpha * grad_u[0]





'''
class FEMSolver:
    def __init__(self, filename, problem):
        self.filename = filename
        self.problem = problem
        self.nodes, self.elements = load_meshio(filename)
        self.mesh = None
        self.basis = None
        self_inner_nodes = None
        self.inner_elements = None
        self.alpha = None

    def setup_basis(self):
        self.mesh = fem.MeshTri(self.nodes, self.elements)
        self.basis = fem.Basis(self.mesh, fem.ElementTriP1())
     
        

    # def get_boundaries(self):
    #     return self.basis.get_dofs({
    #         'up':    lambda x: x[1] == x[1].max(),
    #         'down':  lambda x: x[1] == x[1].min(),
    #         'left':  lambda x: x[0] == x[0].min(),
    #         'right': lambda x: x[0] == x[0].max()
    #     })

    # before is incorrect for sf is because the extreme points do not capture the inner bounday nodes so that when apply the utils.solve the
    # inner nodes are not captured to be fixed by MMS
    def get_boundaries(self):
        return self.basis.get_dofs() # allow the capture of all the boundary nodes
    

class DirectFEMSolver(FEMSolver):
    def solve(self, visual = False):
        self.setup_basis()
        basis = self.basis

        A = asm(mtx_A, basis)
        source_form = self.problem.get_source_form()
        b = asm(source_form, basis)


        dofsBNDs = self.get_boundaries()
        uex = basis.project(lambda x: self.problem.get_val(x[0], x[1]))

        u = fem.utils.solve(*fem.utils.condense(A, b, x=uex, D=dofsBNDs))

        errs = {
            'L2':  compute_error(self.mesh, u, self.problem, type='L2'),
            'H1':  compute_error(self.mesh, u, self.problem, type='H1')
            # 'Max': compute_error(self.mesh, u, self.problem, type='MaxH1')
        }

        if visual:
            base = os.path.basename(self.filename) 
            ResultVisualizer().visual_plot(
                                                self.basis, u, uex, 
                                                filename=self.filename,
                                                title = base,
                                                save_vtk= True,
                                                show = True
                                            )
        return errs, basis.N


class XMeshSolver(FEMSolver): 
    def __init__(self, filename, problem):
        super().__init__(filename, problem)
        self._modify_mesh_by_level_set()

    def _modify_mesh_by_level_set(self):
        xy = self.nodes.T       # Shape is now strictly (N, 2)
        t2v = self.elements.T   # meshio is 0-based and gmsh is 1-based
        
        all_edges = t2v[:, [[0, 1], [1, 2], [2, 0]]].reshape((-1, 2))
        e2v = np.unique(np.sort(all_edges), axis=0)

        lsVal = self.problem.level_set(xy)
        
       
        intersect_edge = e2v[(lsVal[e2v[:, 0]] * lsVal[e2v[:, 1]]) < 0]
        lsVal_safe = np.where(np.abs(lsVal) < 1e-10, 0, lsVal)
        
        
        xyz = xy[intersect_edge]
        v0, v1 = lsVal_safe[intersect_edge[:, 0]], lsVal_safe[intersect_edge[:, 1]]
        ratio = v0 / (v0 - v1)
        iso_zero = (1 - ratio[:, np.newaxis]) * xyz[:, 0] + ratio[:, np.newaxis] * xyz[:, 1]

        
        value = np.abs(lsVal_safe[intersect_edge])
        id_min = np.argmin(value, axis=1)
        activate_nodes = intersect_edge[np.arange(len(intersect_edge)), id_min].reshape(-1, 1)

        
        dist = np.linalg.norm(xy[activate_nodes][:, 0, :] - iso_zero, axis=1)
        unique_node = np.unique(activate_nodes)
        
        selected_indices = [np.where(activate_nodes == node)[0][np.argmin(dist[np.where(activate_nodes == node)[0]])] for node in unique_node]

    
        xy[unique_node] = iso_zero[selected_indices]
        lsVal_safe[unique_node] = 0.0

        all_edges_tri = t2v[:, [[0, 1], [1, 2], [2, 0]]]
        
        ls_pair = lsVal_safe[all_edges_tri]
        mean_ls_ele = np.mean(ls_pair, axis=1)
        
        # Isolate elements based on physical problem condition
        outside_ls = np.where(self.problem.trim_condition(mean_ls_ele))[0]
        ele_trim = t2v[outside_ls]
        
        unique_nodes_trim, inv = np.unique(ele_trim.flatten(), return_inverse=True)
        
      
        out_nodes = xy[unique_nodes_trim].T 
        base_ele = inv.reshape(ele_trim.shape).T

        self.nodes = out_nodes
        self.elements = base_ele
       
        


    def solve(self,visual = False):
        self.setup_basis() 
        basis = self.basis

        self.mesh.save('xmesh_sf_001.msh')

        A = asm(mtx_A, basis)

        # condiNUm = ResultVisualizer.compute_CondiNum(self.problem, A)
        # import postProcessing
        # condiNUm = postProcessing.compute_CondiNum(A)
        # print(f"Condition number of the stiffness matrix: {condiNUm:.2e}")

        source_form = self.problem.get_source_form()
        b = asm(source_form, basis)
        # print("hereeeeeeeeeeeee ")
        dofsBNDs = self.get_boundaries()

        uex = basis.project(lambda x: self.problem.get_val(x[0], x[1]))

        u = fem.utils.solve(*fem.utils.condense(A, b, x=uex, D=dofsBNDs))



        errs = {
            'L2':  compute_error(self.mesh, u, self.problem, type='L2'),
            'H1':  compute_error(self.mesh, u, self.problem, type='H1')
            # 'Max': compute_error(self.mesh, u, self.problem, type='MaxH1')
        }


        if visual:
            base = os.path.basename(self.filename) #self.filename 
            ResultVisualizer().visual_plot(
                                                self.basis, u, uex,
                                                filename = "results.vtk",
                                                title = base,
                                                save_vtk= True,
                                                show = True
                                            )
        return errs,basis.N


# two material cases

'''
class FEMSolver:
    def __init__(self, filename, problem):
        self.filename = filename
        self.problem = problem
        self.nodes, self.elements = load_meshio(filename)
        self.mesh = None
        self.basis = None
        self_inner_nodes = None
        self.inner_elements = None
        # self.innner_nodes, self.inner_elements = self.nodes, self.elements
        

    def setup_basis(self):
        self.mesh = fem.MeshTri(self.nodes, self.elements)
        e = fem.ElementVector(fem.ElementTriP1())
        self.basis = fem.Basis(self.mesh, e)
        # self.basis = fem.Basis(self.mesh, fem.ElementTriP1())

        
    # before is incorrect for sf is because the extreme points do not capture the inner bounday nodes so that when apply the utils.solve the
    # inner nodes are not captured to be fixed by MMS
    def get_boundaries(self):
        return self.basis.get_dofs() # allow the capture of all the boundary nodes

    def compute_heat_flux(self, basis, alpha_interpolate, u):
        # Compute the effective conductivity using the formula K_eff = (1/|Ω|) * ∫_Ω α(x) * ∇u(x) dx
        uh = basis.interpolate(u)
        q = average_flux_x.assemble(
                basis,
                u=uh,
                alpha=alpha_interpolate
            )
        area = 1.
        q /= area
        
        return q

    def compute_stress(self,basis, u, lam, mu):
        uh = basis.interpolate(u)
        eps = sym_grad(uh)
        return 2.0 * mu * eps + lam * eye(trace(eps), eps.shape[0])

    # @Functional
    # def compute_stress1(w):
    #     return w['stress']


    # def average_stress(self, basis, stress):
    def average_stress(self, basis, lam, mu, u):
        uh = basis.interpolate(u)
        eps = sym_grad(uh)
        stress = 2.0 * mu * eps + lam * eye(trace(eps), eps.shape[0])
        area = 1.
        @Functional
        def local_xx(w):
            return w["stress"][0,0]

        @Functional
        def local_yy(w):
            return w["stress"][1,1]

        @Functional
        def local_xy(w):
            return w["stress"][0,1]

        assemble_xx = local_xx.assemble(basis, stress=stress)
        assemble_yy = local_yy.assemble(basis, stress=stress)
        assemble_xy = local_xy.assemble(basis, stress=stress)

        average_stress = np.array([assemble_xx, assemble_yy, assemble_xy])/area
        
        return average_stress

class XMeshSolver(FEMSolver): 

    def __init__(self, filename, problem):
        super().__init__(filename, problem)
        # self.xy, self.ls, self.iso_zero, self.mesh = self._modify_mesh_by_level_set()
        self._modify_mesh_by_level_set()

    def _modify_mesh_by_level_set(self):
        xy = self.nodes.T       # Shape is now strictly (N, 2)
        t2v = self.elements.T   # meshio is 0-based and gmsh is 1-based
        

        all_edges = t2v[:, [[0, 1], [1, 2], [2, 0]]].reshape((-1, 2))
        e2v = np.unique(np.sort(all_edges), axis=0)

        lsVal = levelSetAdapter(xy)
        
       
        intersect_edge = e2v[(lsVal[e2v[:, 0]] * lsVal[e2v[:, 1]]) < 0]
        lsVal_safe = np.where(np.abs(lsVal) < 1e-10, 0, lsVal)
        
        
        xyz = xy[intersect_edge]
        v0, v1 = lsVal_safe[intersect_edge[:, 0]], lsVal_safe[intersect_edge[:, 1]]
        ratio = v0 / (v0 - v1)
        iso_zero = (1 - ratio[:, np.newaxis]) * xyz[:, 0] + ratio[:, np.newaxis] * xyz[:, 1]

        # Plot the iso zero line before node movement on base mesh
        # ResultVisualizer.plot_level_set(self, xy, t2v, iso_zero, lsVal, "Before node movement")
    
        
        value = np.abs(lsVal_safe[intersect_edge])
        id_min = np.argmin(value, axis=1)
        activate_nodes = intersect_edge[np.arange(len(intersect_edge)), id_min].reshape(-1, 1)

        
        dist = np.linalg.norm(xy[activate_nodes][:, 0, :] - iso_zero, axis=1)
        unique_node = np.unique(activate_nodes)
        
        selected_indices = [np.where(activate_nodes == node)[0][np.argmin(dist[np.where(activate_nodes == node)[0]])] for node in unique_node]

        # xy_origin = xy.copy() # save it for future restore the node location
    
        xy[unique_node] = iso_zero[selected_indices] # needs to be modified 
        iso_zero = iso_zero[selected_indices]
       

        # #validated 
         # ls = levelSetAdapter(xy)
        # self.setup_basis()
        # return xy, ls, iso_zero, self.mesh 
       

        lsVal_safe[unique_node] = 0.0

        all_edges_tri = t2v[:, [[0, 1], [1, 2], [2, 0]]]
        
        ls_pair = lsVal_safe[all_edges_tri]
        mean_ls_ele = np.mean(ls_pair, axis=1)
        
        # Isolate elements based on physical problem condition
        outside_ls = np.where(self.problem.trim_condition(mean_ls_ele))[0]
        inside_ls  = np.setdiff1d(np.arange(t2v.shape[0]), outside_ls)
        ele_outer = t2v[outside_ls]
        ele_inter = t2v[inside_ls]
        
        unique_nodes_outer, inv = np.unique(ele_outer.flatten(), return_inverse=True)
        out_nodes = xy[unique_nodes_outer].T 
        # base_ele = inv.reshape(ele_outer.shape).T
        # base_ele = ele_outer.T

        # Plot the iso zero line after node movement on base mesh
        # ResultVisualizer.plot_level_set(self.problem, xy, t2v, xy[unique_node], self.problem.level_set(xy), "After node movement")
        self.inner_nodes = self.nodes.T[np.setdiff1d(np.arange(self.nodes.shape[1]), out_nodes)].T
        self.inner_elements = ele_inter.T
        self.inner_elements_id = inside_ls
        self.outer_elements_id = outside_ls

        # self.nodes = out_nodes
        # self.elements = base_ele
        self.iso_zero = iso_zero 

    

  


    # with homhogenization for two material cases
    def solve(self,visual = True):
        self.setup_basis() 
        # basis = self.basis
        basis = fem.Basis(self.mesh,fem.ElementVector(fem.ElementTriP1()))

        # two material case with different alpha values, outer is always 1
        alpha = np.ones(self.mesh.nelements)
        alpha[self.inner_elements_id] = self.problem.k1

        # linear elastic case with two different materials
        E = np.ones(self.mesh.nelements)
        nu = np.ones(self.mesh.nelements)
        E[self.inner_elements_id] = self.problem.E1
        nu[self.inner_elements_id] = self.problem.nu1
        E[self.outer_elements_id] = self.problem.E2
        nu[self.outer_elements_id] = self.problem.nu2


        # interpolate the alpha value on element basis
        basis0 = basis.with_element(fem.ElementTriP0())
        E_interpolate = basis0.interpolate(E) 
        nu_interpolate = basis0.interpolate(nu)
        lam, mu = lame_parameters(E_interpolate, nu_interpolate)
        # alpha_interpolate = basis0.interpolate(alpha)

        # self.mesh.save('xmesh001.msh')

        @BilinearForm
        def stiffness(u, v, w):
            lam = w['lam']
            mu = w['mu']
            
            eps = sym_grad(u)
            sigma = 2. * mu * eps + lam * eye(trace(eps), eps.shape[0])
            return ddot(sigma, sym_grad(v))

        A = asm(stiffness, basis, lam = lam, mu = mu)


        source_form = self.problem.get_source_form()
        # b = asm(source_form, basis)

        dofsBNDs= self.get_boundaries()
        
        Ux = basis.project(lambda x: np.array([x[0], np.zeros_like(x[0])])) 
        Uy = basis.project(lambda x: np.array([np.zeros_like(x[1]), x[1]])) 
        Uxy = basis.project(lambda x: np.array([0.5*x[1], 0.5*x[0]]))

        # numerical displacement result
        u_x = fem.utils.solve(*fem.utils.condense(A,x=Ux, D = dofsBNDs))
        u_y = fem.utils.solve(*fem.utils.condense(A,x=Uy, D = dofsBNDs))
        u_xy = fem.utils.solve(*fem.utils.condense(A,x=Uxy, D = dofsBNDs))
        
        # calculate the average stress field, which is also the effective C tensor as the input strain is in teh mode of [x,0,0] [0,y,0] [x,y,0]
        avg_stress1 = self.average_stress(self.basis, lam, mu, u_x)
        avg_stress2 = self.average_stress(self.basis, lam, mu, u_y)
        avg_stress3 = self.average_stress(self.basis, lam, mu, u_xy)
        C_eff = np.column_stack((avg_stress1, avg_stress2, avg_stress3))
        print(C_eff)

        # ux = fem.utils.solve(*fem.utils.condense(A, b, x=Tx, D = dofsBNDs))
        # uy = fem.utils.solve(*fem.utils.condense(A, b, x=Ty, D = dofsBNDs))
        # # u = fem.utils.solve(*fem.utils.condense(A, b, x=uex, D=dofsBNDs))
        # in order to plot local stress field, here calculate the local stress
        stress_xx = self.compute_stress(self.basis, u_x, lam, mu)
        stress_yy = self.compute_stress(self.basis, u_y, lam, mu)
        stress_xy = self.compute_stress(self.basis, u_xy, lam, mu)

        # pick the stress field for each input strain mode
        sigma_xx_elem = np.mean(stress_xx[0,0], axis=1)
        sigma_yy_elem = np.mean(stress_yy[1,1], axis=1)
        sigma_xy_elem = np.mean(stress_xy[0,1], axis=1)

        # qx= self.compute_heat_flux(self.basis, alpha_interpolate, ux)
        # qy = self.compute_heat_flux(self.basis, alpha_interpolate, uy)
        # K_eff = np.column_stack((qx, qy))
        dof=basis.N

        # errs = {
        #     'L2':  compute_error(self.mesh, u, self.problem, type='L2')
        #     # 'H1':  compute_error(self.mesh, u, self.problem, type='H1'),
        #     # 'Max': compute_error(self.mesh, u, self.problem, type='MaxH1')
        # }

        if visual:
                ResultVisualizer.plot_element_field(
                                            self.mesh,
                                            sigma_xy_elem,
                                            title=r'Local $\sigma_{xy}$',
                                            colorbar_label=r'$\sigma_{xy}$'
                                        )
                ResultVisualizer.plot_element_field(
                                                            self.mesh,
                                                            sigma_xx_elem,
                                                            title=r'Local $\sigma_{xx}$',
                                                            colorbar_label=r'$\sigma_{xx}$'
                                                        )

                ResultVisualizer.plot_element_field(
                                            self.mesh,
                                            sigma_yy_elem,
                                            title=r'Local $\sigma_{yy}$',
                                            colorbar_label=r'$\sigma_{yy}$'
                                        )
            # base = os.path.basename(self.filename) #self.filename 
            # ResultVisualizer().visual_plot(
            #                                     self.basis, ux, uy, uex, alpha_interpolate,
            #                                     filename = "resultsKeff.vtk",
            #                                     title = base,
            #                                     save_vtk= True,
            #                                     show = True
            #                                 )

        return dof, C_eff


        


        # if visual:
        #     base = os.path.basename(self.filename) #self.filename 
        #     ResultVisualizer().visual_plot(
        #                                         self.basis, u, alpha_interpolate,
        #                                         filename = "results2mate.vtk",
        #                                         title = base,
        #                                         save_vtk= True,
        #                                         show = True
        #                                     )
        # return errs, basis.N


# cut FEM version 1, the one that has problems in convergence analysis
# class CutFEMSolver(FEMSolver):
#     def solve(self, visual = False):
#         meshio_obj = meshio.read(self.filename)
#         fictdom = fictitiousDomainMesher(meshio_obj, self.problem.level_set_cut)
  
#         shavedMesh = fictdom.constructShavedMesh()
#         shavedMesh.write('shavedMesh.vtu')
#         volFrac = fictdom.computeVolumeFraction(shavedMesh)
#         volFrac[volFrac < 1.e-12] = 1

#         self.mesh = fem.MeshTri(shavedMesh.points[:, :2].T, shavedMesh.cells_dict['triangle'].T)
#         e = fem.ElementTriP1()
#         self.basis = fem.Basis(self.mesh, e)

#         basis0 = self.basis.with_element(fem.ElementTriP0())
#         volumeFraction = basis0.interpolate(volFrac)

#         dofsBNDs = self.get_boundaries()

#         A = laplace_cut.assemble(self.basis, volumeFraction=volumeFraction)
#         source_form = self.problem.get_source_form()
#         b = asm(source_form, self.basis)

#         uex = self.basis.project(lambda x: self.problem.get_val(x[0], x[1]))
#         A, b = fem.enforce(A, b, D=dofsBNDs, x=uex)

#         u = fem.solve(A, b)


#         # confomring mesh
#         conformingMesh = fictdom.createConformingMesh()
#         cm = fem.MeshTri(conformingMesh.points[:,:2].T, conformingMesh.cells_dict['triangle'].T)
#         basis_cm = fem.Basis(cm,e)
#         ppp = self.basis.probes(cm.p)  
#         u_cm = ppp @ u



#         errs = {
#             'L2':  compute_error(cm, u_cm, self.problem, type='L2',  cutFEM=True),
#             'H1':  compute_error(cm, u_cm, self.problem, type='H1',  cutFEM=True),
#             'Max': compute_error(cm, u_cm, self.problem, type='MaxH1', cutFEM=True)
#         }



#         if visual:
#             base = os.path.basename(self.filename) 
#             ResultVisualizer().visual_plot(
#                                                 self.basis, u, uex, 
#                                                 filename=self.filename,
#                                                 title = base,
#                                                 save_vtk= True,
#                                                 show = True
#                                             )
#         return errs, self.basis.N



