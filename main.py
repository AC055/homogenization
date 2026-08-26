# this is the executive file that call all the defined functions and class to compute the results
# the input mesh file is stored inside the named folder at the same level of this file


from femsolver import XMeshSolver
import meshio
from postProcessing import ResultVisualizer
from definePhyProblem import CircleLaplacian, CircleLaplacian2mate


# solver = XMeshSolver('loadLevelSet_sate.msh')

# nodes, ls, iso_zero, mesh = solver.xy, solver.ls, solver.iso_zero, solver.mesh
# # print(mesh.type)
# ResultVisualizer().plot_level_set(
#     nodes,
#     solver.elements.T,
#     iso_zero,
#     ls,
#     title="After node movement"
# )
# # meshio.write('xmesh003.msh', mesh)
# # meshio.save_mesh('xmesh003.msh', mesh)

# ResultVisualizer().skewness_map(
#     mesh,
#     'skewness_sate.vtk'
# )


#-----------------------thermo homogenization-----------------------
problem = CircleLaplacian2mate()
thermo_solver = XMeshSolver('loadLevelSet_sate.msh', problem)
dof, Keff = thermo_solver.solve(visual=True)