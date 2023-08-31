"""
sudo docker run --rm -v `pwd`:/work -t -i getfemdoc/getfem:v5.4 /venv/bin/python3 lung.py
"""

"""  This is based on the GetFEM demo_tripod.py
"""

import numpy as np

import getfem as gf

m=gf.Mesh('import','gmsh','lung-tumor.msh') # Mesh from vtk via Gmsh
mfu=gf.MeshFem(m,3) # displacement
mfp=gf.MeshFem(m,1) # pressure
mfd=gf.MeshFem(m,1) # data
mim=gf.MeshIm(m, gf.Integ('IM_TETRAHEDRON(5)')) # Mesh Integration Method
degree = 2
linear = False
incompressible = False # ensure that degree > 1 when incompressible is on..

mfu.set_fem(gf.Fem('FEM_PK(3,%d)' % (degree,)))
mfd.set_fem(gf.Fem('FEM_PK(3,0)'))
mfp.set_fem(gf.Fem('FEM_PK_DISCONTINUOUS(3,0)'))

print('nbcvs=%d, nbpts=%d, qdim=%d, fem = %s, nbdof=%d' % \
      (m.nbcvs(), m.nbpts(), mfu.qdim(), mfu.fem()[0].char(), mfu.nbdof()))

P=m.pts()
print('test', P[1,:])

surfacePointIDs, backOfLung = np.load("surfaceRegions.npy")
pidtop=surfacePointIDs[np.where(np.logical_not(backOfLung))]
pidbot=surfacePointIDs[np.where(backOfLung)]

ftop=m.faces_from_pid(pidtop)
fbot=m.faces_from_pid(pidbot)

NEUMANN_BOUNDARY = 1
DIRICHLET_BOUNDARY = 2

m.set_region(NEUMANN_BOUNDARY,ftop)
m.set_region(DIRICHLET_BOUNDARY,fbot)

E=1e3
Nu=0.3
Lambda = E*Nu/((1+Nu)*(1-2*Nu))
Mu =E/(2*(1+Nu))

md = gf.Model('real')
md.add_fem_variable('u', mfu)
if linear:
    md.add_initialized_data('cmu', Mu)
    md.add_initialized_data('clambda', Lambda)
    md.add_isotropic_linearized_elasticity_brick(mim, 'u', 'clambda', 'cmu')
    if incompressible:
        md.add_fem_variable('p', mfp)
        md.add_linear_incompressibility_brick(mim, 'u', 'p')
else:
    md.add_initialized_data('params', [Lambda, Mu]);
    if incompressible:
        lawname = 'Incompressible Mooney Rivlin';
        md.add_finite_strain_elasticity_brick(mim, lawname, 'u', 'params')
        md.add_fem_variable('p', mfp);
        md.add_finite_strain_incompressibility_brick(mim, 'u', 'p');
    else:
        lawname = 'SaintVenant Kirchhoff';
        md.add_finite_strain_elasticity_brick(mim, lawname, 'u', 'params');
  
md.add_initialized_data('VolumicData', [0,10,0]);
md.add_source_term_brick(mim, 'u', 'VolumicData');

# Attach the back of the lung to the chest wall
md.add_Dirichlet_condition_with_multipliers(mim, 'u', mfu, DIRICHLET_BOUNDARY);

print('running solve...')
md.solve('noisy', 'max iter', 1);
U = md.variable('u');
print('solve done!')


mfdu=gf.MeshFem(m,1)
mfdu.set_fem(gf.Fem('FEM_PK_DISCONTINUOUS(3,1)'))
if linear:
  VM = md.compute_isotropic_linearized_Von_Mises_or_Tresca('u','clambda','cmu', mfdu);
else:
  VM = md.compute_finite_strain_elasticity_Von_Mises(lawname, 'u', 'params', mfdu);

# post-processing
sl=gf.Slice(('boundary',), mfu, degree)

print('Von Mises range: ', VM.min(), VM.max())

# export results to VTK
sl.export_to_vtk('lung-result-surface.mesh.vtk', 'ascii', mfdu,  VM, 'Von Mises Stress', mfu, U, 'Displacement')
mfu.export_to_vtk('lung-result.mesh.vtk', 'ascii', mfdu,  VM, 'Von Mises Stress', mfu, U, 'Displacement')

gf.memstats()
