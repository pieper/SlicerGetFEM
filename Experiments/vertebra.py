import numpy
import os

"""
exec(open("vertebra.py").read())
"""

exec(open("../util_addon.py").read())

loadSteps = 3
loadValue = 15

baseDir = "/mnt/slicer/pieper/vertebra"

try:
    slicer.util.getNode("Cropped*3145mm")
except slicer.util.MRMLNodeNotFoundException:
    ctVolumePath = slicer.util.loadVolume(f"{baseDir}/Cropped volume-MD15011447-L1-0.3145mm.nrrd")
    ctVolume = slicer.util.loadVolume(f"{baseDir}/Cropped volume-MD15011447-L1-0.3145mm.nrrd")
    segmentation = slicer.util.loadSegmentation(f"{baseDir}/Segmentation-MD15011447-L1-0.3145mm.seg.nrrd")
    #inputMesh = slicer.util.loadModel(f"{baseDir}/tetgen-mesh.vtk")
    inputMesh = slicer.util.loadModel(f"{baseDir}/mesh-MD15011447A-L1-transferred.vtk")


meshNodePoints = slicer.util.arrayFromModelPoints(inputMesh)
meshPointIDs = arrayFromModelCellIds(inputMesh)

getFEMDir = slicer.util.tempDirectory()
meshArrays = numpy.array([meshNodePoints, meshPointIDs], dtype=object)
numpy.save(f"{getFEMDir}/vertebra.npy", meshArrays, allow_pickle=True)

getFEMScript = "vertebra.getfem.py"
getFEMScriptPath = getFEMDir + "/" + getFEMScript
open(getFEMScriptPath, "w").write(f'''
"""  This is based on the GetFEM demo_tripod.py and lung.py
"""
import numpy as np
import getfem as gf

#degree = 2
#linear = False
#incompressible = False # ensure that degree > 1 when incompressible is on..

degree = 1
linear = True
incompressible = False # ensure that degree > 1 when incompressible is on..

meshNodePoints, meshPointIDs = np.load("/work/vertebra.npy", allow_pickle=True)

m=gf.Mesh('empty', 3)

# remove per-cell node count (e.g. [4, 0, 3, 6, 7] -> [0, 3, 6, 7] for each cell)
tetraCount = meshPointIDs.shape[0]
tetraIDs = meshPointIDs.reshape(int(tetraCount/5),5).T[1:].T
tetraPoints = meshNodePoints[tetraIDs]

tetraPoints = np.swapaxes(tetraPoints, 1, 2)
# TODO: documentation claims this can be done in a single call without loop
for tetra in tetraPoints:
    m.add_convex(gf.GeoTrans("GT_PK(3,1)"), tetra)

mfu=gf.MeshFem(m,3) # displacement
mfp=gf.MeshFem(m,1) # pressure
mfd=gf.MeshFem(m,1) # data
mim=gf.MeshIm(m, gf.Integ('IM_TETRAHEDRON(5)')) # Mesh Integration Method

mfu.set_fem(gf.Fem('FEM_PK(3,%d)' % (degree,)))
mfd.set_fem(gf.Fem('FEM_PK(3,0)'))
mfp.set_fem(gf.Fem('FEM_PK_DISCONTINUOUS(3,0)'))

print('nbcvs=%d, nbpts=%d, qdim=%d, fem = %s, nbdof=%d' % \
      (m.nbcvs(), m.nbpts(), mfu.qdim(), mfu.fem()[0].char(), mfu.nbdof()))

boxThickness = 0.1
right = tetraPoints[:,0].max()
left = tetraPoints[:,0].min()
front = tetraPoints[:,1].max()
back = tetraPoints[:,1].min()
top = tetraPoints[:,2].max()
bottom = tetraPoints[:,2].min()
ftop = m.outer_faces_in_box([left,back,top],[right,front,top-boxThickness])
fbot = m.outer_faces_in_box([left,back,bottom],[right,front,bottom+boxThickness])

NEUMANN_BOUNDARY = 1
DIRICHLET_BOUNDARY = 2

m.set_region(NEUMANN_BOUNDARY,ftop)
m.set_region(DIRICHLET_BOUNDARY,fbot)

E=1e3
Nu=0.3
Lambda = E*Nu/((1+Nu)*(1-2*Nu))
Mu =E/(2*(1+Nu))


for step in range({loadSteps}):

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

    # apply load to top of vertebra
    md.add_initialized_data('VolumicData', [0,0,-10]);
    md.add_source_term_brick(mim, 'u', 'VolumicData');

    load = step / ({loadSteps}-1) * {loadValue}
    md.set_variable('VolumicData', [0.,0,load]);

    # Attach the bottom of vertebra to plate
    md.add_Dirichlet_condition_with_multipliers(mim, 'u', mfu, DIRICHLET_BOUNDARY);

    print('running solve...')
    md.solve('noisy', 'max iter', 1);
    U = md.variable('u');
    print(step, load, 'solve done!')

    mfdu=gf.MeshFem(m,1)
    mfdu.set_fem(gf.Fem('FEM_PK_DISCONTINUOUS(3,1)'))
    if linear:
      VM = md.compute_isotropic_linearized_Von_Mises_or_Tresca('u','clambda','cmu', mfdu);
    else:
      VM = md.compute_finite_strain_elasticity_Von_Mises(lawname, 'u', 'params', mfdu);
    print('Von Mises range: ', VM.min(), VM.max())

    # export results to VTK
    fileName = 'vertebra-result.mesh_'+str(step)+'.vtk'
    mfu.export_to_vtk(fileName, 'ascii', mfdu,  VM, 'Von Mises Stress', mfu, U, 'Displacement')

gf.memstats()
''')

cmd = f"docker run --rm -v {getFEMDir}:/work -t -i getfemdoc/getfem:v5.4 /venv/bin/python3 /work/{getFEMScript}"
print(cmd)
os.system(cmd)

properties = {
    "coordinateSystem" : slicer.vtkMRMLStorageNode.CoordinateSystemRAS
}

# Create a sequence of loads
sequenceNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", "Loading sequence")

for step in range(loadSteps):
    properties['name'] = f"load step {step}"
    stepMesh = slicer.util.loadNodeFromFile(f"{getFEMDir}/vertebra-result.mesh_{step}.vtk",
                                            filetype="ModelFile", properties=properties)
    sequenceNode.SetDataNodeAtValue(stepMesh, str(step))

sequenceBrowserNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode", "Load Browser")
sequenceBrowserNode.AddSynchronizedSequenceNode(sequenceNode)
slicer.modules.sequences.toolBar().setActiveBrowserNode(sequenceBrowserNode)


TODO = """
resultMesh = slicer.util.loadNodeFromFile(f"{getFEMDir}/vertebra-result.mesh_5.vtk",
                                            filetype="ModelFile", properties=properties)

modelPoints = slicer.util.arrayFromModelPoints(resultMesh)
originalPositions = numpy.array(modelPoints)
displacement = slicer.util.arrayFromModelPointData(resultMesh, "Displacement")

probeGrid = vtk.vtkImageData()
probeDimension = 10
probeGrid.SetDimensions(probeDimension, probeDimension, probeDimension)
probeGrid.AllocateScalars(vtk.VTK_DOUBLE, 1)
meshBounds = [0]*6
inputMesh.GetRASBounds(meshBounds)
probeGrid.SetOrigin(meshBounds[0], meshBounds[2], meshBounds[4])
probeSize = (meshBounds[1] - meshBounds[0], meshBounds[3] - meshBounds[2], meshBounds[5] - meshBounds[4])
probeGrid.SetSpacing(probeSize[0]/probeDimension, probeSize[1]/probeDimension, probeSize[2]/probeDimension)

probeFilter = vtk.vtkProbeFilter()
probeFilter.SetInputData(probeGrid)
probeFilter.SetSourceData(resultMesh.GetUnstructuredGrid())
probeFilter.SetPassPointArrays(True)
probeFilter.Update()

probeImage = probeFilter.GetOutputDataObject(0)
probeArray = vtk.util.numpy_support.vtk_to_numpy(probeImage.GetPointData().GetArray("Displacement"))
probeArray = numpy.reshape(probeArray, (probeDimension,probeDimension,probeDimension,3))
displacementGridNode = addGridTransformFromArray(-1 * probeArray, name="Displacement")
displacementGrid = displacementGridNode.GetTransformFromParent().GetDisplacementGrid()
displacementGrid.SetOrigin(probeImage.GetOrigin())
displacementGrid.SetSpacing(probeImage.GetSpacing())
gridArray = slicer.util.arrayFromGridTransform(displacementGridNode)
originalGridArray = numpy.array(gridArray)

#displacementVolume = addVolumeFromGridTransform(displacementGridNode)
#displacementVolume.SetName("Displacement Volume")
"""

""" animator:

def onValueChanged(int):
    global modelPoints, originalPositions, slider, displacement, resultMesh
    #global gridArray, originalGridArray, displacementGridNode
    animationFactor = slider.value / slider.maximum
    modelPoints[:] = originalPositions + animationFactor * displacement
    #gridArray[:] = animationFactor * originalGridArray
    slicer.util.arrayFromModelPointsModified(resultMesh)
    #slicer.util.arrayFromGridTransformModified(displacementGridNode)

slider = qt.QSlider()
slider.orientation = 1
slider.size = qt.QSize(500, 200)
slider.connect("valueChanged(int)", onValueChanged)
slider.show()

go = True
frame = 0
direction = 1
def animate():
    global frame, slider, direction
    slider.value = frame
    frame += direction
    if frame >= 99:
        direction = -1
    if frame <= 0:
        direction = 1
    if go:
        qt.QTimer.singleShot(10, animate)

animate()
"""

