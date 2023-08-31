import numpy
import os

exec(open("util_addon.py").read())

lungVolume = slicer.util.loadVolume("Lung.nrrd")
nolungVolume = slicer.util.loadVolume("no-lung.nrrd")

inputMesh = slicer.util.loadModel("lung-tumor.mesh.vtk")
surfaceFilter = vtk.vtkDataSetSurfaceFilter()
surfaceFilter.SetInputData(inputMesh.GetUnstructuredGrid())
surfaceFilter.SetPassThroughPointIds(True)
surfaceFilter.Update()
surfaceMesh = surfaceFilter.GetOutputDataObject(0)

surfacePointIDs = vtk.util.numpy_support.vtk_to_numpy(surfaceMesh.GetPointData().GetArray("vtkOriginalPointIds"))
surfacePoints = vtk.util.numpy_support.vtk_to_numpy(surfaceMesh.GetPoints().GetData())

paSurfacePoints = surfacePoints.transpose()[1]
divisionPlane = paSurfacePoints.min() + 0.25 * (paSurfacePoints.max() - paSurfacePoints.min())
backOfLung = (paSurfacePoints < divisionPlane)
numpy.save("surfaceRegions.npy", [surfacePointIDs, backOfLung])


os.system("sudo docker run --rm -v `pwd`:/work -t -i getfemdoc/getfem:v5.4 /venv/bin/python3 lung.getfem.py")

resultMesh = slicer.util.loadModel("lung-result.mesh.vtk")
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


def onValueChanged(int):
    global modelPoints, originalPositions, slider, displacement, resultMesh
    global gridArray, originalGridArray, displacementGridNode
    animationFactor = slider.value / slider.maximum
    modelPoints[:] = originalPositions + animationFactor * displacement
    gridArray[:] = animationFactor * originalGridArray
    slicer.util.arrayFromModelPointsModified(resultMesh)
    slicer.util.arrayFromGridTransformModified(displacementGridNode)

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
    


