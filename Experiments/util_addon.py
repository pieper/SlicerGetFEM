
def arrayFromModelCellIds(modelNode):
    """Return cell id array of a continuum mesh model node as numpy array.

    This method is used for models that have a vtkUnstructuredGrid.
    See :py:meth:`arrayFromModelPolyIds` for models with surface (vtkPolyData)

    These ids are the following format:
    [ n(0), i(0,0), i(0,1), ... i(0,n(00),..., n(j), i(j,0), ... i(j,n(j))...]
    where n(j) is the number of vertices in cell j
    and i(j,k) is the index into the vertex array for vertex k of poly j.

    As described here:
    https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf

    .. warning:: Important: memory area of the returned array is managed by VTK,
      therefore values in the array may be changed, but the array must not be reallocated.
      See :py:meth:`arrayFromVolume` for details.
    """
    import vtk.util.numpy_support
    arrayVtk = modelNode.GetUnstructuredGrid().GetCells().GetData()
    narray = vtk.util.numpy_support.vtk_to_numpy(arrayVtk)
    return narray
