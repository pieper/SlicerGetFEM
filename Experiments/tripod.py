"""
1) install getfem
`sudo apt-get install python3-getfem++`
2) checkout demos
`git clone https://github.com/getfem-doc/getfem.git`
3) run demo
```
cd getfem/interface/python
python3 demo_tripod.py
```
4) run the following in Slicer
"""

m = loadModel("tripod.vtk")
pos = arrayFromModelPoints(m)
disp = arrayFromModelPointData(m, "Displacement")
pos[:] = pos + disp
arrayFromModelPointsModified(m)

