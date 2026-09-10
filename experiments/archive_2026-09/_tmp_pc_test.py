from firedrake.petsc import PETSc
import numpy as np

n = 5
A = PETSc.Mat().createAIJ(size=(n, n), nnz=(2, 2), comm=PETSc.COMM_SELF)
A.setUp()
for i in range(n):
    A.setValue(i, i, 1.0)
A.assemble()
b = A.createVecLeft()
with b.getArray() as ba:
    ba[:] = [1.0, 2.0, 3.0, 4.0, 5.0]
ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
ksp.setTolerances(rtol=1e-6, max_it=1000)
ksp.setGMRESRestart(250)
ksp.setOperators(A)
pc = ksp.getPC()
pc.setType(PETSc.PC.Type.PYTHON)
pc.setPythonContext(BlockLUPC(n, klu, B, BT, ssolver))
x = A.createVecRight()
ksp.setUp()
try:
    ksp.solve(b, x)
    print("SOLVE OK its=", ksp.getIterationNumber(), "reason=", ksp.getConvergedReason())
    with x.getArray(readonly=True) as xa:
        print("x =", xa)
except Exception as e:
    print("SOLVE RAISED:", type(e).__name__, e)
