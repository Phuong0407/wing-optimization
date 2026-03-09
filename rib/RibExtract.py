import gmsh

Y_RIBS = [0, 750, 1500, 2250, 3000]

for Y_RIB in Y_RIBS:
  gmsh.initialize()
  gmsh.model.add("rib")

  gmsh.model.occ.importShapes("wingsolid.stp")
  gmsh.model.occ.synchronize()

  WINGSOLID = gmsh.model.getEntities(dim=3)
  CUTTING_PLANE_SIZE = 10000.0

  P1 = gmsh.model.occ.addPoint(-CUTTING_PLANE_SIZE, Y_RIB, -CUTTING_PLANE_SIZE)
  P2 = gmsh.model.occ.addPoint( CUTTING_PLANE_SIZE, Y_RIB, -CUTTING_PLANE_SIZE)
  P3 = gmsh.model.occ.addPoint( CUTTING_PLANE_SIZE, Y_RIB,  CUTTING_PLANE_SIZE)
  P4 = gmsh.model.occ.addPoint(-CUTTING_PLANE_SIZE, Y_RIB,  CUTTING_PLANE_SIZE)

  L1 = gmsh.model.occ.addLine(P1, P2)
  L2 = gmsh.model.occ.addLine(P2, P3)
  L3 = gmsh.model.occ.addLine(P3, P4)
  L4 = gmsh.model.occ.addLine(P4, P1)

  LOOP_CURVE  = gmsh.model.occ.addCurveLoop([L1, L2, L3, L4])
  plane = gmsh.model.occ.addPlaneSurface([LOOP_CURVE])
  gmsh.model.occ.synchronize()

  result, _ = gmsh.model.occ.intersect(
    [(2, plane)],
    WINGSOLID,
    removeObject=True,
    removeTool=True
  )

  gmsh.model.occ.synchronize()

  gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 2.0)
  gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5)

  gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
  gmsh.option.setNumber("Mesh.MinimumCirclePoints", 36)

  gmsh.model.mesh.generate(2)
  gmsh.write(f"rib{Y_RIB}.msh")
  gmsh.write(f"rib{Y_RIB}.stp")
  gmsh.finalize()