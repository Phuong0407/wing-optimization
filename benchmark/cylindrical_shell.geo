// Gmsh project created on Thu Feb 19 08:59:30 2026
SetFactory("OpenCASCADE");
//+
R = 1.0;
//+
L = 5.0;
//+
Circle(1) = {0.0, 0.0, 0.0, 1.0, 0, 2*Pi};
//+
Extrude {0, 0, 5} {
  Curve{1}; 
}
