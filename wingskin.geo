Merge "wingskin.stp";
Mesh.CharacteristicLengthMin = 0.05;
Mesh.CharacteristicLengthMax = 0.2;

Field[1] = Distance;
Field[1].SurfacesList = {1, 2};

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = 0.02;
Field[2].SizeMax = 0.15;
Field[2].DistMin = 0.05;
Field[2].DistMax = 0.3;

Background Field = 2;

Physical Curve("root", 11) = {6, 3};
Physical Curve("tail", 12) = {5, 1};
Physical Surface("upper", 14) = {2};
Physical Surface("lower", 15) = {1};
