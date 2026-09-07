// Curved P2 sphere wind tunnel with a five-layer prismatic wall region.
// The outer field is tetrahedral; hex blocks can be attached to planar regions.

SetFactory("Built-in");
near = 0.11;
far = 0.38;
r = 0.5;

Point(1) = {0,0,0,near};
Point(2) = {r,0,0,near}; Point(3) = {0,r,0,near};
Point(4) = {-r,0,0,near}; Point(5) = {0,-r,0,near};
Point(6) = {0,0,-r,near}; Point(7) = {0,0,r,near};
Circle(1)={2,1,3}; Circle(2)={3,1,4}; Circle(3)={4,1,5}; Circle(4)={5,1,2};
Circle(5)={3,1,6}; Circle(6)={6,1,5}; Circle(7)={5,1,7}; Circle(8)={7,1,3};
Circle(9)={2,1,7}; Circle(10)={7,1,4}; Circle(11)={4,1,6}; Circle(12)={6,1,2};
Curve Loop(13)={2,8,-10}; Surface(14)={13};
Curve Loop(15)={10,3,7}; Surface(16)={15};
Curve Loop(17)={-8,-9,1}; Surface(18)={17};
Curve Loop(19)={-11,-2,5}; Surface(20)={19};
Curve Loop(21)={-5,-12,-1}; Surface(22)={21};
Curve Loop(23)={-3,11,6}; Surface(24)={23};
Curve Loop(25)={-7,4,9}; Surface(26)={25};
Curve Loop(27)={-4,12,-6}; Surface(28)={27};

old = Geometry.ExtrudeReturnLateralEntities;
Geometry.ExtrudeReturnLateralEntities = 0;
layer[] = Extrude {
  Surface{14:28:2};
  // Five cumulative heights, totaling 0.20, with outward growth.
  Layers{{1,1,1,1,1},{0.016,0.036,0.068,0.120,0.200}};
  Recombine;
};
Geometry.ExtrudeReturnLateralEntities = old;

Point(100)={-2,-2,-2,far}; Point(101)={-2,2,-2,far};
Point(102)={4,-2,-2,far}; Point(103)={4,2,-2,far};
Point(104)={-2,-2,2,far}; Point(105)={-2,2,2,far};
Point(106)={4,-2,2,far}; Point(107)={4,2,2,far};
Line(165)={100,102}; Line(166)={102,103}; Line(167)={103,101}; Line(168)={101,100};
Line(169)={100,104}; Line(170)={104,106}; Line(171)={106,102}; Line(172)={101,105};
Line(173)={105,107}; Line(174)={107,103}; Line(175)={105,104}; Line(176)={106,107};
Curve Loop(177)={167,168,165,166}; Plane Surface(178)={177};
Curve Loop(179)={166,-174,-176,171}; Plane Surface(180)={179};
Curve Loop(181)={171,-165,169,170}; Plane Surface(182)={181};
Curve Loop(183)={176,-173,175,170}; Plane Surface(184)={183};
Curve Loop(185)={172,175,-169,-168}; Plane Surface(186)={185};
Curve Loop(187)={172,173,174,167}; Plane Surface(188)={187};

Surface Loop(1001)={layer[{0:14:2}]};
Surface Loop(1002)={188,186,184,180,178,182};
Volume(1001)={1002,1001};

Physical Volume("fluid",1)={1001,layer[{1:15:2}]};
Physical Surface("inlet",11)={186};
Physical Surface("outlet",12)={180};
Physical Surface("walls",13)={178,182,184,188};
Physical Surface("sphere_wall",14)={14:28:2};

Mesh.Algorithm3D=10;
Mesh.ElementOrder=2;
Mesh.SecondOrderIncomplete=0;
Mesh.SecondOrderLinear=0;
Mesh.HighOrderOptimize=1;
Mesh.MshFileVersion=4.1;
