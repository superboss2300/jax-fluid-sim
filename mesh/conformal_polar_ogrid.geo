SetFactory("OpenCASCADE");

r = 0.5;
r_far = 3.0;
dz = 0.1;

p_c = newp; Point(p_c) = {0, 0, 0};

// Cylinder boundary points
p_cyl_r = newp; Point(p_cyl_r) = {r, 0, 0};
p_cyl_t = newp; Point(p_cyl_t) = {0, r, 0};
p_cyl_l = newp; Point(p_cyl_l) = {-r, 0, 0};
p_cyl_b = newp; Point(p_cyl_b) = {0, -r, 0};

// Outer boundary points
p_far_r = newp; Point(p_far_r) = {r_far, 0, 0};
p_far_t = newp; Point(p_far_t) = {0, r_far, 0};
p_far_l = newp; Point(p_far_l) = {-r_far, 0, 0};
p_far_b = newp; Point(p_far_b) = {0, -r_far, 0};

// Cylinder arcs
c_cyl_tr = newc; Circle(c_cyl_tr) = {p_cyl_r, p_c, p_cyl_t};
c_cyl_tl = newc; Circle(c_cyl_tl) = {p_cyl_t, p_c, p_cyl_l};
c_cyl_bl = newc; Circle(c_cyl_bl) = {p_cyl_l, p_c, p_cyl_b};
c_cyl_br = newc; Circle(c_cyl_br) = {p_cyl_b, p_c, p_cyl_r};

// Outer arcs
c_far_tr = newc; Circle(c_far_tr) = {p_far_r, p_c, p_far_t};
c_far_tl = newc; Circle(c_far_tl) = {p_far_t, p_c, p_far_l};
c_far_bl = newc; Circle(c_far_bl) = {p_far_l, p_c, p_far_b};
c_far_br = newc; Circle(c_far_br) = {p_far_b, p_c, p_far_r};

// 4 pure orthogonal radial lines
l_rad_r = newl; Line(l_rad_r) = {p_cyl_r, p_far_r};
l_rad_t = newl; Line(l_rad_t) = {p_cyl_t, p_far_t};
l_rad_l = newl; Line(l_rad_l) = {p_cyl_l, p_far_l};
l_rad_b = newl; Line(l_rad_b) = {p_cyl_b, p_far_b};

// 4 symmetrical quadrants
cl1 = newcl; Curve Loop(cl1) = {l_rad_r, c_far_tr, -l_rad_t, -c_cyl_tr};
s1 = news; Plane Surface(s1) = {cl1};

cl2 = newcl; Curve Loop(cl2) = {l_rad_t, c_far_tl, -l_rad_l, -c_cyl_tl};
s2 = news; Plane Surface(s2) = {cl2};

cl3 = newcl; Curve Loop(cl3) = {l_rad_l, c_far_bl, -l_rad_b, -c_cyl_bl};
s3 = news; Plane Surface(s3) = {cl3};

cl4 = newcl; Curve Loop(cl4) = {l_rad_b, c_far_br, -l_rad_r, -c_cyl_br};
s4 = news; Plane Surface(s4) = {cl4};

n_rad = 40;
n_circ = 28;
prog_rad = 1.05;

Transfinite Curve {l_rad_r, l_rad_t, l_rad_l, l_rad_b} = n_rad Using Progression prog_rad;
Transfinite Curve {c_cyl_tr, c_far_tr, c_cyl_tl, c_far_tl, c_cyl_bl, c_far_bl, c_cyl_br, c_far_br} = n_circ;

Transfinite Surface {s1}; Recombine Surface {s1};
Transfinite Surface {s2}; Recombine Surface {s2};
Transfinite Surface {s3}; Recombine Surface {s3};
Transfinite Surface {s4}; Recombine Surface {s4};

ext[] = Extrude {0, 0, dz} { Surface{s1, s2, s3, s4}; Layers{1}; Recombine; };
Coherence;

Physical Volume("fluid", 1) = {Volume{:}};
Mesh.ElementOrder = 2;
Mesh.MshFileVersion = 4.1;
