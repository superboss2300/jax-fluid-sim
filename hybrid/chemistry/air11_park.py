"""Park 1990 11-species, 22-reaction air mechanism for hypersonic flows.

Species: e-, N+, O+, NO+, N2+, O2+, N, O, NO, N2, O2
Reactions: Park (1993, 2001) / Mutation++ air11_Park set in SI units.
Coupled two-temperature (T, Tv) kinetic rates with detailed balance.
"""

from .nasa import make_nasa7
from .mechanism import Species, Reaction, ReactionMechanism

# Molecular weights in kg / mol (consistent with standard aerothermodynamic values)
MW_E = 5.48579909065e-7
MW_N = 0.0140067
MW_O = 0.015999
MW_NO = MW_N + MW_O
MW_N2 = 2.0 * MW_N
MW_O2 = 2.0 * MW_O

# NASA-7 polynomial data extracted from standard Glenn database (nasa7.dat)
AIR11_SPECIES = (
    Species(
        name="e-",
        molar_mass=MW_E,
        charge=-1.0,
        poly=make_nasa7(
            298.15, 1000.0, 6000.0,
            [2.5, 0.0, 0.0, 0.0, 0.0, -745.375, -11.7208122],
            [2.5, 0.0, 0.0, 0.0, 0.0, -745.375, -11.7208122],
        ),
    ),
    Species(
        name="N+",
        molar_mass=MW_N,
        charge=1.0,
        poly=make_nasa7(
            298.15, 1000.0, 6000.0,
            [2.79781357, -1.41287178e-03, 2.68686998e-06, -2.31615591e-09, 7.49585578e-13, 225548.731, 3.59928845],
            [2.51210704, 1.75367440e-06, -1.50557538e-08, 7.06834189e-12, -6.31861166e-16, 225597.004, 4.92236293],
        ),
    ),
    Species(
        name="O+",
        molar_mass=MW_O,
        charge=1.0,
        poly=make_nasa7(
            298.15, 1000.0, 6000.0,
            [2.5, 0.0, 0.0, 0.0, 0.0, 187935.284, 4.39337676],
            [2.48542028, 2.56978695e-05, -1.28833378e-08, 1.65525487e-12, 1.09933344e-16, 187940.874, 4.47425446],
        ),
    ),
    Species(
        name="NO+",
        molar_mass=MW_NO,
        charge=1.0,
        is_diatomic=True,
        theta_v=3380.0,
        poly=make_nasa7(
            298.15, 1000.0, 6000.0,
            [3.69301231, -1.34229158e-03, 2.67343395e-06, -1.02609308e-09, -6.95610492e-14, 118103.055, 3.09126691],
            [2.94587702, 1.40325260e-03, -4.95503196e-07, 7.95948973e-11, -4.72076668e-15, 118244.340, 6.70644634],
        ),
    ),
    Species(
        name="N2+",
        molar_mass=MW_N2,
        charge=1.0,
        is_diatomic=True,
        theta_v=3160.0,
        poly=make_nasa7(
            298.15, 1000.0, 6000.0,
            [3.77540711, -2.06459157e-03, 4.75752301e-06, -3.15664228e-09, 6.70509973e-13, 180481.115, 2.69322178],
            [3.58661363, 2.53071949e-04, 1.84778214e-07, -4.55257223e-11, 3.26818029e-15, 180390.994, 3.09584143],
        ),
    ),
    Species(
        name="O2+",
        molar_mass=MW_O2,
        charge=1.0,
        is_diatomic=True,
        theta_v=2700.0,
        poly=make_nasa7(
            298.15, 1000.0, 6000.0,
            [4.61017167, -6.35951952e-03, 1.42425624e-05, -1.20997923e-08, 3.70956878e-12, 139742.229, -0.201326941],
            [3.31675922, 1.11522244e-03, -3.83492556e-07, 5.72784687e-11, -2.77648381e-15, 139876.823, 5.44726469],
        ),
    ),
    Species(
        name="N",
        molar_mass=MW_N,
        charge=0.0,
        poly=make_nasa7(
            200.0, 1000.0, 6000.0,
            [2.5, 0.0, 0.0, 0.0, 0.0, 56104.638, 4.1939088],
            [2.4159429, 1.74890650e-04, -1.19023690e-07, 3.02262440e-11, -2.03609830e-15, 56133.775, 4.6496095],
        ),
    ),
    Species(
        name="O",
        molar_mass=MW_O,
        charge=0.0,
        poly=make_nasa7(
            200.0, 1000.0, 6000.0,
            [3.16826710, -3.27931884e-03, 6.64306396e-06, -6.12806624e-09, 2.11265971e-12, 29122.2592, 2.05193346],
            [2.54363697, -2.73162486e-05, -4.19029520e-09, 4.95481845e-12, -4.79553694e-16, 29226.012, 4.92229457],
        ),
    ),
    Species(
        name="NO",
        molar_mass=MW_NO,
        charge=0.0,
        is_diatomic=True,
        theta_v=2740.0,
        poly=make_nasa7(
            200.0, 1000.0, 6000.0,
            [4.21859896, -4.63988124e-03, 1.10443049e-05, -9.34055507e-09, 2.80554874e-12, 9845.09964, 2.28061001],
            [3.26071234, 1.19101135e-03, -4.29122646e-07, 6.94481463e-11, -4.03295681e-15, 9921.43132, 6.36900518],
        ),
    ),
    Species(
        name="N2",
        molar_mass=MW_N2,
        charge=0.0,
        is_diatomic=True,
        theta_v=3395.0,
        poly=make_nasa7(
            200.0, 1000.0, 6000.0,
            [3.53100528, -1.23660988e-04, -5.02999433e-07, 2.43530612e-09, -1.40881235e-12, -1046.97628, 2.96747038],
            [2.95257637, 1.39690040e-03, -4.92631603e-07, 7.86010195e-11, -4.60755204e-15, -923.948688, 5.87188762],
        ),
    ),
    Species(
        name="O2",
        molar_mass=MW_O2,
        charge=0.0,
        is_diatomic=True,
        theta_v=2270.0,
        poly=make_nasa7(
            200.0, 1000.0, 6000.0,
            [3.78245636, -2.99673416e-03, 9.84730201e-06, -9.68129509e-09, 3.24372837e-12, -1063.94356, 3.65767573],
            [3.66096065, 6.56365811e-04, -1.41149627e-07, 2.05797935e-11, -1.29913436e-15, -1215.97718, 3.41536279],
        ),
    ),
)

# Species indices
SP_E, SP_NP, SP_OP, SP_NOP, SP_N2P, SP_O2P, SP_N, SP_O, SP_NO, SP_N2, SP_O2 = range(11)

# 22 elementary reactions transcribed from Mutation++ / Park (1993, 2001)
AIR11_REACTIONS = (
    # 0: N2 + M <=> 2N + M
    Reaction(
        reactants=((SP_N2, 1),),
        products=((SP_N, 2),),
        A=7.0e15, b=-1.6, theta=113200.0,
        third_body=True,
        efficiencies=((SP_O, 4.285714285714286), (SP_N, 4.285714285714286)),
        fwd_temp="Ta", rev_temp="T",
    ),
    # 1: N2 + e- <=> 2N + e-
    Reaction(
        reactants=((SP_N2, 1), (SP_E, 1)),
        products=((SP_N, 2), (SP_E, 1)),
        A=3.0e18, b=-1.6, theta=113200.0,
        third_body=False,
        fwd_temp="Tv", rev_temp="Tv",
    ),
    # 2: O2 + M <=> 2O + M
    Reaction(
        reactants=((SP_O2, 1),),
        products=((SP_O, 2),),
        A=2.0e15, b=-1.5, theta=59360.0,
        third_body=True,
        efficiencies=((SP_O, 5.0), (SP_N, 5.0)),
        fwd_temp="Ta", rev_temp="T",
    ),
    # 3: NO + M <=> N + O + M
    Reaction(
        reactants=((SP_NO, 1),),
        products=((SP_N, 1), (SP_O, 1)),
        A=5.0e9, b=0.0, theta=75500.0,
        third_body=True,
        efficiencies=((SP_NO, 22.0), (SP_N, 22.0), (SP_O, 22.0), (SP_NP, 22.0), (SP_OP, 22.0)),
        fwd_temp="Ta", rev_temp="T",
    ),
    # 4: N2 + O <=> NO + N
    Reaction(
        reactants=((SP_N2, 1), (SP_O, 1)),
        products=((SP_NO, 1), (SP_N, 1)),
        A=5.7e6, b=0.42, theta=42938.0,
        fwd_temp="T", rev_temp="T",
    ),
    # 5: NO + O <=> O2 + N
    Reaction(
        reactants=((SP_NO, 1), (SP_O, 1)),
        products=((SP_O2, 1), (SP_N, 1)),
        A=8.4e6, b=0.0, theta=19400.0,
        fwd_temp="T", rev_temp="T",
    ),
    # 6: N + O <=> NO+ + e-
    Reaction(
        reactants=((SP_N, 1), (SP_O, 1)),
        products=((SP_NOP, 1), (SP_E, 1)),
        A=5.3e6, b=0.0, theta=31900.0,
        fwd_temp="T", rev_temp="Tv",
    ),
    # 7: O + O <=> O2+ + e-
    Reaction(
        reactants=((SP_O, 2),),
        products=((SP_O2P, 1), (SP_E, 1)),
        A=7.1e-4, b=2.7, theta=80600.0,
        fwd_temp="T", rev_temp="Tv",
    ),
    # 8: N + N <=> N2+ + e-
    Reaction(
        reactants=((SP_N, 2),),
        products=((SP_N2P, 1), (SP_E, 1)),
        A=4.4e1, b=1.5, theta=67500.0,
        fwd_temp="T", rev_temp="Tv",
    ),
    # 9: NO+ + O <=> N+ + O2
    Reaction(
        reactants=((SP_NOP, 1), (SP_O, 1)),
        products=((SP_NP, 1), (SP_O2, 1)),
        A=1.0e6, b=0.5, theta=77200.0,
        fwd_temp="T", rev_temp="T",
    ),
    # 10: N+ + N2 <=> N2+ + N
    Reaction(
        reactants=((SP_NP, 1), (SP_N2, 1)),
        products=((SP_N2P, 1), (SP_N, 1)),
        A=1.0e6, b=0.5, theta=12200.0,
        fwd_temp="T", rev_temp="T",
    ),
    # 11: O2+ + N <=> N+ + O2
    Reaction(
        reactants=((SP_O2P, 1), (SP_N, 1)),
        products=((SP_NP, 1), (SP_O2, 1)),
        A=8.7e7, b=0.14, theta=28600.0,
        fwd_temp="T", rev_temp="T",
    ),
    # 12: O+ + NO <=> N+ + O2
    Reaction(
        reactants=((SP_OP, 1), (SP_NO, 1)),
        products=((SP_NP, 1), (SP_O2, 1)),
        A=1.4e-1, b=1.9, theta=26600.0,
        fwd_temp="T", rev_temp="T",
    ),
    # 13: O2+ + N2 <=> N2+ + O2
    Reaction(
        reactants=((SP_O2P, 1), (SP_N2, 1)),
        products=((SP_N2P, 1), (SP_O2, 1)),
        A=9.9e6, b=0.0, theta=40700.0,
        fwd_temp="T", rev_temp="T",
    ),
    # 14: O2+ + O <=> O+ + O2
    Reaction(
        reactants=((SP_O2P, 1), (SP_O, 1)),
        products=((SP_OP, 1), (SP_O2, 1)),
        A=4.0e6, b=0.09, theta=18000.0,
        fwd_temp="T", rev_temp="T",
    ),
    # 15: NO+ + N <=> O+ + N2
    Reaction(
        reactants=((SP_NOP, 1), (SP_N, 1)),
        products=((SP_OP, 1), (SP_N2, 1)),
        A=3.4e7, b=-1.08, theta=12800.0,
        fwd_temp="T", rev_temp="T",
    ),
    # 16: NO+ + O2 <=> O2+ + NO
    Reaction(
        reactants=((SP_NOP, 1), (SP_O2, 1)),
        products=((SP_O2P, 1), (SP_NO, 1)),
        A=2.4e7, b=0.41, theta=32600.0,
        fwd_temp="T", rev_temp="T",
    ),
    # 17: NO+ + O <=> O2+ + N
    Reaction(
        reactants=((SP_NOP, 1), (SP_O, 1)),
        products=((SP_O2P, 1), (SP_N, 1)),
        A=7.2e6, b=0.29, theta=48600.0,
        fwd_temp="T", rev_temp="T",
    ),
    # 18: O+ + N2 <=> N2+ + O
    Reaction(
        reactants=((SP_OP, 1), (SP_N2, 1)),
        products=((SP_N2P, 1), (SP_O, 1)),
        A=9.1e5, b=0.36, theta=22800.0,
        fwd_temp="T", rev_temp="T",
    ),
    # 19: NO+ + N <=> N2+ + O
    Reaction(
        reactants=((SP_NOP, 1), (SP_N, 1)),
        products=((SP_N2P, 1), (SP_O, 1)),
        A=7.2e7, b=0.0, theta=35500.0,
        fwd_temp="T", rev_temp="T",
    ),
    # 20: O + e- <=> O+ + 2 e-
    Reaction(
        reactants=((SP_O, 1), (SP_E, 1)),
        products=((SP_OP, 1), (SP_E, 2)),
        A=3.9e27, b=-3.78, theta=158500.0,
        fwd_temp="Tv", rev_temp="T",
    ),
    # 21: N + e- <=> N+ + 2 e-
    Reaction(
        reactants=((SP_N, 1), (SP_E, 1)),
        products=((SP_NP, 1), (SP_E, 2)),
        A=2.5e28, b=-3.82, theta=168200.0,
        fwd_temp="Tv", rev_temp="T",
    ),
)


class Air11ParkMechanism(ReactionMechanism):
    """Park 1990 11-species two-temperature hypersonic air mechanism."""

    def __init__(self):
        super().__init__(species=AIR11_SPECIES, reactions=AIR11_REACTIONS)
