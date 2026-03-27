import pybamm

whole_cell = ["negative electrode", "separator", "positive electrode"]

# Domains at cell centres
x_n = pybamm.SpatialVariable(
    "x_n",
    domain=["negative electrode"],
    auxiliary_domains={"secondary": "current collector"},
    coord_sys="cartesian",
)
x_s = pybamm.SpatialVariable(
    "x_s",
    domain=["separator"],
    auxiliary_domains={"secondary": "current collector"},
    coord_sys="cartesian",
)
x_p = pybamm.SpatialVariable(
    "x_p",
    domain=["positive electrode"],
    auxiliary_domains={"secondary": "current collector"},
    coord_sys="cartesian",
)
x = pybamm.SpatialVariable(
    "x",
    domain=whole_cell,
    auxiliary_domains={"secondary": "current collector"},
    coord_sys="cartesian",
)

y = pybamm.SpatialVariable("y", domain="current collector", coord_sys="cartesian")
z = pybamm.SpatialVariable("z", domain="current collector", coord_sys="cartesian")
r_macro = pybamm.SpatialVariable(
    "r_macro", domain="current collector", coord_sys="cylindrical polar"
)

r_n = pybamm.SpatialVariable(
    "r_n",
    domain=["negative particle"],
    auxiliary_domains={
        "secondary": "negative electrode",
        "tertiary": "current collector",
    },
    coord_sys="spherical polar",
)
r_p = pybamm.SpatialVariable(
    "r_p",
    domain=["positive particle"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
    coord_sys="spherical polar",
)
r_n_prim = pybamm.SpatialVariable(
    "r_n_prim",
    domain=["negative primary particle"],
    auxiliary_domains={
        "secondary": "negative electrode",
        "tertiary": "current collector",
    },
    coord_sys="spherical polar",
)
r_p_prim = pybamm.SpatialVariable(
    "r_p_prim",
    domain=["positive primary particle"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
    coord_sys="spherical polar",
)
r_n_sec = pybamm.SpatialVariable(
    "r_n_sec",
    domain=["negative secondary particle"],
    auxiliary_domains={
        "secondary": "negative electrode",
        "tertiary": "current collector",
    },
    coord_sys="spherical polar",
)
r_p_sec = pybamm.SpatialVariable(
    "r_p_sec",
    domain=["positive secondary particle"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
    coord_sys="spherical polar",
)

R_n = pybamm.SpatialVariable(
    "R_n",
    domain=["negative particle size"],
    auxiliary_domains={
        "secondary": "negative electrode",
        "tertiary": "current collector",
    },
    coord_sys="cartesian",
)

R_p = pybamm.SpatialVariable(
    "R_p",
    domain=["positive particle size"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
    coord_sys="cartesian",
)

R_n_prim = pybamm.SpatialVariable(
    "R_n_prim",
    domain=["negative primary particle size"],
    auxiliary_domains={
        "secondary": "negative electrode",
        "tertiary": "current collector",
    },
    coord_sys="cartesian",
)
R_p_prim = pybamm.SpatialVariable(
    "R_p_prim",
    domain=["positive primary particle size"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
    coord_sys="cartesian",
)

R_n_sec = pybamm.SpatialVariable(
    "R_n_sec",
    domain=["negative secondary particle size"],
    auxiliary_domains={
        "secondary": "negative electrode",
        "tertiary": "current collector",
    },
    coord_sys="cartesian",
)
R_p_sec = pybamm.SpatialVariable(
    "R_p_sec",
    domain=["positive secondary particle size"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
    coord_sys="cartesian",
)

# Domains at cell edges
x_n_edge = pybamm.SpatialVariableEdge(
    "x_n",
    domain=["negative electrode"],
    auxiliary_domains={"secondary": "current collector"},
    coord_sys="cartesian",
)
x_s_edge = pybamm.SpatialVariableEdge(
    "x_s",
    domain=["separator"],
    auxiliary_domains={"secondary": "current collector"},
    coord_sys="cartesian",
)
x_p_edge = pybamm.SpatialVariableEdge(
    "x_p",
    domain=["positive electrode"],
    auxiliary_domains={"secondary": "current collector"},
    coord_sys="cartesian",
)
x_edge = pybamm.SpatialVariableEdge(
    "x",
    domain=whole_cell,
    auxiliary_domains={"secondary": "current collector"},
    coord_sys="cartesian",
)

y_edge = pybamm.SpatialVariableEdge(
    "y", domain="current collector", coord_sys="cartesian"
)
z_edge = pybamm.SpatialVariableEdge(
    "z", domain="current collector", coord_sys="cartesian"
)
r_macro_edge = pybamm.SpatialVariableEdge(
    "r_macro", domain="current collector", coord_sys="cylindrical polar"
)


r_n_edge = pybamm.SpatialVariableEdge(
    "r_n",
    domain=["negative particle"],
    auxiliary_domains={
        "secondary": "negative electrode",
        "tertiary": "current collector",
    },
    coord_sys="spherical polar",
)
r_p_edge = pybamm.SpatialVariableEdge(
    "r_p",
    domain=["positive particle"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
    coord_sys="spherical polar",
)

R_n_edge = pybamm.SpatialVariableEdge(
    "R_n",
    domain=["negative particle size"],
    auxiliary_domains={
        "secondary": "negative electrode",
        "tertiary": "current collector",
    },
    coord_sys="cartesian",
)
R_p_edge = pybamm.SpatialVariableEdge(
    "R_p",
    domain=["positive particle size"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
    coord_sys="cartesian",
)

# new concentration domain (legacy single-domain, kept for backwards compatibility)
c_p = pybamm.SpatialVariable(
    "c_p",
    domain=["CCPM positive particle concentration"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
    coord_sys="cartesian",
)

c_p_edge = pybamm.SpatialVariableEdge(
    "c_p",
    domain=["CCPM positive particle concentration"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
    coord_sys="cartesian",
)

# Three-subdomain concentration variables (Clarke 2026 eq 62)
# Domain names contain 'particle' so PrimaryBroadcast from electrode domain works.
c_p_a = pybamm.SpatialVariable(
    "c_p_a",
    domain=["CCPM positive particle branch A"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
    coord_sys="cartesian",
)

c_p_a_edge = pybamm.SpatialVariableEdge(
    "c_p_a",
    domain=["CCPM positive particle branch A"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
    coord_sys="cartesian",
)

c_p_b = pybamm.SpatialVariable(
    "c_p_b",
    domain=["CCPM positive particle branch B"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
    coord_sys="cartesian",
)

c_p_b_edge = pybamm.SpatialVariableEdge(
    "c_p_b",
    domain=["CCPM positive particle branch B"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
    coord_sys="cartesian",
)

c_p_c = pybamm.SpatialVariable(
    "c_p_c",
    domain=["CCPM positive particle branch C"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
    coord_sys="cartesian",
)

c_p_c_edge = pybamm.SpatialVariableEdge(
    "c_p_c",
    domain=["CCPM positive particle branch C"],
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
    coord_sys="cartesian",
)
