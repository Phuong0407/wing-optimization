import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Union, Dict, Any

from mpi4py import MPI
import ufl

from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem

from wing_model import WingModel
from aerodynamic_load import import_foam_traction, map_traction, load_traction_xdmf
from shell_kinematics import shell_strains_from_model
from shell_bcs import bc_full_clamped, bc_torsional_spring, bc_prescribed_moment

from material_clt import clt_composite
from failure_analysis import recover_and_evaluate_failure_cells
from shell_stress import stress_resultants, drilling_terms
from material_constant import CE_PROPS, CE_STRENGTH
from postprocess import displacement_summary, composite_failure_summary, print_summary, export_results
from helper import vprint




@dataclass(frozen=True)
class TagsConfig:
    skin:   Tuple[int, ...] = (14, 15)
    ribs:   Tuple[int, ...] = (38, 39, 40, 41, 42)
    spars:  Tuple[int, ...] = (43, 44)

    @property
    def all_material_tags(self) -> Tuple[int, ...]:
        return self.skin + self.ribs + self.spars

@dataclass(frozen=True)
class ClampedBC:
    root_tag: int = 45

@dataclass(frozen=True)
class TorsionalSpringBC:
    root_tag:   int             = 45
    k_theta:    float           = 1E5
    components: Tuple[int, ...] = (0, 2)

@dataclass(frozen=True)
class PrescribedMomentBC:
    root_tag: int                           = 45
    moment:   Tuple[float, float, float]    = (0.0, 1000.0, 0.0)

BCType = Union[ ClampedBC, TorsionalSpringBC, PrescribedMomentBC ]

PETSC_OPTIONS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",

    "snes_type": "newtonls",
    "snes_rtol": 1e-8,
    "snes_atol": 1e-8,
    "snes_max_it": 25,
    "snes_monitor": None,

    "mat_mumps_icntl_14": 80,
    "mat_mumps_icntl_23": 2000,
}

class WingComputationModel:
    def __init__(self,
        mesh_file,
        foam_file,
        output_dir,
        tags: TagsConfig,
        traction_map_tags,
        bc: BCType,
        petsc_options=PETSC_OPTIONS,
        verbose=True
    ):
        self.comm = MPI.COMM_WORLD
        self.verbose = bool(verbose)

        self.tags       = tags
        self.TAG_SKIN   = list(tags.skin)
        self.TAG_RIBS   = list(tags.ribs)
        self.TAG_SPARS  = list(tags.spars)
        self.bc         = bc

        self.CE_PROPS       = CE_PROPS
        self.CE_STRENGTH    = CE_STRENGTH

        self.petsc_options = petsc_options or {}

        self.init_paths(mesh_file, foam_file, output_dir)
        self.init_wing_model()
        self.init_traction(traction_map_tags)
        self.init_bcs()



    def init_paths(self, mesh_file, foam_file, output_dir):
        self.mesh_file = Path(mesh_file)
        self.foam_file = Path(foam_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def init_wing_model(self):
        self.model = WingModel(str(self.mesh_file), self.comm)
        self.model.local_frame()
        self.model.function_space()

        self.dx = self.model.dx
        self.ds = self.model.ds

        (
            self.eps,
            self.kappa,
            self.gamma,
            self.drill,
            self.eps_,
            self.kappa_,
            self.gamma_,
            self.drill_,
        ) = shell_strains_from_model(self.model)


    def init_traction(self, traction_map_tags):
        map_file = self.output_dir / "MappedTraction.xdmf"
        if not map_file.exists():
            foam_xdmf = self.output_dir / "FOAMData.xdmf"
            import_foam_traction(self.foam_file, foam_xdmf)
            map_traction(foam_xdmf, self.mesh_file, map_file, traction_map_tags, "mm")
        self.FTraction = load_traction_xdmf(map_file, self.model.mesh)

    def init_bcs(self):
        if isinstance(self.bc, ClampedBC):
            self.BCS, self.bc_extras = bc_full_clamped(
                # self.model.mesh,
                self.model.V,
                self.model.facet_tags,
                self.model.fdim,
                self.bc.root_tag,
                comm=self.comm,
                verbose=self.verbose,
            )
        elif isinstance(self.bc, TorsionalSpringBC):
            self.BCS, self.bc_extras = bc_torsional_spring(
                self.model.mesh,
                self.model.V,
                self.model.facet_tags,
                self.model.fdim,
                self.bc.root_tag,
                comm=self.comm,
                verbose=self.verbose
            )
        elif isinstance(self.bc, PrescribedMomentBC):
            self.BCS, self.bc_extras = bc_prescribed_moment(
                self.model.mesh,
                self.model.V,
                self.model.facet_tags,
                self.model.fdim,
                self.bc.root_tag,
                self.bc.moment,
                self.ds,
                comm=self.comm,
                verbose=self.verbose,
            )
        else:
            raise ValueError("Unknown BC type")

    def create_material(self, layup, t_ply, label):
        mat = clt_composite(
            layup,
            t_ply,
            self.CE_PROPS["E1"],
            self.CE_PROPS["E2"],
            self.CE_PROPS["G12"],
            self.CE_PROPS["nu12"],
            G13=self.CE_PROPS["G12"],
            G23=self.CE_PROPS["G12"] * 0.5,
            kappa_s=5 / 6,
            verbose=self.verbose, # Ensure verbose output from clt_composite
            label=label,
        )
        mat.rho = self.CE_PROPS["rho"]
        mat._layup_angles  = list(layup)
        mat._t_ply = float(t_ply)
        mat._E1   = float(self.CE_PROPS["E1"])
        mat._E2   = float(self.CE_PROPS["E2"])
        mat._G12  = float(self.CE_PROPS["G12"])
        mat._nu12 = float(self.CE_PROPS["nu12"])
        # Debug check after all properties are set
        if self.verbose and self.comm.rank == 0:
            if mat.H <= 1e-12 or mat.rho <= 1e-12:
                print(f"[DEBUG] WARNING: Material {label} created with H={mat.H:.3e} or rho={mat.rho:.3e} effectively zero.")
        return mat

    def build_materials(self, design):
        t_skin = design.get("t_skin", 0.5E-3)
        t_spar = design.get("t_spar", 0.5E-3)
        t_rib  = design.get("t_rib",  0.5E-3)

        skin_layup = design.get("skin_layup", [45, -45, 0, 90, 90, 0, -45, 45])
        spar_layup = design.get("spar_layup", [0, 45, -45, 0, 0, -45, 45, 0])
        rib_layup  = design.get("rib_layup",  [45, -45, 90, 0, 0, 90, -45, 45])

        MAT_SKIN = self.create_material(skin_layup, t_skin, "SKIN")
        MAT_RIB  = self.create_material(rib_layup,  t_rib,  "RIB")
        MAT_SPAR = self.create_material(spar_layup, t_spar, "SPAR")

        self.MATS_COMP = {
            "SKIN": MAT_SKIN,
            "RIB":  MAT_RIB,
            "SPAR": MAT_SPAR,
        }

        self.MATS = {}
        for tag in self.TAG_SKIN:
            self.MATS[tag] = MAT_SKIN
        for tag in self.TAG_RIBS:
            self.MATS[tag] = MAT_RIB
        for tag in self.TAG_SPARS:
            self.MATS[tag] = MAT_SPAR

    # def build_materials(self, design):

    #     # ----------------------------------
    #     # thickness
    #     # ----------------------------------

    #     t_skin = design.get("t_skin", 0.5e-3)
    #     t_spar = design.get("t_spar", 0.5e-3)

    #     # ribs thickness list
    #     t_ribs_from_design = design.get("t_ribs")
    #     if t_ribs_from_design is None:
    #         # Fallback to single t_rib if the optimizer is not yet providing a list
    #         # Or if the user wants to run static analysis with a single t_rib
    #         t_rib_single = design.get("t_rib", 0.5e-3)
    #         t_ribs = [t_rib_single] * len(self.TAG_RIBS)
    #     else:
    #         t_ribs = t_ribs_from_design # Use the list directly
        
    #     assert len(t_ribs) == len(self.TAG_RIBS)


    #     # ----------------------------------
    #     # layups
    #     # ----------------------------------

    #     skin_layup = design.get(
    #         "skin_layup",
    #         [45,-45,0,90,90,0,-45,45]
    #     )

    #     spar_layup = design.get(
    #         "spar_layup",
    #         [0,45,-45,0,0,-45,45,0]
    #     )

    #     rib_layup = design.get(
    #         "rib_layup",
    #         [45,-45,90,0,0,90,-45,45]
    #     )


    #     # ----------------------------------
    #     # create materials
    #     # ----------------------------------

    #     MAT_SKIN = self.create_material(
    #         skin_layup,
    #         t_skin,
    #         "SKIN"
    #     )

    #     MAT_SPAR = self.create_material(
    #         spar_layup,
    #         t_spar,
    #         "SPAR"
    #     )


    #     # ----------------------------------
    #     # store materials
    #     # ----------------------------------

    #     self.MATS = {}

    #     self.MATS_COMP = {
    #         "SKIN": MAT_SKIN,
    #         "SPAR": MAT_SPAR,
    #         "RIBS": []
    #     }


    #     # ----------------------------------
    #     # skin tags
    #     # ----------------------------------

    #     for tag in self.TAG_SKIN:
    #         self.MATS[tag] = MAT_SKIN


    #     # ----------------------------------
    #     # spar tags
    #     # ----------------------------------

    #     for tag in self.TAG_SPARS:
    #         self.MATS[tag] = MAT_SPAR


    #     # ----------------------------------
    #     # ribs (each rib different)
    #     # ----------------------------------

    #     for tag, t_rib in zip(self.TAG_RIBS, t_ribs):

    #         MAT_RIB = self.create_material(
    #             rib_layup,
    #             t_rib,
    #             f"RIB_{tag}"
    #         )
    #         # Debug check after all properties are set
    #         if self.verbose and self.comm.rank == 0:
    #             if MAT_RIB.H <= 1e-12 or MAT_RIB.rho <= 1e-12:
    #                 print(f"[DEBUG] WARNING: RIB Material {tag} created with H={MAT_RIB.H:.3e} or rho={MAT_RIB.rho:.3e} effectively zero.")

    #         self.MATS[tag] = MAT_RIB

    #         self.MATS_COMP["RIBS"].append(MAT_RIB)

    def init_weakform(self):
        a_int            = self.build_internal_form()
        L_ext            = self.build_external_load()
        a_int, L_ext     = self.apply_bc_contributions(a_int, L_ext)
        residual         = a_int - L_ext
        tangent          = ufl.derivative(residual, self.model.v, self.model.dv)
        try:
            self.solve_system(residual, tangent)
            status = "success"
        except Exception as e:
            if self.comm.rank == 0 and self.verbose:
                print("[WingComputationModel] Solve failed:", repr(e))
            return dict(mass=np.inf, u_max=np.inf, status="failed")
        mass, u_max = self.postprocess()
        return dict(mass=mass, u_max=u_max, status=status)

    def build_internal_form(self):
        form_pieces = []
        for tag, mat in self.MATS.items():
            N, M, Q = stress_resultants(mat, self.eps, self.kappa, self.gamma)
            _, drill_t = drilling_terms(mat, self.model.mesh, self.drill)
            piece = (
                ufl.inner(N, self.eps_)
                + ufl.inner(M, self.kappa_)
                + ufl.inner(Q, self.gamma_)
                + drill_t * self.drill_
            ) * self.dx(tag)
            form_pieces.append(piece)
        a_int = sum(form_pieces)
        return a_int

    def build_external_load(self):
        L_ext = sum(ufl.dot(self.FTraction, self.model.u_) * self.dx(tag) for tag in self.TAG_SKIN)
        return L_ext

    def apply_bc_contributions(self, a_int, L_ext):
        if isinstance(self.bc_extras, dict):
            a_int += self.bc_extras.get("a_extra", ufl.as_ufl(0.0))
            L_ext += self.bc_extras.get("L_extra", ufl.as_ufl(0.0))
        return a_int, L_ext
    
    def solve(self, design):
        self.build_materials(design)
        a_int = self.build_internal_form()
        L_ext = self.build_external_load()
        if isinstance(self.bc_extras, dict):
            a_int += self.bc_extras.get("spring_form", 0)
            L_ext += self.bc_extras.get("moment_form", 0)

        residual = a_int - L_ext
        tangent  = ufl.derivative(residual, self.model.v, self.model.dv)

        # Debug: Check the norm of the assembled external load vector
        if self.verbose and self.comm.rank == 0:
            # Assemble L_ext_form to check its norm before BCs are applied
            # Note: This is a raw assembly of the form, not the final RHS vector
            # after lifting. It confirms if the load form itself is non-zero.
            assembled_L_ext_form = fem.petsc.assemble_vector(fem.form(L_ext))
            print(f"[DEBUG] Norm of assembled external load form (before BCs): {assembled_L_ext_form.norm():.3e}")
            if assembled_L_ext_form.norm() < 1e-12:
                print("[DEBUG] WARNING: Assembled external load form is effectively zero. Check aerodynamic_load.py output.")
            assembled_L_ext_form.destroy() # Clean up PETSc vector

        problem = NonlinearProblem(
            residual,
            self.model.v,
            bcs=self.BCS,
            J=tangent,
            petsc_options_prefix="wing",
            petsc_options=self.petsc_options
        )
        problem.solve()
        return { "solution": self.model.v}


    def compute_wing_mass(self):
        mass_form = ufl.as_ufl(0.0)
        for tag, mat in self.MATS.items():
            mass_form += (mat.rho * mat.H) * self.dx(tag)
        mass_local = fem.assemble_scalar(fem.form(mass_form))
        mass = float(self.comm.allreduce(mass_local, op=MPI.SUM))
        return mass

    # def assemble_stiffness(self):
    #     K_form = ufl.as_ufl(0.0)
    #     for tag, mat in self.MATS.items():
    #         N, M, Q = stress_resultants(
    #             mat,
    #             self.eps,
    #             self.kappa,
    #             self.gamma
    #         )
    #         _, drill_t = drilling_terms(mat, self.model.mesh, self.drill)

    #         K_form += (
    #             ufl.inner(N, self.eps_)
    #             + ufl.inner(M, self.kappa_)
    #             + ufl.inner(Q, self.gamma_)
    #             + drill_t * self.drill_
    #         ) * self.dx(tag)

    #     K = fem.petsc.assemble_matrix(fem.form(K_form), bcs=self.BCS)
    #     K.assemble()
    #     return K

    def assemble_stiffness(self):

        a_int = ufl.as_ufl(0.0)

        for tag, mat in self.MATS.items():

            N, M, Q = stress_resultants(
                mat,
                self.eps,
                self.kappa,
                self.gamma
            )

            _, drill_t = drilling_terms(
                mat,
                self.model.mesh,
                self.drill
            )

            a_int += (
                ufl.inner(N, self.eps_)
                + ufl.inner(M, self.kappa_)
                + ufl.inner(Q, self.gamma_)
                + drill_t * self.drill_
            ) * self.dx(tag)

        # Linearize
        K_form = ufl.derivative(
            a_int,
            self.model.v,
            self.model.dv
        )

        K = fem.petsc.assemble_matrix(
            fem.form(K_form),
            bcs=self.BCS
        )

        K.assemble()

        return K

    def assemble_mass(self):

        dv = self.model.dv      # TrialFunction
        v_ = self.model.v_      # TestFunction

        du, dtheta = ufl.split(dv)
        u_, theta_ = ufl.split(v_)

        M_form = 0

        for tag, mat in self.MATS.items():

            rho = float(mat.rho)
            h   = float(mat.H)

            M_form += (
                rho * h * ufl.inner(du, u_)
                + rho * h**3 / 12.0 * ufl.inner(dtheta, theta_)
            ) * self.dx(tag)

        M = fem.petsc.assemble_matrix(
            fem.form(M_form),
            bcs=self.BCS
        )
        M.assemble()

        return M

    def assemble_external_load(self):
        F_form = ufl.as_ufl(0.0)
        for tag in self.TAG_SKIN:
            F_form += ufl.dot(
                self.FTraction,
                self.model.u_
            ) * self.dx(tag)
        F = fem.petsc.assemble_vector(fem.form(F_form))

        fem.petsc.apply_lifting(
            F,
            [fem.form(F_form)],
            bcs=[self.BCS]
        )

        F.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE
        )
        fem.petsc.set_bc(F, self.BCS)
        return F



    def assemble_aero_stiffness(self):

        dv = self.model.dv      # TrialFunction
        v_ = self.model.v_      # TestFunction

        du, _ = ufl.split(dv)
        u_, _ = ufl.split(v_)

        Ka_form = 0

        for tag in self.TAG_SKIN:
            Ka_form += ufl.inner(du, u_) * self.dx(tag)

        Ka = fem.petsc.assemble_matrix(
            fem.form(Ka_form),
            bcs=self.BCS
        )
        Ka.assemble()

        return Ka



    def postprocess(self, export=True):

        v = self.model.v

        # displacement
        u_max = displacement_summary(v, self.comm)

        # mass
        wing_mass = self.compute_wing_mass()


        # -------------------------------
        # build regions dynamically
        # -------------------------------

        regions = []

        # skin
        for tag in self.TAG_SKIN:
            regions.append((tag, self.MATS[tag], "SKIN"))

        # spars
        for tag in self.TAG_SPARS:
            regions.append((tag, self.MATS[tag], "SPAR"))

        # ribs (each rib different material)
        for tag in self.TAG_RIBS:
            regions.append((tag, self.MATS[tag], "RIBS"))


        # -------------------------------
        # failure evaluation
        # -------------------------------

        fi_results = composite_failure_summary(
            self.model.mesh,
            v,
            self.model.cell_tags,
            regions,
            self.CE_STRENGTH,
            recover_and_evaluate_failure_cells
        )


        # -------------------------------
        # printing
        # -------------------------------

        if self.comm.rank == 0:
            print_summary(u_max, fi_results)
            vprint(self.verbose, f"wing mass = {wing_mass:.6e} kg")


        # -------------------------------
        # export
        # -------------------------------

        if export:

            export_results(
                self.model.mesh,
                v,
                self.model.cell_tags,
                self.tags.all_material_tags,
                self.model.mesh.geometry.dim,
                write_output=True,
                comm=self.comm
            )


        return {
            "mass": wing_mass,
            "u_max": u_max,
            "FI": fi_results
        }