from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SolveResult:
    displacement: np.ndarray
    traction_magnitude: float
    linear_scale: float
    attempts: int = 1


@dataclass
class PreparedLinearSystem:
    """Matrix factorization tied to one sampled attachment state."""

    matrix: object
    boundary_conditions: list
    solver: object


class LinearElasticSolver:
    """Serial quasi-static P1 tetrahedral FEM with explicit index mappings.

    The returned displacement is ordered by *topological vertex*, exactly like
    ``volume_points`` and ``volume_cells``. This avoids relying on accidental
    equality between DOLFINx topology, geometry-node and function-DOF ordering.
    """

    def __init__(
        self,
        volume_xdmf: str,
        mesh_name: str,
        young: float,
        poisson: float,
        petsc_options,
        *,
        require_serial: bool = True,
    ):
        try:
            from mpi4py import MPI
            from petsc4py import PETSc
            import ufl
            from dolfinx import fem, io, mesh
            from dolfinx.fem import petsc as fem_petsc
        except Exception as exc:
            raise RuntimeError("Requires DOLFINx, UFL, petsc4py and mpi4py") from exc

        self.MPI = MPI
        self.PETSc = PETSc
        self.ufl = ufl
        self.fem = fem
        self.io = io
        self.mesh = mesh
        self.fp = fem_petsc

        with io.XDMFFile(MPI.COMM_WORLD, str(Path(volume_xdmf)), "r") as file:
            self.domain = file.read_mesh(name=mesh_name)
        if require_serial and self.domain.comm.size != 1:
            raise RuntimeError(
                "This multiprocessing pipeline requires one MPI rank per process; "
                "launch it with ordinary Python, not mpirun"
            )

        tdim = self.domain.topology.dim
        self.domain.topology.create_connectivity(tdim, 0)
        self.domain.topology.create_connectivity(0, tdim)
        self.domain.topology.create_connectivity(tdim - 1, tdim)

        vertex_map = self.domain.topology.index_map(0)
        self.num_vertices = int(vertex_map.size_local + vertex_map.num_ghosts)
        self.topology_vertices = np.arange(self.num_vertices, dtype=np.int32)
        topology_to_geometry = mesh.entities_to_geometry(
            self.domain, 0, self.topology_vertices
        ).reshape(-1)
        if len(topology_to_geometry) != self.num_vertices:
            raise RuntimeError("Could not map all topology vertices to geometry nodes")
        self.topology_to_geometry = topology_to_geometry.astype(np.int32)
        self.volume_points = np.asarray(
            self.domain.geometry.x[self.topology_to_geometry], dtype=float
        )

        connectivity = self.domain.topology.connectivity(tdim, 0)
        if connectivity is None:
            raise RuntimeError("Could not build tetrahedron-to-vertex connectivity")
        if connectivity.num_nodes < 1 or len(connectivity.links(0)) != 4:
            raise ValueError("The current backend requires tetrahedral volume meshes")
        self.volume_cells = np.asarray(connectivity.array, dtype=np.int64).reshape(
            -1, 4
        )

        self.V = fem.functionspace(self.domain, ("Lagrange", 1, (3,)))
        self.Q = fem.functionspace(self.domain, ("Lagrange", 1))
        self.scalar_vertex_to_dof = self._build_vertex_to_dof(self.Q)
        self.vector_vertex_to_dof = self._build_vertex_to_dof(self.V)
        if self.V.dofmap.bs != 3:
            raise RuntimeError(
                "Expected a blocked three-component vector function space"
            )

        mu = young / (2.0 * (1.0 + poisson))
        lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        trial = ufl.TrialFunction(self.V)
        test = ufl.TestFunction(self.V)

        def epsilon(value):
            return ufl.sym(ufl.grad(value))

        def sigma(value):
            return 2.0 * mu * epsilon(value) + lam * ufl.tr(
                epsilon(value)
            ) * ufl.Identity(3)

        self.a = fem.form(ufl.inner(sigma(trial), epsilon(test)) * ufl.dx)
        self.test = test
        self.petsc_options = dict(petsc_options)
        self.ds = ufl.Measure("ds", domain=self.domain)
        # The UFL structure is invariant across samples. Keep one compiled RHS
        # form and update only its coefficient arrays for each proposed load.
        self._weight_function = fem.Function(self.Q)
        self._traction = fem.Constant(
            self.domain, np.zeros(3, dtype=self.PETSc.ScalarType)
        )
        self._linear_form = fem.form(
            ufl.dot(self._weight_function * self._traction, self.test) * self.ds
        )

    def _build_vertex_to_dof(self, function_space):
        """Build an explicit topology-vertex to block-DOF lookup once."""

        output = np.empty(self.num_vertices, dtype=np.int32)
        for vertex in range(self.num_vertices):
            dofs = self.fem.locate_dofs_topological(
                function_space,
                0,
                np.asarray([vertex], dtype=np.int32),
                remote=False,
            )
            if len(dofs) != 1:
                raise RuntimeError(
                    f"Expected one P1 dof for vertex {vertex}, found {len(dofs)}"
                )
            output[vertex] = int(dofs[0])
        if len(np.unique(output)) != self.num_vertices:
            raise RuntimeError("Vertex-to-DOF mapping is not one-to-one")
        return output

    def mapping_diagnostics(self):
        return {
            "n_topology_vertices": self.num_vertices,
            "n_geometry_nodes": int(len(self.domain.geometry.x)),
            "topology_geometry_identity": bool(
                np.array_equal(
                    self.topology_to_geometry,
                    np.arange(self.num_vertices, dtype=np.int32),
                )
            ),
            "scalar_dof_identity": bool(
                np.array_equal(
                    self.scalar_vertex_to_dof,
                    np.arange(self.num_vertices, dtype=np.int32),
                )
            ),
            "vector_dof_identity": bool(
                np.array_equal(
                    self.vector_vertex_to_dof,
                    np.arange(self.num_vertices, dtype=np.int32),
                )
            ),
        }

    def _assemble_stiffness(
        self, hard_nodes, soft_nodes, soft_weights, relative_stiffness
    ):
        hard_nodes = np.unique(np.asarray(hard_nodes, dtype=np.int32))
        hard_dofs = self.vector_vertex_to_dof[hard_nodes]
        boundary_condition = self.fem.dirichletbc(
            np.zeros(3, dtype=self.PETSc.ScalarType), hard_dofs, self.V
        )

        matrix = self.fp.assemble_matrix(self.a, bcs=[boundary_condition])
        matrix.assemble()
        diagonal = matrix.getDiagonal()
        try:
            values = np.abs(diagonal.getArray(readonly=True))
            positive = values[values > 0]
            base_stiffness = float(np.median(positive)) if len(positive) else 1.0
        finally:
            diagonal.destroy()

        soft_nodes = np.asarray(soft_nodes, dtype=np.int32)
        soft_weights = np.asarray(soft_weights, dtype=float)
        if len(soft_nodes):
            block_dofs = self.vector_vertex_to_dof[soft_nodes]
            components = np.arange(self.V.dofmap.bs, dtype=np.int32)
            scalar_dofs = (
                block_dofs[:, None] * self.V.dofmap.bs + components[None, :]
            ).reshape(-1)
            spring_values = np.repeat(
                base_stiffness * float(relative_stiffness) * soft_weights,
                self.V.dofmap.bs,
            )
            spring_diagonal = matrix.createVecLeft()
            try:
                spring_diagonal.set(0.0)
                spring_diagonal.setValues(
                    scalar_dofs,
                    spring_values,
                    addv=self.PETSc.InsertMode.ADD_VALUES,
                )
                spring_diagonal.assemblyBegin()
                spring_diagonal.assemblyEnd()
                matrix.setDiagonal(
                    spring_diagonal, addv=self.PETSc.InsertMode.ADD_VALUES
                )
            finally:
                spring_diagonal.destroy()
        matrix.assemble()
        return matrix, [boundary_condition]

    def _load_function(self, load_nodes, load_weights):
        function = self._weight_function
        function.x.array[:] = 0.0
        nodes = np.asarray(load_nodes, dtype=np.int32)
        weights = np.asarray(load_weights, dtype=float)
        if len(nodes) != len(weights):
            raise ValueError("load_nodes and load_weights must have the same length")
        if len(nodes) and (nodes.min() < 0 or nodes.max() >= self.num_vertices):
            raise IndexError("Load node outside topology vertex range")
        dofs = self.scalar_vertex_to_dof[nodes]
        function.x.array[dofs] = weights
        function.x.scatter_forward()
        return function

    def _make_solver(self, matrix):
        solver = self.PETSc.KSP().create(self.domain.comm)
        solver.setOperators(matrix)
        prefix = f"meshgen_{id(self)}_{id(matrix)}_"
        solver.setOptionsPrefix(prefix)
        options = self.PETSc.Options()
        for key, value in self.petsc_options.items():
            options[f"{prefix}{key}"] = value
        solver.setFromOptions()
        return solver

    def prepare_system(
        self, hard_nodes, soft_nodes, soft_weights, soft_relative_stiffness
    ):
        """Assemble and factorize once for a fixed anchor configuration."""

        matrix, boundary_conditions = self._assemble_stiffness(
            hard_nodes, soft_nodes, soft_weights, soft_relative_stiffness
        )
        return PreparedLinearSystem(
            matrix=matrix,
            boundary_conditions=boundary_conditions,
            solver=self._make_solver(matrix),
        )

    @staticmethod
    def release_system(prepared):
        """Release PETSc resources as soon as an anchor realization ends."""

        for value in (prepared.solver, prepared.matrix):
            try:
                value.destroy()
            except Exception:
                pass

    def _solution_by_vertex(self, solution):
        raw = np.asarray(solution.array, dtype=float).reshape(-1, self.V.dofmap.bs)
        if self.vector_vertex_to_dof.max(initial=-1) >= len(raw):
            raise RuntimeError("Vector DOF mapping exceeds the solution array")
        return raw[self.vector_vertex_to_dof].copy()

    def solve_prepared(
        self,
        *,
        prepared,
        load_nodes,
        load_weights,
        direction,
        hard_nodes,
        soft_nodes,
        soft_weights,
        soft_relative_stiffness,
        target_max,
        initial_traction,
        target_nodes=None,
    ):
        self._load_function(load_nodes, load_weights)
        self._traction.value[:] = np.asarray(direction, dtype=float) * float(
            initial_traction
        )
        vector = self.fp.assemble_vector(self._linear_form)
        solution = prepared.matrix.createVecRight()
        try:
            self.fp.apply_lifting(vector, [self.a], bcs=[prepared.boundary_conditions])
            vector.ghostUpdate(
                addv=self.PETSc.InsertMode.ADD,
                mode=self.PETSc.ScatterMode.REVERSE,
            )
            for boundary_condition in prepared.boundary_conditions:
                boundary_condition.set(vector.array_w)

            prepared.solver.solve(vector, solution)
            if int(prepared.solver.getConvergedReason()) <= 0:
                raise RuntimeError("PETSc linear solve failed")

            unit_displacement = self._solution_by_vertex(solution)
        finally:
            # One retry allocates a PETSc RHS and solution.  Releasing those
            # objects deterministically prevents a long invalid-sample retry
            # sequence from retaining native-memory buffers.
            solution.destroy()
            vector.destroy()
        target_view = (
            unit_displacement
            if target_nodes is None
            else unit_displacement[np.asarray(target_nodes, dtype=np.int64)]
        )
        current_max = float(np.linalg.norm(target_view, axis=1).max())
        if current_max <= 1e-12:
            raise RuntimeError("Near-zero FEM response")
        scale = float(target_max) / current_max
        displacement = unit_displacement * scale

        hard = np.asarray(hard_nodes, dtype=np.int64)
        if len(hard):
            hard_max = float(np.linalg.norm(displacement[hard], axis=1).max())
            if hard_max > 1.0e-8:
                raise RuntimeError(f"Hard Dirichlet nodes moved by {hard_max:.3e} mm")
        return SolveResult(
            displacement=displacement,
            traction_magnitude=float(initial_traction) * scale,
            linear_scale=scale,
        )

    def solve(
        self,
        *,
        load_nodes,
        load_weights,
        direction,
        hard_nodes,
        soft_nodes,
        soft_weights,
        soft_relative_stiffness,
        target_max,
        initial_traction,
        target_nodes=None,
    ):
        """Compatibility one-shot solve; retry loops should use prepare_system."""

        prepared = self.prepare_system(
            hard_nodes, soft_nodes, soft_weights, soft_relative_stiffness
        )
        try:
            return self.solve_prepared(
                prepared=prepared,
                load_nodes=load_nodes,
                load_weights=load_weights,
                direction=direction,
                hard_nodes=hard_nodes,
                soft_nodes=soft_nodes,
                soft_weights=soft_weights,
                soft_relative_stiffness=soft_relative_stiffness,
                target_max=target_max,
                initial_traction=initial_traction,
                target_nodes=target_nodes,
            )
        finally:
            self.release_system(prepared)


LinearElasticSinglePeakSolver = LinearElasticSolver
