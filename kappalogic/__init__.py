from .core import (
    sgn, k, reg, NOT, AND, OR, NAND, NOR, XOR, XNOR, AND_n, OR_n,
    eq, neq, gt, lt, ge, le, DEFAULT_XI, DEFAULT_KERNEL,
)
from .funcs import intf, par, maxf, minf, clamp
from .applications import (
    kronecker_delta, levi_civita, dirac_delta_integral,
    line_intersection_y, collatz_step, collatz_sequence,
    choc_bar_source,
)
from .kernels import KERNELS, apply_kernel
from .heat import xi_of_time, time_of_xi, heat_step_profile, heat_kernel_gaussian, gaussian_match
from .quantum_well import (
    box_heat_kernel, box_heat_kernel_eigen,
    infinite_well_propagator, infinite_well_propagator_eigen,
)
from .search import (
    soft_gt, soft_eq, soft_or, soft_and, anneal_solve, numeric_grad, l2_penalty,
    is_dont_care_variable, find_dont_care_variables,
)
from .field_theory import kink_profile, kink_eom_residual, kink_energy_exact, topological_charge
from .stat_mech import box_partition_function, box_partition_function_exact, witten_index_toy
from .info_theory import (
    gaussian_variance, differential_entropy_gaussian,
    fisher_information_gaussian, de_bruijn_check,
)
from .topology import morse_index, euler_characteristic
from .spacetime import inside_lightcone, future_lightcone
from .bridge import floor_smooth, sum_via_integral, gamma_via_riemann_sum
from .electronic_structure import (
    fermi_occupation, tight_binding_chain_energies,
    find_chemical_potential, total_energy_smeared,
    tight_binding_dos_continuum, continuum_filling, continuum_energy, continuum_find_mu,
)
from .theory import (
    max_gradient_location, max_gradient_value, is_single_passing_at_zero, fusion_is_safe,
    fusion_error_bound,
    or_gradient_closed_form, gradient_landscape_stats, or_fusion_disagreement_rate,
    and_partial_dilation_invariance, or_breaks_partial_dilation_invariance,
    or_second_resonance_w0, or_second_resonance_location, or_second_resonance_numeric_argmax,
    or_misclassification_boundary_sum, or_misclassification_boundary_numeric, or_value,
    or_n_misclassification_K, or_n_misclassification_boundary_sum,
    or_n_misclassification_boundary_numeric, or_n_value,
)
from .dynamics import force_fixed_point, multiplier_at, koenigs_coordinate, abel_function, fractional_iterate
from .identities import (
    rapidity, addition, n_tuple_angle, lambert_continued_fraction,
    mittag_leffler_sum, weierstrass_product, gudermannian, verify_identities,
)
from .gauge import (
    xi_dilation_equals_x_dilation, dilation_generator_matches_xi_generator,
    translation_dilation_algebra, hyperbolic_metric_is_scale_invariant,
    killing_vectors_of_xi_halfplane, geodesic_conserved_quantities,
    local_xi_connection_symbolic, local_xi_connection_numeric,
    verify_gauge_structure,
)

__all__ = [
    "sgn", "k", "reg", "NOT", "AND", "OR", "NAND", "NOR", "XOR", "XNOR", "AND_n", "OR_n",
    "eq", "neq", "gt", "lt", "ge", "le", "DEFAULT_XI", "DEFAULT_KERNEL",
    "intf", "par", "maxf", "minf", "clamp",
    "kronecker_delta", "levi_civita", "dirac_delta_integral",
    "line_intersection_y", "collatz_step", "collatz_sequence",
    "choc_bar_source",
    "KERNELS", "apply_kernel",
    "xi_of_time", "time_of_xi", "heat_step_profile", "heat_kernel_gaussian", "gaussian_match",
    "box_heat_kernel", "box_heat_kernel_eigen",
    "infinite_well_propagator", "infinite_well_propagator_eigen",
    "soft_gt", "soft_eq", "soft_or", "soft_and", "anneal_solve", "numeric_grad", "l2_penalty",
    "is_dont_care_variable", "find_dont_care_variables",
    "kink_profile", "kink_eom_residual", "kink_energy_exact", "topological_charge",
    "box_partition_function", "box_partition_function_exact", "witten_index_toy",
    "gaussian_variance", "differential_entropy_gaussian",
    "fisher_information_gaussian", "de_bruijn_check",
    "morse_index", "euler_characteristic",
    "inside_lightcone", "future_lightcone",
    "floor_smooth", "sum_via_integral", "gamma_via_riemann_sum",
    "fermi_occupation", "tight_binding_chain_energies",
    "find_chemical_potential", "total_energy_smeared",
    "tight_binding_dos_continuum", "continuum_filling", "continuum_energy", "continuum_find_mu",
    "max_gradient_location", "max_gradient_value", "is_single_passing_at_zero", "fusion_is_safe",
    "fusion_error_bound",
    "or_gradient_closed_form", "gradient_landscape_stats", "or_fusion_disagreement_rate",
    "and_partial_dilation_invariance", "or_breaks_partial_dilation_invariance",
    "or_second_resonance_w0", "or_second_resonance_location", "or_second_resonance_numeric_argmax",
    "or_misclassification_boundary_sum", "or_misclassification_boundary_numeric", "or_value",
    "or_n_misclassification_K", "or_n_misclassification_boundary_sum",
    "or_n_misclassification_boundary_numeric", "or_n_value",
    "force_fixed_point", "multiplier_at", "koenigs_coordinate", "abel_function", "fractional_iterate",
    "rapidity", "addition", "n_tuple_angle", "lambert_continued_fraction",
    "mittag_leffler_sum", "weierstrass_product", "gudermannian", "verify_identities",
    "xi_dilation_equals_x_dilation", "dilation_generator_matches_xi_generator",
    "translation_dilation_algebra", "hyperbolic_metric_is_scale_invariant",
    "killing_vectors_of_xi_halfplane", "geodesic_conserved_quantities",
    "local_xi_connection_symbolic", "local_xi_connection_numeric",
    "verify_gauge_structure",
]

__version__ = "0.20.0"
