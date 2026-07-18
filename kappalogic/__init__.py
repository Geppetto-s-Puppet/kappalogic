from .core import (
    sgn, k, reg, NOT, AND, OR, NAND, NOR, XOR, XNOR, AND_n, OR_n,
    eq, neq, gt, lt, ge, le, DEFAULT_XI, DEFAULT_KERNEL,
)
from .funcs import intf, par, maxf, minf, clamp, modf, delta_approx
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
    fisher_metric_gaussian, fisher_metric_matches_gauge_hyperbolic_metric,
    fisher_gaussian_curvature,
)
from .topology import morse_index, euler_characteristic
from .spacetime import inside_lightcone, future_lightcone
from .bridge import floor_smooth, sum_via_integral, product_via_integral, gamma_via_riemann_sum
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
    or_full_gradient_magnitude_argmax,
    or_misclassification_boundary_sum, or_misclassification_boundary_numeric, or_value,
    or_n_misclassification_K, or_n_misclassification_boundary_sum,
    or_n_misclassification_boundary_numeric, or_n_value,
    or_n_threshold_Cstar, or_n_fold_error_bound, or_n_fusion_is_safe, or_n_naive_fold,
    or_n_trigger_condition_is_not_necessary,
    xor_2d_boundary_v_given_large_u, xor_2d_boundary_numeric,
    xnor_2d_boundary_v_given_large_u, xnor_2d_boundary_numeric,
    nand_threshold_ab, nand_threshold_numeric, gate_dilation_type,
    nor_misclassification_boundary_sum, nor_misclassification_boundary_numeric,
    xor_zero_cross_section_threshold, xor_zero_cross_section_numeric,
    xor_diagonal_lower_threshold, xor_diagonal_upper_threshold, xor_diagonal_threshold_numeric,
    xnor_zero_cross_section_threshold, xnor_zero_cross_section_numeric,
    not_composition_tower, not_tower_threshold, not_tower_threshold_numeric,
)
from .dynamics import (
    force_fixed_point, multiplier_at, koenigs_coordinate, abel_function, fractional_iterate,
    not_map_fixed_point, not_map_multiplier, not_map_critical_xi,
    squared_map_cubic_coefficient, squared_map_convergence_exponent,
    asymptotic_fatou_coordinate, asymptotic_fatou_coordinate_local_check,
)
from .identities import (
    rapidity, addition, n_tuple_angle, lambert_continued_fraction,
    mittag_leffler_sum, weierstrass_product, gudermannian, verify_identities,
    integral_of_k, integral_of_reg, integral_of_NOT, integral_of_AND_wrt_a, integral_of_sech,
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
    "intf", "par", "maxf", "minf", "clamp", "modf", "delta_approx",
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
    "fisher_metric_gaussian", "fisher_metric_matches_gauge_hyperbolic_metric",
    "fisher_gaussian_curvature",
    "morse_index", "euler_characteristic",
    "inside_lightcone", "future_lightcone",
    "floor_smooth", "sum_via_integral", "product_via_integral", "gamma_via_riemann_sum",
    "fermi_occupation", "tight_binding_chain_energies",
    "find_chemical_potential", "total_energy_smeared",
    "tight_binding_dos_continuum", "continuum_filling", "continuum_energy", "continuum_find_mu",
    "max_gradient_location", "max_gradient_value", "is_single_passing_at_zero", "fusion_is_safe",
    "fusion_error_bound",
    "or_gradient_closed_form", "gradient_landscape_stats", "or_fusion_disagreement_rate",
    "and_partial_dilation_invariance", "or_breaks_partial_dilation_invariance",
    "or_second_resonance_w0", "or_second_resonance_location", "or_second_resonance_numeric_argmax",
    "or_full_gradient_magnitude_argmax",
    "or_misclassification_boundary_sum", "or_misclassification_boundary_numeric", "or_value",
    "or_n_misclassification_K", "or_n_misclassification_boundary_sum",
    "or_n_misclassification_boundary_numeric", "or_n_value",
    "or_n_threshold_Cstar", "or_n_fold_error_bound", "or_n_fusion_is_safe", "or_n_naive_fold",
    "or_n_trigger_condition_is_not_necessary",
    "xor_2d_boundary_v_given_large_u", "xor_2d_boundary_numeric",
    "xnor_2d_boundary_v_given_large_u", "xnor_2d_boundary_numeric",
    "nand_threshold_ab", "nand_threshold_numeric", "gate_dilation_type",
    "nor_misclassification_boundary_sum", "nor_misclassification_boundary_numeric",
    "xor_zero_cross_section_threshold", "xor_zero_cross_section_numeric",
    "xor_diagonal_lower_threshold", "xor_diagonal_upper_threshold", "xor_diagonal_threshold_numeric",
    "xnor_zero_cross_section_threshold", "xnor_zero_cross_section_numeric",
    "not_composition_tower", "not_tower_threshold", "not_tower_threshold_numeric",
    "force_fixed_point", "multiplier_at", "koenigs_coordinate", "abel_function", "fractional_iterate",
    "not_map_fixed_point", "not_map_multiplier", "not_map_critical_xi",
    "squared_map_cubic_coefficient", "squared_map_convergence_exponent",
    "asymptotic_fatou_coordinate", "asymptotic_fatou_coordinate_local_check",
    "rapidity", "addition", "n_tuple_angle", "lambert_continued_fraction",
    "mittag_leffler_sum", "weierstrass_product", "gudermannian", "verify_identities",
    "integral_of_k", "integral_of_reg", "integral_of_NOT", "integral_of_AND_wrt_a", "integral_of_sech",
    "xi_dilation_equals_x_dilation", "dilation_generator_matches_xi_generator",
    "translation_dilation_algebra", "hyperbolic_metric_is_scale_invariant",
    "killing_vectors_of_xi_halfplane", "geodesic_conserved_quantities",
    "local_xi_connection_symbolic", "local_xi_connection_numeric",
    "verify_gauge_structure",
]

__version__ = "0.40.0"
