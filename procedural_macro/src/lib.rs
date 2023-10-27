#![doc = include_str!("../README.md")]
#![doc(html_root_url = "https://docs.rs/lattice_qcd_rs-procedural_macro/0.3.0")]
//
#![warn(clippy::pedantic)]
#![warn(clippy::cargo)]
//
#![warn(clippy::absolute_paths)]
#![warn(clippy::allow_attributes)] // attributes
#![warn(clippy::allow_attributes_without_reason)] //attributes
#![warn(clippy::as_underscore)]
#![warn(clippy::assertions_on_result_states)]
#![warn(clippy::branches_sharing_code)]
#![warn(clippy::clear_with_drain)]
#![warn(clippy::clone_on_ref_ptr)]
#![warn(clippy::cognitive_complexity)]
#![warn(clippy::collection_is_never_read)]
#![warn(clippy::create_dir)]
#![warn(clippy::debug_assert_with_mut_call)]
#![warn(clippy::decimal_literal_representation)]
#![warn(clippy::default_numeric_fallback)]
#![warn(clippy::default_union_representation)]
#![warn(clippy::disallowed_script_idents)] // cspell: ignore idents
#![warn(clippy::empty_drop)]
#![warn(clippy::empty_line_after_doc_comments)]
#![warn(clippy::empty_line_after_outer_attr)]
#![warn(clippy::empty_structs_with_brackets)]
#![warn(clippy::equatable_if_let)]
#![warn(clippy::error_impl_error)]
#![warn(clippy::exhaustive_enums)]
#![warn(clippy::fallible_impl_from)]
#![warn(clippy::filetype_is_file)]
#![warn(clippy::float_cmp_const)]
#![warn(clippy::fn_to_numeric_cast_any)]
#![warn(clippy::format_push_string)]
#![warn(clippy::if_then_some_else_none)]
#![warn(clippy::impl_trait_in_params)]
#![warn(clippy::implicit_saturating_sub)]
#![warn(clippy::imprecise_flops)]
#![warn(clippy::iter_on_empty_collections)]
#![warn(clippy::iter_on_single_items)]
#![warn(clippy::iter_with_drain)]
#![warn(clippy::large_include_file)]
#![warn(clippy::large_stack_frames)]
#![warn(clippy::let_underscore_must_use)]
#![warn(clippy::let_underscore_untyped)]
#![warn(clippy::lossy_float_literal)]
#![warn(clippy::manual_clamp)]
#![deny(clippy::exit)] // deny
#![warn(clippy::future_not_send)]
//#![warn(clippy::mem_forget)] // memory, mistake allow
#![warn(clippy::map_err_ignore)]
#![warn(clippy::map_unwrap_or)]
#![warn(clippy::match_wildcard_for_single_variants)]
#![warn(clippy::missing_assert_message)]
#![warn(clippy::missing_const_for_fn)]
#![warn(clippy::missing_enforced_import_renames)]
#![warn(clippy::missing_inline_in_public_items)]
#![warn(clippy::mixed_read_write_in_expression)]
// #![warn(clippy::module_name_repetitions)] // allow
// #![warn(clippy::multiple_unsafe_ops_per_block)]
#![warn(clippy::mutex_atomic)]
#![warn(clippy::mutex_integer)]
#![warn(clippy::needless_collect)]
#![warn(clippy::needless_raw_strings)]
#![warn(clippy::nonstandard_macro_braces)]
#![warn(clippy::non_ascii_literal)]
#![warn(clippy::option_if_let_else)]
#![warn(clippy::or_fun_call)]
#![warn(clippy::path_buf_push_overwrite)]
// #![warn(clippy::pattern_type_mismatch)] // maybe
// #![warn(clippy::ptr_as_ptr)] // allowed ?
#![warn(clippy::pub_without_shorthand)] // style choice
#![warn(clippy::rc_buffer)]
#![warn(clippy::rc_mutex)]
#![warn(clippy::readonly_write_lock)]
#![warn(clippy::redundant_clone)]
#![warn(clippy::redundant_pub_crate)]
#![warn(clippy::redundant_type_annotations)]
#![warn(clippy::rest_pat_in_fully_bound_structs)]
#![warn(clippy::same_name_method)]
#![deny(clippy::self_named_module_files)] // style, file
//#![deny(clippy::mod_module_files)]
#![warn(clippy::semicolon_outside_block)] // style
// cspell: ignore scrutinee
#![warn(clippy::significant_drop_in_scrutinee)] // maybe ?
#![warn(clippy::significant_drop_tightening)] // maybe ?
#![warn(clippy::str_to_string)] // style
#![warn(clippy::string_add)] // restriction, style
#![warn(clippy::string_lit_chars_any)] // perf
#![warn(clippy::string_to_string)] // mistake
#![warn(clippy::suboptimal_flops)] // precision
#![warn(clippy::suspicious_operation_groupings)] // mistake
#![warn(clippy::suspicious_xor_used_as_pow)] // mistake
#![warn(clippy::tests_outside_test_module)] // mistake, perf, readability
#![warn(clippy::todo)] // reminder
#![warn(clippy::trailing_empty_array)] // mistake
#![warn(clippy::trait_duplication_in_bounds)] // mistake, readability
// cspell: ignore repr
#![warn(clippy::transmute_undefined_repr)] // safety
#![warn(clippy::trivial_regex)] // perf, mistake
#![warn(clippy::try_err)] // restriction. style
#![warn(clippy::tuple_array_conversions)] // style
#![warn(clippy::type_repetition_in_bounds)] // style, mistake
#![warn(clippy::undocumented_unsafe_blocks)] // Doc
#![warn(clippy::unimplemented)] // reminder
#![warn(clippy::unnecessary_self_imports)] // style
#![warn(clippy::unnecessary_struct_initialization)] // style , readability
// cspell: ignore unseparated
#![warn(clippy::unseparated_literal_suffix)] // style
// cspell: ignore peekable
#![warn(clippy::unused_peekable)] // mistake
#![warn(clippy::unused_rounding)] // mistake, readability
#![warn(clippy::unwrap_in_result)] // mistake, error propagation
#![warn(clippy::unwrap_used)] // allow ? style
#![warn(clippy::use_debug)] // debug removing
#![warn(clippy::use_self)] // style
#![warn(clippy::useless_let_if_seq)] // style
#![warn(clippy::verbose_file_reads)]
#![deny(unsafe_code)]
//
//---------------
// doc
#![warn(missing_docs)] // doc
#![warn(clippy::missing_docs_in_private_items)] // doc
#![deny(rustdoc::broken_intra_doc_links)] // cspell: ignore rustdoc
#![deny(rustdoc::private_intra_doc_links)]
#![deny(rustdoc::invalid_codeblock_attributes)]
#![deny(rustdoc::invalid_html_tags)]
#![deny(rustdoc::invalid_rust_codeblocks)]
#![deny(rustdoc::bare_urls)]
#![deny(rustdoc::unescaped_backticks)]
#![deny(rustdoc::redundant_explicit_links)]
//---------------
// allow
#![allow(clippy::module_name_repetitions)]

/// Only contains the version test.
#[cfg(test)]
mod test;

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::Ident;

/// Maximum dimension to impl [`Direction`](https://abouttefeux.github.io/lattice-qcd-rs/lattice_qcd_rs/lattice/struct.Direction.html) for.
const MAX_DIM: usize = 127;

/// Implement [`DirectionList`](https://abouttefeux.github.io/lattice-qcd-rs/lattice_qcd_rs/lattice/trait.DirectionList.html)
/// for [`Direction`](https://abouttefeux.github.io/lattice-qcd-rs/lattice_qcd_rs/lattice/struct.Direction.html) of `1` to `127` the value of `MAX_DIM`.
///
/// Using const generics might render this unnecessary. Waiting for stabilization of `feature(generic_const_exprs)`.
#[proc_macro]
pub fn implement_direction_list(_item: TokenStream) -> TokenStream {
    let mut implem = Vec::with_capacity(MAX_DIM); // cspell: ignore implem implems

    // creation of the vec containing the directions.
    for i in 1_usize..=MAX_DIM {
        let mut array_direction = Vec::with_capacity(MAX_DIM);
        let mut array_direction_positives = Vec::with_capacity(MAX_DIM);
        for j in 0..i {
            array_direction.push(quote! {
                Direction { index_dir: #j, is_positive: true },
                Direction { index_dir: #j, is_positive: false }
            });
            array_direction_positives.push(quote! {
                Direction { index_dir: #j, is_positive: true }
            });
        }

        let u_dir_ident = Ident::new(&format!("U{i}_DIR"), Span::call_site());
        let u_dir_pos_ident = Ident::new(&format!("U{i}_DIR_POS"), Span::call_site());
        // we store the values in array so we can access them as fast as possible.
        let comment_all = format!(
            "Array of all [`Direction`] in dimension {i}. Added automatically by a procedural macro."
        );
        let comment_pos = format!("Array of positive [`Direction`] in dimension {i}. Added automatically by a procedural macro.");

        let s = quote! {
            #[doc=#comment_all]
            const #u_dir_ident: [Direction<#i>; #i * 2] = [ #(#array_direction),* ];
            #[doc=#comment_pos]
            const #u_dir_pos_ident: [Direction<#i>; #i] = [ #(#array_direction_positives),* ];

            impl DirectionList for Direction<#i> {
                #[inline]
                fn directions() -> &'static [Self] {
                    &#u_dir_ident
                }

                #[inline]
                fn positive_directions() -> &'static [Self] {
                    &#u_dir_pos_ident
                }
            }
        };

        implem.push(s);
    }
    // We need to concat the final implems togethers.
    let final_stream = quote! {
        #(#implem)*
    };
    final_stream.into()
}

/// The max dimension to imply the [`From`] and [`std::convert::TryFrom`] and for
/// [`Direction`](https://abouttefeux.github.io/lattice-qcd-rs/lattice_qcd_rs/lattice/struct.Direction.html).
///
/// It takes just way to long for 127.
const MAX_DIM_FROM_IMPLEM: usize = 10;

/// Implement trait [`From`] and [`std::convert::TryFrom`] and for directions
#[proc_macro]
pub fn implement_direction_from(_item: TokenStream) -> TokenStream {
    // implementation of the error returned by the TryFrom trait.
    let mut implem = Vec::with_capacity(MAX_DIM_FROM_IMPLEM * (MAX_DIM_FROM_IMPLEM - 1) / 2);

    for i in 1_usize..MAX_DIM_FROM_IMPLEM {
        for j in i + 1..=MAX_DIM_FROM_IMPLEM {
            //let u_ident_from = syn::Ident::new(&format!("U{}", i), proc_macro2::Span::call_site());
            //let u_ident_to = syn::Ident::new(&format!("U{}", j), proc_macro2::Span::call_site());
            implem.push(quote! {
                impl From<Direction<#i>> for Direction<#j> {
                    #[inline]
                    fn from(from: Direction<#i>) -> Self {
                        // i > j so so it is always Some.
                        Self::new(from.index(), from.is_positive()).expect("index is always within bounds")
                    }
                }

                impl From<&Direction<#i>> for Direction<#j> {
                    #[inline]
                    fn from(from: &Direction<#i>) -> Self {
                        (*from).into()
                    }
                }

                #[allow(clippy::absolute_paths)]
                impl std::convert::TryFrom<Direction<#j>> for Direction<#i> {
                    type Error = crate::lattice::DirectionConversionError;

                    #[inline]
                    fn try_from(from: Direction<#j>) -> Result<Self, Self::Error> {
                        Self::new(from.index(), from.is_positive())
                            .ok_or(Self::Error::IndexOutOfBound)
                    }
                }

                #[allow(clippy::absolute_paths)]
                impl std::convert::TryFrom<&Direction<#j>> for Direction<#i> {
                    type Error = <Self as TryFrom<Direction<#j>>>::Error;

                    #[inline]
                    fn try_from(from: &Direction<#j>) -> Result<Self, Self::Error> {
                        (*from).try_into()
                    }
                }
            });
        }
    }

    // We need to concat the final implems togethers.
    let final_stream = quote! {
        #(#implem)*
    };
    final_stream.into()
}
