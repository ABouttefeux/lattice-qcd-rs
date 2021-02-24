
extern crate proc_macro;
extern crate quote;
extern crate syn;
extern crate proc_macro2;

use quote::quote;
use proc_macro::TokenStream;

#[proc_macro]
pub fn implement_direction_list(_item: TokenStream) -> TokenStream {
    let mut implem = vec![];
    for i in 1_usize..=127_usize {
        let mut array_direction = vec![];
        let mut array_direction_positives = vec![];
        for j in 0..i {
            array_direction.push(
                quote!{
                    Direction{index_dir: #j, is_positive: true, _phantom: PhantomData},
                    Direction{index_dir: #j, is_positive: false, _phantom: PhantomData}
                }
            );
            array_direction_positives.push(
                quote!{
                    Direction{index_dir: #j, is_positive: true, _phantom: PhantomData}
                }
            );
        }
        let u_indent = syn::Ident::new(&format!("U{}", i), proc_macro2::Span::call_site());
        let u_dir_indent = syn::Ident::new(&format!("U{}_DIR", i), proc_macro2::Span::call_site());
        let u_dir_pos_indent = syn::Ident::new(&format!("U{}_DIR_POS", i), proc_macro2::Span::call_site());
        let s = quote!{
            const #u_dir_indent: [Direction<#u_indent>; #i * 2] = [ #(#array_direction),* ];
            const #u_dir_pos_indent: [Direction<#u_indent>; #i] = [ #(#array_direction_positives),* ];
            impl DirectionList for Direction<#u_indent> {
                fn get_all_directions() -> & 'static [Self] {
                    &#u_dir_indent
                }
                fn get_positives_directions() -> & 'static [Self] {
                    &#u_dir_pos_indent
                }
            }
        };
        implem.push(s);
    }
    let final_stream = quote!{
        #(#implem)*
    };
    final_stream.into()
}
