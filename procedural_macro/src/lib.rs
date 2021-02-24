
extern crate proc_macro;
extern crate quote;

use proc_macro::TokenStream;

#[proc_macro]
pub fn implement_direction_list(_item: TokenStream) -> TokenStream {
    let mut string = String::default();
    for i in 1..=127 {
        let mut array_direction = String::default();
        let mut array_direction_positives = String::default();
        for j in 0..i {
            array_direction.push_str(&format!("Direction{{index_dir: {}, is_positive: {}, _phantom: PhantomData}},", j, true));
            array_direction.push_str(&format!("Direction{{index_dir: {}, is_positive: {}, _phantom: PhantomData}},", j, false));
            array_direction_positives.push_str(&format!("Direction{{index_dir: {}, is_positive: {}, _phantom: PhantomData}},", j, true));
        }
        
        string.push_str(&format!("const U{i}_DIR: [Direction<U{i}>; {i} * 2] = [{array_dir}]; const U{i}_DIR_POS: [Direction<U{i}>; {i}] = [{array_positive}]; impl DirectionList for Direction<U{i}> {{ fn get_all_directions() -> & 'static [Self] {{ &U{i}_DIR }} fn get_positives_directions() -> & 'static [Self] {{ &U{i}_DIR_POS }} }}",
        i = i, array_dir = array_direction, array_positive = array_direction_positives));
    }
    string.parse().unwrap()
}
