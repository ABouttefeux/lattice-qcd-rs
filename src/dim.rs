//! Depreciated, module for dimension reexport.
//!
//! The library use now const generic and not this sytem anymore. It can be safely ignored.
//! It a sytem used before const generic was introduced.

macro_rules! reexport_name_dim{
    ($($i:ident) ,+) => {
        $(
            pub use na::base::dimension::$i;
        )*
    }
}

pub use na::base::dimension::DimName;
reexport_name_dim!(
    U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20,
    U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39,
    U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53, U54, U55, U56, U57, U58,
    U59, U60, U61, U62, U63, U64, U65, U66, U67, U68, U69, U70, U71, U72, U73, U74, U75, U76, U77,
    U78, U79, U80, U81, U82, U83, U84, U85, U86, U87, U88, U89, U90, U91, U92, U93, U94, U95, U96,
    U97, U98, U99, U100, U101, U102, U103, U104, U105, U106, U107, U108, U109, U110, U111, U112,
    U113, U114, U115, U116, U117, U118, U119, U120, U121, U122, U123, U124, U125, U126, U127
);

/*
U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18,
U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37,
U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53, U54, U55, U56,
U57, U58, U59, U60, U61, U62, U63, U64, U65, U66, U67, U68, U69, U70, U71, U72, U73, U74, U75,
U76, U77, U78, U79, U80, U81, U82, U83, U84, U85, U86, U87, U88, U89, U90, U91, U92, U93, U94,
U95, U96, U97, U98, U99, U100, U101, U102, U103, U104, U105, U106, U107, U108, U109, U110,
U111, U112, U113, U114, U115, U116, U117, U118, U119, U120, U121, U122, U123, U124, U125, U126,
U127
*/

/*
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
9, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94,
95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
127
*/
/*
(0, U0), (1, U1), (2, U2), (3, U3), (4, U4), (5, U5), (6, U6), (7, U7), (8, U8), (9, U9),
(10, U10), (11, U11), (12, U12), (13, U13), (14, U14), (15, U15), (16, U16), (17, U17),
(18, U18), (19, U19), (20, U20), (21, U21), (22, U22), (23, U23), (24, U24), (25, U25),
(26, U26), (27, U27), (28, U28), (29, U29), (30, U30), (31, U31), (32, U32), (33, U33),
(34, U34), (35, U35), (36, U36), (37, U37), (38, U38), (39, U39), (40, U40), (41, U41),
(42, U42), (43, U43), (44, U44), (45, U45), (46, U46), (47, U47), (48, U48), (49, U49),
(50, U50), (51, U51), (52, U52), (53, U53), (54, U54), (55, U55), (56, U56), (57, U57),
(58, U58), (59, U59), (60, U60), (61, U61), (62, U62), (63, U63), (64, U64), (65, U65),
(66, U66), (67, U67), (68, U68), (69, U69), (70, U70), (71, U71), (72, U72), (73, U73),
(74, U74), (75, U75), (76, U76), (77, U77), (78, U78), (79, U79), (80, U80), (81, U81),
(82, U82), (83, U83), (84, U84), (85, U85), (86, U86), (87, U87), (88, U88), (89, U89),
(90, U90), (91, U91), (92, U92), (93, U93), (94, U94), (95, U95), (96, U96), (97, U97),
(98, U98), (99, U99), (100, U100), (101, U101), (102, U102), (103, U103), (104, U104),
(105, U105), (106, U106), (107, U107), (108, U108), (109, U109), (110, U110), (111, U111),
(112, U112), (113, U113), (114, U114), (115, U115), (116, U116), (117, U117), (118, U118),
(119, U119), (120, U120), (121, U121), (122, U122), (123, U123), (124, U124), (125, U125),
(126, U126), (127, U127),
*/
