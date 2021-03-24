
//! copied form (https://github.com/38/plotters/blob/master/examples/console.rs) and sligthly modified
//!
//! This file is distributed under the additional notice.
//!
//! MIT License
//!
//! Copyright (c) 2019 Hao Hou <haohou302@gmail.com>
//!
//! Permission is hereby granted, free of charge, to any person obtaining a copy
//! of this software and associated documentation files (the "Software"), to deal
//! in the Software without restriction, including without limitation the rights
//! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! copies of the Software, and to permit persons to whom the Software is
//! furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in all
//! copies or substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//! SOFTWARE.

use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, VPos};
use plotters::coord::ranged1d::AsRangedCoord;
use plotters_backend::{
  BackendColor, BackendStyle, BackendTextStyle, DrawingBackend, DrawingErrorKind,
};
use std::error::Error;
use plotters::coord::types::RangedCoordf64;

#[derive(Copy, Clone)]
pub enum PixelState {
    Empty,
    HLine,
    VLine,
    Cross,
    Pixel,
    Text(char),
    Circle(bool),
}

impl PixelState {
    const fn to_char(self) -> char {
        match self {
            Self::Empty => ' ',
            Self::HLine => '-',
            Self::VLine => '|',
            Self::Cross => '+',
            Self::Pixel => '.',
            Self::Text(c) => c,
            Self::Circle(filled) => {
                if filled {
                    '@'
                } else {
                    'O'
                }
            }
        }
    }

    fn update(&mut self, new_state: PixelState) {
        let next_state = match (*self, new_state) {
            (Self::HLine, Self::VLine) => Self::Cross,
            (Self::VLine, Self::HLine) => Self::Cross,
            (_, Self::Circle(what)) => Self::Circle(what),
            (Self::Circle(what), _) => Self::Circle(what),
            (_, Self::Pixel) => Self::Pixel,
            (Self::Pixel, _) => Self::Pixel,
            (_, new) => new,
        };

        *self = next_state;
    }
}

pub struct TextDrawingBackend(pub Vec<PixelState>);

impl DrawingBackend for TextDrawingBackend {
    type ErrorType = std::io::Error;

    fn get_size(&self) -> (u32, u32) {
        (100, 30)
    }

    fn ensure_prepared(&mut self) -> Result<(), DrawingErrorKind<std::io::Error>> {
        Ok(())
    }

    fn present(&mut self) -> Result<(), DrawingErrorKind<std::io::Error>> {
        for r in 0..30 {
            let mut buf = String::new();
            for c in 0..100 {
                buf.push(self.0[r * 100 + c].to_char());
            }
            println!("{}", buf);
        }

        Ok(())
    }
    
    #[allow(clippy::cast_sign_loss)]
    fn draw_pixel(
        &mut self,
        pos: (i32, i32),
        color: BackendColor,
    ) -> Result<(), DrawingErrorKind<std::io::Error>> {
        if color.alpha > 0.3 {
            self.0[(pos.1 * 100 + pos.0) as usize].update(PixelState::Pixel);
        }
        Ok(())
    }
    
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    fn draw_line<S: BackendStyle>(
        &mut self,
        from: (i32, i32),
        to: (i32, i32),
        style: &S,
    ) -> Result<(), DrawingErrorKind<Self::ErrorType>> {
        if from.0 == to.0 {
            let x = from.0;
            let y0 = from.1.min(to.1);
            let y1 = from.1.max(to.1);
            for y in y0..y1 {
                self.0[(y * 100 + x) as usize].update(PixelState::VLine);
            }
            return Ok(());
        }

        if from.1 == to.1 {
            let y = from.1;
            let x0 = from.0.min(to.0);
            let x1 = from.0.max(to.0);
            for x in x0..x1 {
                self.0[(y * 100 + x) as usize].update(PixelState::HLine);
            }
            return Ok(());
        }

        plotters_backend::rasterizer::draw_line(self, from, to, style)
    }
    
    #[allow(clippy::cast_possible_truncation)]
    fn estimate_text_size<S: BackendTextStyle>(
        &self,
        text: &str,
        _: &S,
    ) -> Result<(u32, u32), DrawingErrorKind<Self::ErrorType>> {
        Ok((text.len() as u32, 1))
    }
    
    #[allow(clippy::cast_possible_wrap)]
    #[allow(clippy::cast_sign_loss)]
    fn draw_text<S: BackendTextStyle>(
        &mut self,
        text: &str,
        style: &S,
        pos: (i32, i32),
    ) -> Result<(), DrawingErrorKind<Self::ErrorType>> {
        let (width, height) = self.estimate_text_size(text, style)?;
        let (width, height) = (width as i32, height as i32);
        let dx = match style.anchor().h_pos {
            HPos::Left => 0,
            HPos::Right => -width,
            HPos::Center => -width / 2,
        };
        let dy = match style.anchor().v_pos {
            VPos::Top => 0,
            VPos::Center => -height / 2,
            VPos::Bottom => -height,
        };
        let offset = (pos.1 + dy).max(0) * 100 + (pos.0 + dx).max(0);
        for (idx, chr) in (offset..).zip(text.chars()) {
            self.0[idx as usize].update(PixelState::Text(chr));
        }
        Ok(())
    }
}

pub fn draw_chart<DB: DrawingBackend>(
    b: &DrawingArea<DB, plotters::coord::Shift>,
    range_x: impl AsRangedCoord<CoordDescType = RangedCoordf64>,
    range_y: impl AsRangedCoord<CoordDescType = RangedCoordf64>,
    series : impl Iterator<Item = (f64, f64)>,
    text: &str,
) -> Result<(), Box<dyn Error>>
where
    DB::ErrorType: 'static,
{
    let mut chart = ChartBuilder::on(&b)
        .margin(1)
        .caption(text, ("sans-serif", (10).percent_height()))
        .set_label_area_size(LabelAreaPosition::Left, (5_i32).percent_width())
        .set_label_area_size(LabelAreaPosition::Bottom, (10_i32).percent_height())
        .build_cartesian_2d(range_x, range_y)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .draw()?;

    chart.draw_series(LineSeries::new(
        series,
        &RED,
    ))?;

    b.present()?;

    Ok(())
}
