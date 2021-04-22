//use plotter_backend_text::*;
use na::{Complex, ComplexField};
use plotters::prelude::*;

const STEP_BY: usize = 1_000;

pub fn plot_data(
    data: &[[f64; 2]],
    delta_t: f64,
    file_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let data_mean = data;

    let mut y_min = data_mean[0][0];
    let mut y_max = data_mean[0][0];
    for el in data_mean {
        y_min = y_min.min(el[0]);
        y_max = y_max.max(el[0]);
    }

    let root = SVGBackend::new(file_name, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(60)
        .build_cartesian_2d(
            0_f64..data_mean.len() as f64 * delta_t,
            (y_min).min(0_f64)..y_max,
        )?;

    chart
        .configure_mesh()
        .y_desc("Correlation E")
        .x_desc("t")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    chart.draw_series(LineSeries::new(
        data_mean
            .iter()
            .enumerate()
            .step_by(STEP_BY)
            .map(|(index, el)| (index as f64 * delta_t, el[0])),
        BLACK.filled(),
    ))?;

    Ok(())
}

pub fn plot_data_fft(
    data: &[Complex<f64>],
    delta_t: f64,
    file_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut y_min = data[0].modulus() / (data.len() as f64).sqrt();
    let mut y_max = data[0].modulus() / (data.len() as f64).sqrt();
    for el in data {
        y_min = y_min.min(el.modulus() / (data.len() as f64).sqrt());
        y_max = y_max.max(el.modulus() / (data.len() as f64).sqrt());
    }

    let root = SVGBackend::new(file_name, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let step = 1_f64 / (delta_t * data.len() as f64 * 2_f64);

    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(60)
        .right_y_label_area_size(60)
        .build_cartesian_2d(
            (step..step * data.len() as f64).log_scale(),
            (y_min.max(1E-15)..y_max * 1.1_f64).log_scale(),
        )?
        .set_secondary_coord(
            (delta_t..data.len() as f64 * delta_t).log_scale(),
            -std::f64::consts::PI..std::f64::consts::PI,
        );

    chart
        .configure_mesh()
        .y_desc("Modulus")
        .x_desc("w")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;
    chart
        .configure_secondary_axes()
        .axis_desc_style(("sans-serif", 15))
        .y_desc("Argument")
        .draw()?;

    chart.draw_series(LineSeries::new(
        data.iter().enumerate().step_by(1).map(|(index, el)| {
            (
                index as f64 * step,
                el.modulus().max(1E-15) / (data.len() as f64).sqrt(),
            )
        }),
        &BLACK,
    ))?;

    chart.draw_secondary_series(LineSeries::new(
        data.iter()
            .enumerate()
            .step_by(1)
            .map(|(index, el)| (index as f64 * delta_t, el.argument())),
        &RED,
    ))?;

    Ok(())
}

#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn plot_data_fft_2(
    data: &[Complex<f64>],
    delta_t: f64,
    file_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut y_min = data[0].modulus() / (data.len() as f64).sqrt();
    let mut y_max = data[0].modulus() / (data.len() as f64).sqrt();
    for el in data {
        y_min = y_min.min(el.modulus() / (data.len() as f64).sqrt());
        y_max = y_max.max(el.modulus() / (data.len() as f64).sqrt());
    }

    let root = SVGBackend::new(file_name, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    const MAX_W: f64 = 4_f64;

    let step = 1_f64 / (delta_t * data.len() as f64 * 2_f64);
    let max_step = ((MAX_W / step).ceil() as usize + 1).min(data.len());

    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(60)
        .right_y_label_area_size(60)
        .build_cartesian_2d(0_f64..MAX_W, y_min..y_max)?;

    chart
        .configure_mesh()
        .y_desc("Modulus")
        .x_desc("w")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    chart.draw_series(
        data.iter()
            .take(max_step)
            .enumerate()
            .step_by(1)
            .map(|(index, el)| {
                Circle::new(
                    (
                        index as f64 * step,
                        el.modulus() / (data.len() as f64).sqrt(),
                    ),
                    1,
                    BLACK.filled(),
                )
            }),
    )?;

    Ok(())
}
