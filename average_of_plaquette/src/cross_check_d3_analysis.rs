use std::fs::File;
use std::vec::Vec;

use average_of_plaquette::{observable, rng::*, sim::get_pb_template};
use crossbeam::thread;
use indicatif::{ProgressBar, ProgressStyle};
use lattice_qcd_rs::statistics;
use plotters::prelude::*;
use rand::Rng;
use rayon::prelude::*;

const BETA: f64 = 24_f64;
const N_ARRAY: [usize; 15] = [10, 11, 12, 13, 14, 15, 17, 18, 21, 24, 28, 34, 44, 60, 96];
//const N_ARRAY: [usize; 10] = [10, 11, 12, 13, 14, 15, 17, 18, 21, 24];
const BOOSTRAP_NUMBER_OF_TIMES: usize = 1_000;

const SEED: u64 = 0xd6_4b_ef_fd_9f_c8_b2_a4;

type DataComputed = (usize, [f64; 2], [f64; 2], [f64; 2]);

fn main() {
    let pb = ProgressBar::new(N_ARRAY.len() as u64);

    pb.set_style(
        ProgressStyle::default_bar()
            .progress_chars("=>-")
            .template(get_pb_template()),
    );
    pb.enable_steady_tick(300);

    let data = N_ARRAY
        .par_iter()
        .map(|n_size| {
            let file_name = format!("raw_measures_{}.csv", n_size);
            let result = read_file(&file_name, 1_000).unwrap();
            let (mean_and_err_block, mean_and_err_individual, mean_and_err_mean) =
                thread::scope(|s| {
                    let handle_block = s.spawn(|_| {
                        let mut rng = get_rand_from_seed(SEED);
                        rng.jump();
                        statistical_boot_strap_method_block(
                            &result,
                            BETA,
                            BOOSTRAP_NUMBER_OF_TIMES,
                            &mut rng,
                        )
                    });
                    let handle_individual = s.spawn(|_| {
                        let mut rng = get_rand_from_seed(SEED);
                        rng.jump();
                        rng.jump();
                        statistical_boot_strap_method_individual(
                            &result,
                            BETA,
                            BOOSTRAP_NUMBER_OF_TIMES,
                            &mut rng,
                        )
                    });

                    let result_mean = result
                        .iter()
                        .map(|vec| observable::parameter_volume(statistics::mean(vec), BETA))
                        .collect::<Vec<f64>>();
                    let mut rng = get_rand_from_seed(SEED);
                    let mean_and_err_mean = statistical_boot_strap_mean(
                        &result_mean,
                        BOOSTRAP_NUMBER_OF_TIMES,
                        &mut rng,
                    );

                    let mean_and_err_block = handle_block.join().unwrap();
                    let mean_and_err_individual = handle_individual.join().unwrap();
                    (
                        mean_and_err_block,
                        mean_and_err_individual,
                        mean_and_err_mean,
                    )
                })
                .unwrap();
            pb.inc(1);
            (
                *n_size,
                mean_and_err_block,
                mean_and_err_individual,
                mean_and_err_mean,
            )
        })
        .collect::<Vec<DataComputed>>();
    pb.finish();

    let data_b = data
        .iter()
        .map(|(n_size, mean_and_err_block, _, _)| (*n_size, *mean_and_err_block))
        .collect::<Vec<(usize, [f64; 2])>>();
    plot_data(&data_b, BETA, "plot_volume_data_block.svg").unwrap();

    let data_i = data
        .iter()
        .map(|(n_size, _, mean_and_err_individual, _)| (*n_size, *mean_and_err_individual))
        .collect::<Vec<(usize, [f64; 2])>>();
    plot_data(&data_i, BETA, "plot_volume_data_individual.svg").unwrap();

    let data_m = data
        .iter()
        .map(|(n_size, _, _, mean_and_err_mean)| (*n_size, *mean_and_err_mean))
        .collect::<Vec<(usize, [f64; 2])>>();
    plot_data(&data_m, BETA, "plot_volume_data_mean.svg").unwrap();
    write_to_file_csv(&data, "data_raw_bootstrap.csv").unwrap();
}

fn write_to_file_csv(data: &[DataComputed], name: &str) -> std::io::Result<()> {
    let file = File::create(name)?;
    let mut wtr = csv::Writer::from_writer(file);
    for data_el in data {
        wtr.serialize(data_el)?;
    }
    wtr.flush()?;
    Ok(())
}

fn read_file(file_name: &str, size_hint: usize) -> Result<Vec<Vec<f64>>, csv::Error> {
    let file_handler = File::open(file_name)?;
    let mut reader = csv::Reader::from_reader(file_handler);

    let mut result_vec = Vec::with_capacity(size_hint);

    for result in reader.deserialize() {
        let v: Vec<f64> = result?;
        result_vec.push(v);
    }
    Ok(result_vec)
}

fn statistical_boot_strap_method_block(
    data: &[Vec<f64>],
    beta: f64,
    number_of_times: usize,
    rng: &mut impl Rng,
) -> [f64; 2] {
    let size = data.len();
    let mut mean_vec = Vec::with_capacity(number_of_times);
    for _ in 0..number_of_times {
        let mut data_set = Vec::with_capacity(size * data[0].len());
        for _ in 0..size {
            let pos_1 = rng.gen_range(0..data.len());

            let mut data_s = data[pos_1]
                .iter()
                .map(|el| observable::parameter_volume(*el, beta))
                .collect();
            data_set.append(&mut data_s);
        }

        let mean = statistics::mean(&data_set);
        mean_vec.push(mean);
    }

    let [mean, var] = statistics::mean_and_variance(&mean_vec);
    let err = (var * (mean_vec.len() as f64 - 1_f64) / (mean_vec.len() as f64)).sqrt();
    [mean, err]
}

fn statistical_boot_strap_method_individual(
    data: &[Vec<f64>],
    beta: f64,
    number_of_times: usize,
    rng: &mut impl Rng,
) -> [f64; 2] {
    let size = data.len() * data[0].len();
    let mut mean_vec = Vec::with_capacity(number_of_times);
    for _ in 0..number_of_times {
        let mut data_set = Vec::with_capacity(size);
        for _ in 0..size {
            let pos_1 = rng.gen_range(0..data.len());
            let pos_2 = rng.gen_range(0..data[pos_1].len());
            let data_r = observable::parameter_volume(data[pos_1][pos_2], beta);
            data_set.push(data_r);
        }

        let mean = statistics::mean(&data_set);
        mean_vec.push(mean);
    }

    let [mean, var] = statistics::mean_and_variance(&mean_vec);
    let err = (var * (mean_vec.len() as f64 - 1_f64) / (mean_vec.len() as f64)).sqrt();
    [mean, err]
}

fn statistical_boot_strap_mean(
    data: &[f64],
    number_of_times: usize,
    rng: &mut impl Rng,
) -> [f64; 2] {
    let size = data.len();
    let mut mean_vec = Vec::with_capacity(number_of_times);
    for _ in 0..number_of_times {
        let mut data_set = Vec::with_capacity(size);
        for _ in 0..size {
            let pos_1 = rng.gen_range(0..data.len());
            data_set.push(data[pos_1]);
        }

        let mean = statistics::mean(&data_set);
        mean_vec.push(mean);
    }

    let [mean, var] = statistics::mean_and_variance(&mean_vec);
    let err = (var * (mean_vec.len() as f64 - 1_f64) / (mean_vec.len() as f64)).sqrt();
    [mean, err]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ErrorPlot {
    NoData,
}

impl core::fmt::Display for ErrorPlot {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoData => write!(f, "No Data"),
        }
    }
}

impl std::error::Error for ErrorPlot {}

// format of the data: size [mean, error]
fn plot_data(
    data: &[(usize, [f64; 2])],
    beta: f64,
    file_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if data.is_empty() {
        return Err(ErrorPlot::NoData.into());
    }

    let mut y_min = data[0].1[0] - data[0].1[1];
    let mut y_max = data[0].1[0] + data[0].1[1];
    let mut err_max = data[0].1[1];
    for el in data {
        err_max = err_max.max(el.1[1]);
        y_min = y_min.min(el.1[0] - el.1[1]);
        y_max = y_max.max(el.1[0] + el.1[1]);
    }

    let root = SVGBackend::new(file_name, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(60)
        .build_cartesian_2d(0_f64..2.6, (y_min - err_max).min(0_f64)..y_max + err_max)?;

    chart
        .configure_mesh()
        .y_desc("")
        .x_desc("Beta / N")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;
    chart.draw_series(data.iter().map(|(n, [mean, err])| {
        ErrorBar::new_vertical(
            beta / (*n as f64),
            mean - err,
            *mean,
            mean + err,
            BLACK.filled(),
            10,
        )
    }))?;
    Ok(())
}
