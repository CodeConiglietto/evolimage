use tui::text::Span;
use tui::layout::Constraint;
use tui::layout::Layout;
use ggez::input::keyboard::KeyMods;
use std::collections::VecDeque;
use std::env;
use std::f32::consts::PI;
use std::io;
use std::io::Stdout;
use std::path::PathBuf;
use termion::raw::RawTerminal;
use tui::style::Style;
use tui::widgets::Axis;
use tui::widgets::Block;
use tui::widgets::Chart;
use tui::widgets::Dataset;
use tui::layout::{Direction};
use tui::widgets::Borders;

use tui::{symbols::Marker, widgets::GraphType, Terminal};

use itertools::Itertools;

use termion::raw::IntoRawMode;
use tui::backend::{Backend, TermionBackend};

use float_ord::FloatOrd;
use ggez::{
    conf::WindowMode,
    event::{self, EventHandler, KeyCode},
    graphics::{self, Color, DrawParam, Image},
    input::keyboard,
    Context, ContextBuilder, GameResult,
};
use image::{
    imageops::{FilterType, *},
    ImageBuffer, Pixel, Rgba, RgbaImage,
};
use ndarray::{Array2, Zip};
use palette::{encoding::srgb::Srgb, rgb::Rgb, Hsv, RgbHue};
use rand::prelude::*;
use rayon::prelude::*;
use structopt::StructOpt;

use bunnyfont::{
    char_transforms::{CharMirror, CharRotation},
    integrations::image::{ImageBunnyChar, ImageBunnyFont},
    traits::into_scalar::IntoScalar,
};
use protoplasm::prelude::*;

#[derive(Debug, StructOpt)]
struct ProgArgs {
    #[structopt(short = "-i", long = "--image", parse(from_os_str))]
    source_image_path: PathBuf,

    #[structopt(long = "--scale", default_value = "1.0")]
    scale_factor: f32,
}

fn main() {
    let args = ProgArgs::from_args();

    let source_image = image::open(args.source_image_path.clone())
        .unwrap()
        .into_rgba8();

    let (resized_width, resized_height) = (
        (source_image.width() as f32 * args.scale_factor) as u32,
        (source_image.height() as f32 * args.scale_factor) as u32,
    );

    let source_image = resize(
        &source_image,
        resized_width,
        resized_height,
        FilterType::Nearest,
    );

    // disable winit's hidpi
    env::set_var("WINIT_X11_SCALE_FACTOR", "2");
    env::set_var("WINIT_HIDPI_FACTOR", "2");

    // Make a Context.
    let (mut ctx, event_loop) = ContextBuilder::new("my_game", "Cool Game Author")
        .window_mode(
            WindowMode::default()
                .dimensions(source_image.width() as f32 * 2.0, source_image.height() as f32 * 2.0),
        )
        .build()
        .expect("aieee, could not create ggez context!");

    // Create an instance of your event handler.
    // Usually, you should provide it with the Context object to
    // use when setting your game up.
    let my_game = MyGame::new(&mut ctx, &args, source_image);

    // Run!
    event::run(ctx, event_loop, my_game);
}

struct MyGame {
    font: ImageBunnyFont,
    source_image: RgbaImage,
    char_buffer: Array2<ImageBunnyChar>,
    processed_char_image: RgbaImage,
    display_frame: ggez::graphics::Image,
    terminal: Terminal<TermionBackend<RawTerminal<Stdout>>>,
    current_tic: usize,
    target_scores: VecDeque<(usize, f32)>,
    scores: VecDeque<(usize, f32)>,
    deltas: VecDeque<(usize, f32)>,
}

impl MyGame {
    pub fn new(ctx: &mut Context, args: &ProgArgs, source_image: RgbaImage) -> MyGame {
        println!("Starting EvolImage");

        let (char_width, char_height) = (8, 8);
        let font = ImageBunnyFont::new(
            image::open(dbg!(PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
                .join("resources")
                .join("master8x8.png")))
            .unwrap()
            .into_rgba8(),
            (char_width, char_height),
        );

        let (char_width, char_height) = font.char_dimensions();

        let char_buf_width = source_image.width() as usize / char_width;
        let char_buf_height = source_image.height() as usize / char_height;

        let char_buffer = Array2::from_shape_fn((char_buf_width, char_buf_height), |(_, _)| {
            gen_random_char(&font)
        });

        let processed_char_image = process_char_image(&char_buffer, &font);

        let display_frame = process_display_frame(ctx, &processed_char_image);

        let mut terminal =
            Terminal::new(TermionBackend::new(io::stdout().into_raw_mode().unwrap())).unwrap();
        terminal.clear().unwrap();

        let current_tic = 1;

        let mut target_scores = VecDeque::new();
        target_scores.push_front((0, 0.0));

        let mut scores = VecDeque::new();
        scores.push_front((0, 0.0));

        let mut deltas = VecDeque::new();
        deltas.push_front((0, 0.0));

        MyGame {
            font,
            source_image,
            char_buffer,
            processed_char_image,
            display_frame,
            terminal,
            current_tic,
            target_scores,
            scores,
            deltas,
        }
    }
}

impl EventHandler for MyGame {
    fn quit_event(&mut self, _ctx: &mut Context) -> bool {
        perform_cleanup();

        false
    }

    fn key_down_event(
        &mut self,
        ctx: &mut Context,
        keycode: KeyCode,
        _keymods: KeyMods,
        _repeat: bool,
    ) {
        if keycode == KeyCode::Escape {
            perform_cleanup();
            ggez::event::quit(ctx);
        }
    }

    fn update(&mut self, ctx: &mut Context) -> GameResult<()> {
        let (char_width, char_height) = self.font.char_dimensions();
        let comparison_image = &self.source_image;

        let (buffer_width, buffer_height) = self.char_buffer.dim();

        let total_chars = buffer_width * buffer_height;

        let mut target_score = self.target_scores[0].1;

        let frame_score = Zip::indexed(&mut self.char_buffer).par_fold(
            || 0.0,
            |sum, (x, y), current_char| {
                let pixel_x = x as u32 * char_width as u32;
                let pixel_y = y as u32 * char_height as u32;

                let current_score = char_score(
                    &self.font,
                    &current_char,
                    comparison_image,
                    pixel_x,
                    pixel_y,
                );

                if current_score < target_score {
                    let potential_chars: Vec<(ImageBunnyChar, f32)> = (0..4)
                        .map(|_| {
                            let mutated_char = mutated_char(&current_char, &self.font);
                            let char_score = char_score(
                                &self.font,
                                &mutated_char,
                                comparison_image,
                                pixel_x,
                                pixel_y,
                            );

                            (mutated_char, char_score)
                        })
                        .collect();

                    let (best_potential_char, best_potential_score) = potential_chars
                        .iter()
                        .max_by_key(|(_, score)| FloatOrd(*score))
                        .unwrap();

                    if *best_potential_score > current_score {
                        *current_char = *best_potential_char;
                        sum + best_potential_score
                    } else {
                        sum + current_score
                    }
                } else {
                    sum + current_score
                }
            },
            |a, b| a + b,
        ) / total_chars as f32;

        self.processed_char_image = process_char_image(&self.char_buffer, &self.font);
        self.display_frame = process_display_frame(ctx, &self.processed_char_image);

        let frame_score_delta = frame_score - self.scores[0].1;

        // print!("\rCurrent frame score is: {:.10}, score delta is: {:.10}", frame_score, frame_score_delta);

        if frame_score > target_score {
            target_score = target_score + (1.0 - target_score) * 0.01;
        }

        self.target_scores
            .push_front((self.current_tic, target_score));
        self.scores
            .push_front((self.current_tic, frame_score));
        self.deltas
            .push_front((self.current_tic, frame_score_delta));

        if self.target_scores.len() > 256 {
            self.target_scores.pop_back();
        }
        if self.scores.len() > 256 {
            self.scores.pop_back();
        }
        if self.deltas.len() > 256 {
            self.deltas.pop_back();
        }

        let target_score_range = self
            .target_scores
            .iter()
            .minmax_by_key(|(_, score)| FloatOrd(*score))
            .into_option()
            .unwrap();
        let score_range = self
            .scores
            .iter()
            .minmax_by_key(|(_, score)| FloatOrd(*score))
            .into_option()
            .unwrap();
        let delta_range = self
            .deltas
            .iter()
            .minmax_by_key(|(_, score)| FloatOrd(*score))
            .into_option()
            .unwrap();

        self.terminal
            .draw(|frame| {
                let raw_target_scores = self
                    .target_scores
                    .iter()
                    .map(|(tic, target_score)| (*tic as f64, *target_score as f64))
                    .collect::<Vec<_>>();
                let target_scores = Dataset::default()
                    .name("Target")
                    .marker(Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(tui::style::Color::Green))
                    .data(&raw_target_scores);
                let raw_scores = self
                    .scores
                    .iter()
                    .map(|(tic, score)| (*tic as f64, *score as f64))
                    .collect::<Vec<_>>();
                let scores = Dataset::default()
                    .name("Score")
                    .marker(Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(tui::style::Color::Red))
                    .data(&raw_scores);
                let raw_deltas = self
                    .deltas
                    .iter()
                    .map(|(tic, delta)| (*tic as f64, *delta as f64))
                    .collect::<Vec<_>>();
                let deltas = Dataset::default()
                    .name("Delta")
                    .marker(Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(tui::style::Color::Blue))
                    .data(&raw_deltas);

                    let unified_score_range = (score_range.0.1.min(target_score_range.0.1) as f64, score_range.1.1.max(target_score_range.1.1) as f64);

                    let score_chart = Chart::new(vec![target_scores, scores])
                        .block(Block::default().title("Score").borders(Borders::ALL))
                        .x_axis(
                            Axis::default()
                                .bounds([self.current_tic as f64 - 256.0, self.current_tic as f64]),
                        )
                        .y_axis(
                            Axis::default().bounds([unified_score_range.0 as f64, unified_score_range.1 as f64])
                                .labels(vec!(Span::from(format!("{:.10}", unified_score_range.0)), Span::from(format!("{:.10}", unified_score_range.1)))),
                        );

                    let delta_chart = Chart::new(vec![deltas])
                        .block(Block::default().title("Delta").borders(Borders::ALL))
                        .x_axis(
                            Axis::default()
                                .bounds([self.current_tic as f64 - 256.0, self.current_tic as f64]),
                        )
                        .y_axis(
                            Axis::default().bounds([delta_range.0.1 as f64, delta_range.1.1 as f64])
                                .labels(vec!(Span::from(format!("{:.10}", delta_range.0.1)), Span::from(format!("{:.10}", delta_range.1.1)))),
                        );

                let layout = Layout::default().direction(Direction::Horizontal).constraints([Constraint::Ratio(1, 2); 2]).split(frame.size());

                frame.render_widget(score_chart, layout[0]);
                frame.render_widget(delta_chart, layout[1]);
            })
            .unwrap();

        self.current_tic += 1;

        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult<()> {
        graphics::clear(ctx, Color::BLACK);

        ggez::graphics::draw(
            ctx,
            &self.display_frame,
            DrawParam::new().scale([2.0, 2.0]).color(Color::WHITE),
        )?;

        graphics::present(ctx)
    }
}

fn char_score(
    font: &ImageBunnyFont,
    bunny_char: &ImageBunnyChar,
    comparison_image: &RgbaImage,
    pixel_x: u32,
    pixel_y: u32,
) -> f32 {
    let (char_width, char_height) = font.char_dimensions();

    let total_pixels = char_width * char_height;

    let total_score: f32 = (0..char_width)
        .flat_map(|x| {
            (0..char_height).map(move |y| {
                let a = font.char_pixel(bunny_char, x, y);
                let b = comparison_image.get_pixel(x as u32 + pixel_x, y as u32 + pixel_y);

                let r_score = (a.0[0] as f32 - b.0[0] as f32).abs() / 256.0;
                let g_score = (a.0[1] as f32 - b.0[1] as f32).abs() / 256.0;
                let b_score = (a.0[2] as f32 - b.0[2] as f32).abs() / 256.0;
                let a_score = (a.0[3] as f32 - b.0[3] as f32).abs() / 256.0;

                let final_score = 1.0 - ((r_score + g_score + b_score + a_score) / 4.0);

                assert!(final_score >= 0.0);
                assert!(final_score <= 1.0);

                final_score
            })
        })
        .sum();

    total_score / total_pixels as f32
}

fn process_char_image(char_buffer: &Array2<ImageBunnyChar>, font: &ImageBunnyFont) -> RgbaImage {
    let (buffer_width, buffer_height) = char_buffer.dim();

    let (char_width, char_height) = font.char_dimensions();

    RgbaImage::from_fn(
        (buffer_width * char_width) as u32,
        (buffer_height * char_height) as u32,
        |x, y| {
            let (char_pix_x, char_pix_y) = (x as usize % char_width, y as usize % char_height);

            let (buffer_x, buffer_y) = (x as usize / char_width, y as usize / char_height);

            let bunny_char = char_buffer[[buffer_x, buffer_y]];

            font.char_pixel(&bunny_char, char_pix_x, char_pix_y)
        },
    )
}

fn process_display_frame(ctx: &mut Context, image: &RgbaImage) -> ggez::graphics::Image {
    ggez::graphics::Image::from_rgba8(
        ctx,
        image.width() as u16,
        image.height() as u16,
        image.as_flat_samples().as_slice(),
    )
    .unwrap()
}

fn gen_random_char(font: &ImageBunnyFont) -> ImageBunnyChar {
    ImageBunnyChar::new(
        thread_rng().gen_range(0..(font.total_char_indices() - 80)), //TODO: remove workaround and introduce functionality to bunnyfont instead
        [
            random::<u8>(),
            random::<u8>(),
            random::<u8>(),
            random::<u8>(),
        ]
        .into(),
        Some(
            [
                random::<u8>(),
                random::<u8>(),
                random::<u8>(),
                random::<u8>(),
            ]
            .into(),
        ),
        random_rotation(),
        random_mirror(),
    )
}

fn mutated_char(original_char: &ImageBunnyChar, font: &ImageBunnyFont) -> ImageBunnyChar {
    let new_char = original_char.clone();

    match thread_rng().gen_range(0..=7) {
        0 => new_char.index(thread_rng().gen_range(0..(font.total_char_indices() - 80))), //TODO: remove workaround and introduce functionality to bunnyfont instead

        1 => new_char.foreground(mutated_color(&new_char.foreground)),
        2 => new_char.foreground(
            [
                random::<u8>(),
                random::<u8>(),
                random::<u8>(),
                random::<u8>(),
            ]
            .into(),
        ),

        3 => new_char.background(Some(mutated_color(&new_char.background.unwrap()))),
        4 => new_char.background(Some(
            [
                random::<u8>(),
                random::<u8>(),
                random::<u8>(),
                random::<u8>(),
            ]
            .into(),
        )),

        5 => new_char.rotation(random_rotation()),
        6 => new_char.mirror(random_mirror()),

        7 => gen_random_char(&font),
        _ => panic!(),
    }
}

fn mutated_color(original_color: &Rgba<u8>) -> Rgba<u8> {
    let mut new_color = original_color.clone();

    let channel = thread_rng().gen_range(0..4);

    if thread_rng().gen_bool(0.5) {
        if thread_rng().gen_bool(0.5) {
            new_color.channels_mut()[channel] = new_color.channels_mut()[channel].saturating_add(1);
        } else {
            new_color.channels_mut()[channel] = new_color.channels_mut()[channel].saturating_sub(1);
        }
    } else {
        new_color.channels_mut()[channel] = random::<u8>();
    }

    new_color
}

fn random_rotation() -> CharRotation {
    match thread_rng().gen_range(0..=3) {
        0 => CharRotation::None,
        1 => CharRotation::Rotation90,
        2 => CharRotation::Rotation180,
        3 => CharRotation::Rotation270,
        _ => panic!(),
    }
}

fn random_mirror() -> CharMirror {
    match thread_rng().gen_range(0..=3) {
        0 => CharMirror::None,
        1 => CharMirror::MirrorX,
        2 => CharMirror::MirrorY,
        3 => CharMirror::MirrorBoth,
        _ => panic!(),
    }
}

fn perform_cleanup() {
    println!("\r\n");
    println!("Thank you for using EvolImage\r");
}
