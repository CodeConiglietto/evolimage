use std::env;
use std::f32::consts::PI;
use std::path::PathBuf;

use ggez::{
    conf::WindowMode,
    event::{self, EventHandler, KeyCode},
    graphics::{self, Color, DrawParam, Image},
    input::keyboard,
    Context, ContextBuilder, GameResult,
};
use image::{ImageBuffer, Pixel, Rgba, RgbaImage};
use ndarray::{Array2, Zip};
use palette::{encoding::srgb::Srgb, rgb::Rgb, Hsv, RgbHue};
use rand::prelude::*;
use rayon::prelude::*;
use structopt::StructOpt;

use bunnyfont::{
    char_transforms::{CharMirror, CharRotation},
    integrations::image::{ImageBunnyFont, ImageBunnyChar},
    traits::into_scalar::IntoScalar,
};
use protoplasm::prelude::*;

#[derive(Debug, StructOpt)]
struct ProgArgs {
    #[structopt(short = "-i", long = "--image", parse(from_os_str))]
    source_image_path: PathBuf,
}

fn main() {
    let args = ProgArgs::from_args();

    let source_image = image::open(args.source_image_path.clone()).unwrap().into_rgba8();

    // disable winit's hidpi
    env::set_var("WINIT_X11_SCALE_FACTOR", "2");
    env::set_var("WINIT_HIDPI_FACTOR", "2");

    // Make a Context.
    let (mut ctx, event_loop) = ContextBuilder::new("my_game", "Cool Game Author")
        .window_mode(WindowMode::default().dimensions(source_image.width() as f32, source_image.height() as f32))
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
}

impl MyGame {
    pub fn new(ctx: &mut Context, args: &ProgArgs, source_image: RgbaImage) -> MyGame {
        dbg!(ggez::filesystem::resources_dir(ctx));

        let (char_width, char_height) = (8, 8);
        let font = ImageBunnyFont::new(
            image::open(dbg!(PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("resources").join("master8x8.png"))).unwrap().into_rgba8(),
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

        MyGame {
            font,
            source_image,
            char_buffer,
            processed_char_image,
            display_frame,
        }
    }
}

impl EventHandler for MyGame {
    fn update(&mut self, ctx: &mut Context) -> GameResult<()> {

        let (char_width, char_height) = self.font.char_dimensions();
        let comparison_image = &self.source_image;

        Zip::indexed(&mut self.char_buffer).par_for_each(|(x, y), current_char| {
            // let potential_char = gen_random_char(&self.font);
            let potential_char = mutated_char(&current_char, &self.font);

            let pixel_x = x as u32 * char_width as u32;
            let pixel_y = y as u32 * char_height as u32;

            let current_score = char_score(&self.font, &current_char, comparison_image, pixel_x, pixel_y);
            let potential_score = char_score(&self.font, &potential_char, comparison_image, pixel_x, pixel_y);

            if potential_score > current_score {
                *current_char = potential_char;
            }
        });

        self.processed_char_image = process_char_image(&self.char_buffer, &self.font);
        self.display_frame = process_display_frame(ctx, &self.processed_char_image);

        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult<()> {
        graphics::clear(ctx, Color::BLACK);

        ggez::graphics::draw(
            ctx,
            &self.display_frame,
            DrawParam::new().color(Color::WHITE),
        )?;

        graphics::present(ctx)
    }
}

fn char_score(font: &ImageBunnyFont, bunny_char: &ImageBunnyChar, comparison_image: &RgbaImage, pixel_x: u32, pixel_y: u32) -> f32 {
    let (char_width, char_height) = font.char_dimensions();

    let total_pixels = char_width * char_height;

    let total_score: f32 = (0..char_width).flat_map(|x| (0..char_height).map(move |y| {
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
    })).sum();

    total_score / total_pixels as f32
}

fn process_char_image(char_buffer: &Array2<ImageBunnyChar>, font: &ImageBunnyFont) -> RgbaImage {
    let (buffer_width, buffer_height) = char_buffer.dim();

    let (char_width, char_height) = font.char_dimensions();

    RgbaImage::from_fn((buffer_width * char_width) as u32, (buffer_height * char_height) as u32, |x, y| {
        let (char_pix_x, char_pix_y) = (x as usize % char_width, y as usize % char_height);

        let (buffer_x, buffer_y) = (x as usize / char_width, y as usize / char_height);

        let bunny_char = char_buffer[[buffer_x, buffer_y]];

        font.char_pixel(&bunny_char, char_pix_x, char_pix_y)
    })
}

fn process_display_frame(ctx: &mut Context, image: &RgbaImage) -> ggez::graphics::Image {
    ggez::graphics::Image::from_rgba8(
        ctx,
        image.width() as u16,
        image.height() as u16,
        image.as_flat_samples().as_slice(),
    ).unwrap()
}

fn gen_random_char(font: &ImageBunnyFont) -> ImageBunnyChar {
    ImageBunnyChar::new(
        thread_rng().gen_range(0..(font.total_char_indices() - 80)),//TODO: remove workaround and introduce functionality to bunnyfont instead 
        [random::<u8>(),random::<u8>(),random::<u8>(),random::<u8>()].into(), 
        Some([random::<u8>(),random::<u8>(),random::<u8>(),random::<u8>()].into()), 
        random_rotation(),
        random_mirror()
    )
}

fn mutated_char(original_char: &ImageBunnyChar, font: &ImageBunnyFont) -> ImageBunnyChar {
    let new_char = original_char.clone();

    match thread_rng().gen_range(0..=7) {
        0 => new_char.index(thread_rng().gen_range(0..(font.total_char_indices() - 80))),//TODO: remove workaround and introduce functionality to bunnyfont instead

        1 => new_char.foreground(mutated_color(&new_char.foreground)),
        2 => new_char.foreground([random::<u8>(),random::<u8>(),random::<u8>(),random::<u8>()].into()),
        
        3 => new_char.background(Some(mutated_color(&new_char.background.unwrap()))),
        4 => new_char.background(Some([random::<u8>(),random::<u8>(),random::<u8>(),random::<u8>()].into())),
        
        5 => new_char.rotation(random_rotation()),
        6 => new_char.mirror(random_mirror()),
        
        7 => gen_random_char(&font),
        _ => panic!()
    }
}

fn mutated_color(original_color: &Rgba<u8>) -> Rgba<u8> {
    let mut new_color = original_color.clone();
    
    new_color.channels_mut()[thread_rng().gen_range(0..4)] = random::<u8>();

    new_color
}

fn random_rotation() -> CharRotation {
    match thread_rng().gen_range(0..=3) {
        0 => CharRotation::None,
        1 => CharRotation::Rotation90,
        2 => CharRotation::Rotation180,
        3 => CharRotation::Rotation270,
        _ => panic!()
    }
}

fn random_mirror() -> CharMirror {
    match thread_rng().gen_range(0..=3) {
        0 => CharMirror::None,
        1 => CharMirror::MirrorX,
        2 => CharMirror::MirrorY,
        3 => CharMirror::MirrorBoth,
        _ => panic!()
    }
}