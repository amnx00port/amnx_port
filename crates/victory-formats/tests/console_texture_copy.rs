use victory_formats::texture::{Dimension, Swizzle, Texture, TextureDesc};

#[test]
fn console_formats_copy_packed_texels_across_standard_microtiles() {
    // A 256-byte standard tile holds sixteen 16-byte rows of R4G4.
    // For 32-bit texels it holds two side-by-side 4x8 strips, each 128 bytes.
    for (format, bytes, tile_width) in [(189, 4, 8), (190, 1, 16)] {
        let width = tile_width + 1;
        let texture = Texture::new(&TextureDesc {
            dimension: Dimension::Tex2d,
            width: width as u32,
            height: 3,
            depth_or_array: 2,
            mips: 1,
            format,
            samples: 1,
            swizzle: Swizzle::SW_256B_S,
        })
        .unwrap();
        let row_bytes = width * bytes;
        assert_eq!(texture.footprint(1).unwrap().row_pitch, row_bytes);
        assert_eq!(texture.size, 1024);

        // Only slice 1 is copied. Leave padding between rows and omit it
        // after the last row, as a caller using the minimum buffer size can.
        let pitch = row_bytes + 5;
        let mut linear = vec![0xee; 2 * pitch + row_bytes];
        let mut expected = vec![0xcc; 1024];
        for y in 0..3 {
            for x in 0..width {
                let tile = 512 + 256 * (x / tile_width);
                let column = (x % tile_width) * bytes;
                let at = tile + (column / 16) * 128 + y * 16 + column % 16;
                for byte in 0..bytes {
                    let value = ((y * width + x) * bytes + byte) as u8;
                    linear[y * pitch + x * bytes + byte] = value;
                    expected[at + byte] = value;
                }
            }
        }

        let mut tiled = vec![0xcc; 1024];
        texture.tile(&mut tiled, 1, &linear, pitch, 0).unwrap();
        assert_eq!(tiled, expected, "tile format {format}");

        let mut decoded = vec![0xee; linear.len()];
        texture
            .detile(&expected, 1, &mut decoded, pitch, 0)
            .unwrap();
        assert_eq!(decoded, linear, "detile format {format}");
    }
}
