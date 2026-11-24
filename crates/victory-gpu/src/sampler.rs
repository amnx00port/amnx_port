//! Samplers as the host describes them, from the console's sampler
//! description (`CreateSamplerX`), which names the hardware's sampler
//! fields. (Immediate samplers in shader code come decoded with the
//! translation.)

use windows::Win32::Graphics::Direct3D12::*;

/// The fields both forms share.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Fields {
    /// Point 0, bilinear 1, anisotropic point 2, anisotropic bilinear 3.
    mag: u32,
    min: u32,
    /// None 0, point 1, linear 2.
    mip: u32,
    /// Plain 0, minimum 1, maximum 2.
    mode: u32,
    /// Host address modes.
    address: [u32; 3],
    anisotropy: u32,
    /// Host comparison function.
    compare: u32,
    min_lod: f32,
    max_lod: f32,
    bias: f32,
    border: [f32; 4],
}

impl Fields {
    fn host(&self) -> D3D12_SAMPLER_DESC {
        let linear = |f: u32| u32::from(f == 1 || f == 3);
        let aniso = self.mag >= 2 || self.min >= 2;
        // The console compares when the instruction says so, with the
        // sampler's function; the host needs a comparison sampler for that.
        // A sampler given a real test (less ... greater-or-equal) is one;
        // never, always and none are plain samplers.
        const LESS: u32 = 2;
        const GREATER_EQUAL: u32 = 7;
        let compares = (LESS..=GREATER_EQUAL).contains(&self.compare);
        let reduction = match self.mode {
            1 => 2,
            2 => 3,
            _ => u32::from(compares),
        };
        let filter = if aniso {
            0x55
        } else {
            linear(self.min) << 4 | linear(self.mag) << 2 | u32::from(self.mip == 2)
        };
        let addr = |mode: u32| {
            if (1..=5).contains(&mode) {
                mode
            } else {
                3 // D3D12_TEXTURE_ADDRESS_MODE_CLAMP
            }
        };
        let bias = if self.bias.is_finite() {
            self.bias.clamp(-16.0, 15.99)
        } else {
            0.0
        };
        let min_lod = if self.min_lod.is_nan() {
            0.0
        } else {
            self.min_lod
        };
        let max_lod = if self.mip == 0 {
            min_lod
        } else if self.max_lod.is_nan() {
            f32::MAX
        } else {
            self.max_lod.max(min_lod)
        };
        let compare = if (1..=8).contains(&self.compare) {
            self.compare
        } else {
            1 // D3D12_COMPARISON_FUNC_NEVER
        };
        D3D12_SAMPLER_DESC {
            Filter: D3D12_FILTER((reduction << 7 | filter) as i32),
            AddressU: D3D12_TEXTURE_ADDRESS_MODE(addr(self.address[0]) as i32),
            AddressV: D3D12_TEXTURE_ADDRESS_MODE(addr(self.address[1]) as i32),
            AddressW: D3D12_TEXTURE_ADDRESS_MODE(addr(self.address[2]) as i32),
            MipLODBias: bias,
            MaxAnisotropy: if aniso {
                self.anisotropy.clamp(1, 16)
            } else {
                1
            },
            ComparisonFunc: D3D12_COMPARISON_FUNC(compare as i32),
            BorderColor: self.border,
            MinLOD: min_lod,
            // Without a mip filter the console samples mip 0 only.
            MaxLOD: max_lod,
        }
    }
}

/// The console's sampler description (27 words): flags, magnification,
/// minification, mip and depth filters, filter mode, three address modes,
/// minimum and maximum LOD, LOD bias, secondary bias, anisotropy, its bias
/// and threshold, comparison, border color, then tuning words.
pub(crate) fn console(words: &[u32; 27]) -> D3D12_SAMPLER_DESC {
    let f = |i: usize| f32::from_bits(words[i]);
    Fields {
        mag: words[1],
        min: words[2],
        mip: words[3],
        mode: words[5],
        address: [words[6], words[7], words[8]],
        min_lod: f(9),
        max_lod: f(10),
        bias: f(11),
        anisotropy: words[13],
        compare: words[16],
        border: [f(17), f(18), f(19), f(20)],
    }
    .host()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn console_descriptions_decode() {
        let mut w = [0u32; 27];
        w[1] = 1; // bilinear magnification
        w[2] = 0; // point minification
        w[3] = 2; // linear mips
        w[6..9].copy_from_slice(&[3, 3, 3]);
        w[10] = 1000.0f32.to_bits();
        let d = console(&w);
        assert_eq!(d.Filter.0, 0x5, "min point, mag and mip linear");
        assert_eq!(d.AddressW, D3D12_TEXTURE_ADDRESS_MODE_CLAMP);
        assert_eq!(d.MaxLOD, 1000.0);
        // A less-or-equal test makes a comparison sampler; never does not.
        w[16] = 4;
        assert_eq!(console(&w).Filter.0, 0x85);
        assert_eq!(console(&w).ComparisonFunc, D3D12_COMPARISON_FUNC_LESS_EQUAL);
        w[16] = 1;
        assert_eq!(console(&w).Filter.0, 0x5);
    }

    #[test]
    fn sampler_clamps_invalid_and_nan_fields() {
        let mut w = [0u32; 27];
        w[3] = 2; // linear mips
        w[6] = 0; // invalid address mode -> CLAMP (3)
        w[7] = 99; // invalid address mode -> CLAMP (3)
        w[8] = 1; // WRAP (1)
        w[9] = f32::NAN.to_bits(); // min_lod -> 0.0
        w[10] = (-5.0f32).to_bits(); // max_lod < min_lod -> clamped to min_lod (0.0)
        w[11] = 40.0f32.to_bits(); // bias -> clamped to 15.99
        w[16] = 99; // invalid compare -> NEVER (1)
        let d = console(&w);
        assert_eq!(d.AddressU, D3D12_TEXTURE_ADDRESS_MODE_CLAMP);
        assert_eq!(d.AddressV, D3D12_TEXTURE_ADDRESS_MODE_CLAMP);
        assert_eq!(d.AddressW, D3D12_TEXTURE_ADDRESS_MODE_WRAP);
        assert_eq!(d.MinLOD, 0.0);
        assert_eq!(d.MaxLOD, 0.0);
        assert_eq!(d.MipLODBias, 15.99);
        assert_eq!(d.ComparisonFunc, D3D12_COMPARISON_FUNC_NEVER);
    }
}
