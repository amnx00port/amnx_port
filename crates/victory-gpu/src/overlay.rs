//! The profiler's on-screen panel (`VICTORY_PROFILE_OVERLAY=1`): its
//! image, made by `victory_profile::overlay` in the back buffer's own
//! format, copied from upload memory into the back buffer's top left
//! corner on the presenter's copy list, after the frame and before the
//! present. F3, while our window has focus, shows and hides it.
//!
//! No shaders or pipelines: one `CopyTextureRegion` from a buffer kept
//! between frames, rewritten only when the panel is redrawn (a few times a
//! second). Every submission is waited for, so the buffer is free again by
//! the next frame.

use std::sync::Arc;

use victory_profile::overlay::{Image, PixelFormat};
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;

use crate::host::Host;

/// The panel's distance from the corner, in font pixels.
const MARGIN: u32 = 8;

/// The presenter's panel state.
#[derive(Default)]
pub(crate) struct Overlay {
    /// Upload memory, mapped, and its size.
    upload: Option<(ID3D12Resource, *mut u8, u64)>,
    /// The image and format last written into it.
    written: Option<(Arc<Image>, i32)>,
    /// F3 was down at the last frame.
    key_down: bool,
}

impl Overlay {
    /// Polls F3 and, if the panel is shown and fits, records its copy into
    /// `back` (`width` x `height` of `format`, in the copy destination
    /// state) on `list`.
    pub(crate) fn record(
        &mut self,
        host: &Host,
        list: &ID3D12GraphicsCommandList,
        back: &ID3D12Resource,
        (width, height, format): (u32, u32, DXGI_FORMAT),
        hwnd: HWND,
    ) {
        self.poll_key(hwnd);
        let Some(pixel) = PixelFormat::from_dxgi(format.0 as u32) else {
            return;
        };
        let Some(image) = victory_profile::overlay::image(height) else {
            return;
        };
        let margin = MARGIN * image.scale;
        if image.width + margin > width || image.height + margin > height {
            return;
        }
        let row_pitch = (image.width * pixel.bytes() as u32)
            .next_multiple_of(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
        let size = u64::from(row_pitch) * u64::from(image.height);
        if self.upload.as_ref().is_none_or(|&(_, _, have)| have < size) {
            self.written = None;
            self.upload = host
                .upload_buffer(size)
                .map(|(buffer, data)| (buffer, data, size));
        }
        let Some((buffer, data, have)) = &self.upload else {
            return;
        };
        let current = self
            .written
            .as_ref()
            .is_some_and(|(i, f)| Arc::ptr_eq(i, &image) && *f == format.0);
        if !current {
            // SAFETY: the buffer is mapped for `have` bytes and the GPU is
            // done with it (the last frame's submission was waited for).
            let out = unsafe { std::slice::from_raw_parts_mut(*data, *have as usize) };
            victory_profile::overlay::write_rows(&image, pixel, row_pitch as usize, out);
            self.written = Some((image.clone(), format.0));
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(back) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: format,
                        Width: image.width,
                        Height: image.height,
                        Depth: 1,
                        RowPitch: row_pitch,
                    },
                },
            },
        };
        // SAFETY: an open list; the back buffer is a copy destination and
        // the footprint lies inside the buffer.
        unsafe { list.CopyTextureRegion(&to, margin, margin, 0, &from, None) };
    }

    /// Toggles the panel when F3 goes down while our window has focus.
    fn poll_key(&mut self, hwnd: HWND) {
        use windows::Win32::UI::Input::KeyboardAndMouse::{GetAsyncKeyState, VK_F3};
        use windows::Win32::UI::WindowsAndMessaging::GetForegroundWindow;
        // SAFETY: plain queries.
        let down =
            unsafe { GetForegroundWindow() == hwnd && GetAsyncKeyState(i32::from(VK_F3.0)) < 0 };
        if down && !self.key_down {
            victory_profile::overlay::toggle();
        }
        self.key_down = down;
    }
}
