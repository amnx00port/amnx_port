//! Choose a diagram profile from physical input only. The keyboard stand-in
//! and the dead zones used here do not change the readings given to the title.
use super::{HostPad, PADS, PadState};

#[cfg(any(windows, test))]
pub(super) fn sdl_profile(kind: i32, vendor: u16, product: u16) -> &'static str {
    // Older SDL versions may report a generic type for Valve hardware.
    // The 0x11ff Steam Virtual Gamepad does not identify the physical pad.
    let steam_hardware = vendor == 0x28de
        && matches!(
            product,
            0x1101 | 0x1102 | 0x1105 | 0x1106 | 0x1142 | 0x1201 | 0x1202 | 0x1205 | 0x1302..=0x1305
        );
    match kind {
        4..=6 => "playstation", // SDL_GAMEPAD_TYPE_PS3 through PS5
        7..=11 => "nintendo",   // Switch Pro, Joy-Cons and GameCube
        12 => "steam",          // SDL_GAMEPAD_TYPE_STEAM
        _ if steam_hardware => "steam",
        _ => "xbox",
    }
}

#[derive(Clone, Copy)]
struct Seen {
    id: u32,
    profile: &'static str,
    held: u32,
    last_used: u64,
}

#[derive(Default)]
pub(super) struct Glyphs {
    pads: [Option<Seen>; PADS],
    poll: u64,
}

impl Glyphs {
    pub(super) fn poll(&mut self, pads: &[Option<HostPad>; PADS]) {
        self.poll += 1;
        self.pads = pads.map(|pad| {
            let pad = pad?;
            let previous = self.pads.iter().flatten().find(|p| p.id == pad.id);
            let before = previous.map_or(0, |p| p.held);
            let held = controls(pad.state, before);
            // Connecting a controller with a trigger already held should
            // not displace the one the player is using.
            let last_used = previous.map_or(0, |p| {
                if held & !before != 0 {
                    self.poll
                } else {
                    p.last_used
                }
            });
            Some(Seen {
                id: pad.id,
                profile: pad.profile,
                held,
                last_used,
            })
        });
    }

    pub(super) fn profile(&self) -> &'static str {
        // Lower slots win ties, including the first poll before any input.
        self.pads
            .iter()
            .rev()
            .flatten()
            .max_by_key(|p| p.last_used)
            .map_or("xbox", |p| p.profile)
    }
}

/// The physical buttons occupy the low 14 bits. Treat a deliberate trigger
/// movement or stick direction as another button press, with hysteresis so
/// a stick held near its threshold cannot keep claiming the profile.
fn controls(pad: PadState, previous: u32) -> u32 {
    let directions = [
        pad.left_trigger,
        pad.right_trigger,
        pad.left_x,
        -pad.left_x,
        pad.left_y,
        -pad.left_y,
        pad.right_x,
        -pad.right_x,
        pad.right_y,
        -pad.right_y,
    ];
    let mut controls = pad.buttons;
    for (i, amount) in directions.iter().enumerate() {
        let bit = 1 << (16 + i);
        let threshold = if previous & bit != 0 { 0.25 } else { 0.5 };
        if *amount >= threshold {
            controls |= bit;
        }
    }
    controls
}

#[test]
fn maps_sdl_types_and_physical_valve_devices() {
    assert_eq!(sdl_profile(6, 0, 0), "playstation");
    assert_eq!(sdl_profile(7, 0, 0), "nintendo");
    assert_eq!(sdl_profile(11, 0, 0), "nintendo");
    assert_eq!(sdl_profile(12, 0, 0), "steam");
    assert_eq!(sdl_profile(1, 0x28de, 0x1205), "steam");
    assert_eq!(sdl_profile(1, 0x28de, 0x11ff), "xbox");
    assert_eq!(sdl_profile(3, 0, 0), "xbox");
}
