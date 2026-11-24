//! Turns a title's image into something the Windows loader maps but does
//! not run. Pure byte manipulation, so it is testable on any host.

use victory_pe::layout::{self, dir, dll_flags, file_flags};
use victory_pe::{ImageInfo, PeError};

#[derive(Debug, thiserror::Error)]
pub enum PrepareError {
    #[error(transparent)]
    Pe(#[from] PeError),
    #[error("cannot patch image: {0}")]
    Patch(&'static str),
}

/// A patched copy of an image plus everything we need from the original.
#[derive(Debug, Clone)]
pub struct PreparedImage {
    /// Parsed from the *original* bytes: imports, TLS callbacks and entry
    /// point as the title expects them.
    pub info: ImageInfo,
    /// The patched bytes to hand to the Windows loader.
    pub bytes: Vec<u8>,
    /// BLAKE3 of the original bytes: identifies the exact build.
    pub original_hash: blake3::Hash,
}

impl PreparedImage {
    /// File name for the patched copy, unique per original build.
    pub fn file_name(&self) -> String {
        format!("victory-image-{}.dll", &self.original_hash.to_hex()[..16])
    }
}

/// Produces the patched copy. The changes are:
///
/// * `IMAGE_FILE_DLL` set, entry point cleared: the OS maps it like a DLL
///   and calls nothing;
/// * import directory points to its empty terminator and the bound-import
///   directory is cleared: we bind the original slots using `info.imports`;
/// * TLS callback pointer cleared: the OS still allocates static TLS for
///   the image, but callbacks only run when we call them (after binding);
///   its base relocation is made a no-op so the 0 survives a rebase;
/// * `FORCE_INTEGRITY` cleared, checksum zeroed: the copy is unsigned.
pub fn prepare(original: &[u8]) -> Result<PreparedImage, PrepareError> {
    let info = ImageInfo::parse(original)?;
    let mut bytes = original.to_vec();

    let fh = layout::file_header_offset(&bytes).ok_or(PrepareError::Patch("no file header"))?;
    let opt =
        layout::optional_header_offset(&bytes).ok_or(PrepareError::Patch("no optional header"))?;

    let characteristics = info.characteristics | file_flags::DLL;
    write_u16(
        &mut bytes,
        fh + layout::FILE_HEADER_CHARACTERISTICS,
        characteristics,
    )?;
    write_u32(&mut bytes, opt + layout::OPT_ADDRESS_OF_ENTRY_POINT, 0)?;
    write_u32(&mut bytes, opt + layout::OPT_CHECKSUM, 0)?;
    let dll_characteristics = info.dll_characteristics & !dll_flags::FORCE_INTEGRITY;
    write_u16(
        &mut bytes,
        opt + layout::OPT_DLL_CHARACTERISTICS,
        dll_characteristics,
    )?;

    // Windows 10 assigns static TLS to the prepared DLL when its empty
    // import table is present. The parser has already walked one descriptor
    // per module up to the null terminator, regardless of the recorded Size.
    if let Some(at) = layout::data_directory_offset(&bytes, dir::IMPORT) {
        let rva = layout::read_u32(&bytes, at).ok_or(PrepareError::Patch("no import RVA"))?;
        if rva != 0 {
            let terminator = u32::try_from(info.imports.len())
                .ok()
                .and_then(|count| count.checked_mul(IMPORT_DESCRIPTOR_SIZE))
                .and_then(|offset| rva.checked_add(offset))
                .ok_or(PrepareError::Patch("import terminator RVA overflow"))?;
            write_u32(&mut bytes, at, terminator)?;
            write_u32(&mut bytes, at + 4, IMPORT_DESCRIPTOR_SIZE)?;
        }
    }
    if let Some(at) = layout::data_directory_offset(&bytes, dir::BOUND_IMPORT) {
        write(&mut bytes, at, &[0; layout::DATA_DIRECTORY_SIZE])?;
    }

    if let Some(tls) = &info.tls
        && tls.callbacks_va != 0
    {
        let at = info
            .rva_to_file_offset(tls.directory_rva)
            .ok_or(PrepareError::Patch("TLS directory is not file-backed"))?;
        write_u64(
            &mut bytes,
            at as usize + layout::TLS64_ADDRESS_OF_CALLBACKS,
            0,
        )?;
        // The field's base relocation would turn the 0 into `delta` whenever
        // the loader maps the copy away from its preferred base (Wine always
        // does for DYNAMIC_BASE images), and the loader then reads callbacks
        // from there. Make that entry a no-op.
        let field_rva = tls.directory_rva + layout::TLS64_ADDRESS_OF_CALLBACKS as u32;
        for (entry, target) in dir64_relocations(&bytes, &info) {
            if target == field_rva {
                write_u16(&mut bytes, entry, 0)?; // IMAGE_REL_BASED_ABSOLUTE
            }
        }
    }

    Ok(PreparedImage {
        info,
        bytes,
        original_hash: blake3::hash(original),
    })
}

/// Size of an IMAGE_IMPORT_DESCRIPTOR in the PE file.
const IMPORT_DESCRIPTOR_SIZE: u32 = 20;

/// Every `IMAGE_REL_BASED_DIR64` base relocation: (file offset of its entry,
/// RVA it patches). Empty if the image has no readable relocation table.
fn dir64_relocations(bytes: &[u8], info: &ImageInfo) -> Vec<(usize, u32)> {
    const DIR64: u16 = 10;
    let mut out = Vec::new();
    let Some(d) = layout::data_directory_offset(bytes, dir::BASERELOC) else {
        return out;
    };
    let (Some(rva), Some(size)) = (layout::read_u32(bytes, d), layout::read_u32(bytes, d + 4))
    else {
        return out;
    };
    let Some(start) = info.rva_to_file_offset(rva).filter(|_| size != 0) else {
        return out;
    };
    let (mut at, end) = (start as usize, start as usize + size as usize);
    while at + 8 <= end {
        let (Some(page), Some(block)) =
            (layout::read_u32(bytes, at), layout::read_u32(bytes, at + 4))
        else {
            break;
        };
        let block = block as usize;
        if block < 8 || at + block > end {
            break;
        }
        for entry in (at + 8..at + block).step_by(2) {
            let Some(e) = bytes.get(entry..entry + 2) else {
                break;
            };
            let e = u16::from_le_bytes([e[0], e[1]]);
            if e >> 12 == DIR64 {
                out.push((entry, page + u32::from(e & 0xFFF)));
            }
        }
        at += block;
    }
    out
}

fn write_u16(bytes: &mut [u8], at: usize, v: u16) -> Result<(), PrepareError> {
    write(bytes, at, &v.to_le_bytes())
}
fn write_u32(bytes: &mut [u8], at: usize, v: u32) -> Result<(), PrepareError> {
    write(bytes, at, &v.to_le_bytes())
}
fn write_u64(bytes: &mut [u8], at: usize, v: u64) -> Result<(), PrepareError> {
    write(bytes, at, &v.to_le_bytes())
}
fn write(bytes: &mut [u8], at: usize, v: &[u8]) -> Result<(), PrepareError> {
    bytes
        .get_mut(at..at + v.len())
        .ok_or(PrepareError::Patch("header field outside of file"))?
        .copy_from_slice(v);
    Ok(())
}

#[cfg(test)]
mod tests {
    #[cfg_attr(not(windows), allow(unused_imports))]
    use super::*;

    #[test]
    #[cfg(windows)] // reads its own PE test binary / Windows path rules
    fn patches_own_test_binary() {
        let original = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        let prepared = prepare(&original).unwrap();
        assert!(!prepared.info.is_dll(), "info describes the original");
        assert!(prepared.info.import_count() > 0);

        let patched = ImageInfo::parse(&prepared.bytes).unwrap();
        assert!(patched.is_dll());
        assert_eq!(patched.entry_rva, 0);
        assert!(patched.imports.is_empty());
        assert_eq!(patched.dll_characteristics & dll_flags::FORCE_INTEGRITY, 0);
        if let Some(tls) = &patched.tls {
            assert_eq!(tls.callbacks_va, 0);
            let field = tls.directory_rva + layout::TLS64_ADDRESS_OF_CALLBACKS as u32;
            assert!(
                dir64_relocations(&prepared.bytes, &patched)
                    .iter()
                    .all(|&(_, target)| target != field),
                "the cleared callback pointer must not be relocated"
            );
        }
        // Nothing else moved: same sections, same size.
        assert_eq!(patched.sections.len(), prepared.info.sections.len());
        assert_eq!(prepared.bytes.len(), original.len());
    }

    #[test]
    #[cfg(windows)] // reads its own PE test binary / Windows path rules
    fn keeps_iat_slots_when_import_size_excludes_the_terminator() {
        let mut original = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        let info = ImageInfo::parse(&original).unwrap();
        let directory = layout::data_directory_offset(&original, dir::IMPORT).unwrap();
        // This linker layout counts the descriptors for real modules only.
        let descriptor_bytes = u32::try_from(info.imports.len()).unwrap() * IMPORT_DESCRIPTOR_SIZE;
        write_u32(&mut original, directory + 4, descriptor_bytes).unwrap();
        let prepared = prepare(&original).unwrap();
        let empty = layout::read_u32(&prepared.bytes, directory).unwrap();
        assert_eq!(
            empty,
            layout::read_u32(&original, directory).unwrap() + descriptor_bytes
        );
        assert_eq!(
            layout::read_u32(&prepared.bytes, directory + 4),
            Some(IMPORT_DESCRIPTOR_SIZE)
        );
        let at = info.rva_to_file_offset(empty).unwrap() as usize;
        assert_eq!(
            &prepared.bytes[at..at + IMPORT_DESCRIPTOR_SIZE as usize],
            &[0; 20]
        );
        let iat = layout::data_directory_offset(&original, dir::IAT).unwrap();
        assert_eq!(
            &prepared.bytes[iat..iat + layout::DATA_DIRECTORY_SIZE],
            &original[iat..iat + layout::DATA_DIRECTORY_SIZE]
        );
        assert_eq!(prepared.info.import_count(), info.import_count());
        for module in &prepared.info.imports {
            for entry in &module.entries {
                let at = info.rva_to_file_offset(entry.iat_slot_rva).unwrap() as usize;
                assert_eq!(&prepared.bytes[at..at + 8], &original[at..at + 8]);
            }
        }
    }

    #[cfg(windows)]
    #[test]
    fn windows_assigns_static_tls_to_a_prepared_dll() {
        // Model a title DLL using our own linked image. Loading a DLL leaves
        // its assigned index intact; the executable-specific redirect resets it.
        let mut original = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        let flags =
            layout::file_header_offset(&original).unwrap() + layout::FILE_HEADER_CHARACTERISTICS;
        let characteristics =
            ImageInfo::parse(&original).unwrap().characteristics | file_flags::DLL;
        write_u16(&mut original, flags, characteristics).unwrap();
        let prepared = prepare(&original).unwrap();
        let info = &prepared.info;
        let tls = info.tls.as_ref().expect("the test binary uses static TLS");
        let directory = info.rva_to_file_offset(tls.directory_rva).unwrap() as usize;
        let index_va = layout::read_u64(&original, directory + 16).unwrap();
        let index_rva = index_va.checked_sub(info.image_base).unwrap();
        assert!(index_rva + 4 <= u64::from(info.size_of_image));

        let folder = tempfile::tempdir().unwrap();
        let image = crate::load(&prepared, folder.path()).unwrap();
        // SAFETY: the mapped test image contains the checked DWORD above.
        let assigned =
            unsafe { ((image.base() + index_rva as usize) as *const u32).read_unaligned() };
        // SAFETY: the prepared DLL has never been attached or given to a caller.
        unsafe { image.unload() };
        assert_ne!(
            assigned, 0,
            "the process executable already owns TLS slot zero"
        );
    }

    #[test]
    #[cfg(windows)] // reads its own PE test binary / Windows path rules
    fn file_name_is_stable_per_build() {
        let original = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        let a = prepare(&original).unwrap();
        let b = prepare(&original).unwrap();
        assert_eq!(a.file_name(), b.file_name());
    }

    // Off Windows the test binary is not a PE, so the tests above are gated.
    // These build a minimal title image by hand: the same patching, on any
    // host.

    const BASE: u64 = 0x1_4000_0000;
    /// File offset of the one section, mapped at RVA 0x1000.
    const fn at(rva: usize) -> usize {
        rva - 0x1000 + 0x200
    }

    /// A PE32+ executable with one section holding an import of
    /// `kernel32.dll!Sleep`, a TLS directory with one callback, a DIR64 base
    /// relocation on that callback pointer (and one on the field before
    /// it), a bound-import directory, `FORCE_INTEGRITY` and a checksum.
    fn synthetic_exe() -> Vec<u8> {
        let mut d = vec![0u8; 0x600];
        let p16 = |d: &mut Vec<u8>, at: usize, v: u16| write_u16(d, at, v).unwrap();
        let p32 = |d: &mut Vec<u8>, at: usize, v: u32| write_u32(d, at, v).unwrap();
        let p64 = |d: &mut Vec<u8>, at: usize, v: u64| write_u64(d, at, v).unwrap();
        d[0..2].copy_from_slice(b"MZ");
        p32(&mut d, layout::DOS_E_LFANEW, 0x40);
        d[0x40..0x44].copy_from_slice(b"PE\0\0");
        p16(&mut d, 0x44, 0x8664); // AMD64
        p16(&mut d, 0x46, 1); // sections
        p16(&mut d, 0x54, 240); // optional header size
        p16(&mut d, 0x56, 0x0022); // executable, large address aware
        let opt = 0x58;
        p16(&mut d, opt, 0x20B);
        p32(&mut d, opt + layout::OPT_ADDRESS_OF_ENTRY_POINT, 0x1300);
        p64(&mut d, opt + 24, BASE);
        p32(&mut d, opt + 32, 0x1000); // section alignment
        p32(&mut d, opt + 36, 0x200); // file alignment
        p32(&mut d, opt + 56, 0x2000); // size of image
        p32(&mut d, opt + 60, 0x200); // size of headers
        p32(&mut d, opt + layout::OPT_CHECKSUM, 0xDEAD_BEEF);
        p16(&mut d, opt + 68, 3); // console
        p16(
            &mut d,
            opt + layout::OPT_DLL_CHARACTERISTICS,
            dll_flags::FORCE_INTEGRITY | dll_flags::DYNAMIC_BASE | dll_flags::NX_COMPAT,
        );
        p32(&mut d, opt + layout::OPT_NUMBER_OF_RVA_AND_SIZES, 16);
        let directory = |d: &mut Vec<u8>, index: usize, rva: u32, size: u32| {
            let at = opt + layout::OPT_DATA_DIRECTORY + index * layout::DATA_DIRECTORY_SIZE;
            p32(d, at, rva);
            p32(d, at + 4, size);
        };
        directory(&mut d, dir::IMPORT, 0x1000, 2 * IMPORT_DESCRIPTOR_SIZE);
        directory(&mut d, dir::BASERELOC, 0x1380, 12);
        directory(&mut d, 9, 0x1200, 40); // TLS
        directory(&mut d, dir::BOUND_IMPORT, 0x1F0, 0x10);
        directory(&mut d, dir::IAT, 0x1140, 16);
        let sec = opt + 240;
        d[sec..sec + 5].copy_from_slice(b".data");
        p32(&mut d, sec + 8, 0x400); // virtual size
        p32(&mut d, sec + 12, 0x1000);
        p32(&mut d, sec + 16, 0x400); // raw size
        p32(&mut d, sec + 20, 0x200); // raw offset
        p32(&mut d, sec + 36, 0xE000_0060); // code, data, read/write/execute

        // One import descriptor, then the null one.
        p32(&mut d, at(0x1000), 0x1100); // name table
        p32(&mut d, at(0x100C), 0x1180); // DLL name
        p32(&mut d, at(0x1010), 0x1140); // IAT
        p64(&mut d, at(0x1100), 0x1190);
        p64(&mut d, at(0x1140), 0x1190);
        d[at(0x1180)..at(0x1180) + 13].copy_from_slice(b"kernel32.dll\0");
        d[at(0x1192)..at(0x1192) + 6].copy_from_slice(b"Sleep\0");
        // TLS directory: raw data, index slot, callback array.
        p64(&mut d, at(0x1200), BASE + 0x1280);
        p64(&mut d, at(0x1208), BASE + 0x1288);
        p64(&mut d, at(0x1210), BASE + 0x1290);
        p64(&mut d, at(0x1218), BASE + 0x1240);
        p64(&mut d, at(0x1240), BASE + 0x1300);
        d[at(0x1300)] = 0xC3; // ret
        // One relocation block for page 0x1000.
        p32(&mut d, at(0x1380), 0x1000);
        p32(&mut d, at(0x1384), 12);
        p16(&mut d, at(0x1388), (10 << 12) | 0x210); // index slot pointer
        p16(&mut d, at(0x138A), (10 << 12) | 0x218); // callback pointer
        d
    }

    #[test]
    fn patches_a_synthetic_title_on_any_host() {
        let original = synthetic_exe();
        let prepared = prepare(&original).unwrap();
        let info = &prepared.info;
        assert!(!info.is_dll(), "info describes the original");
        assert_eq!(info.entry_rva, 0x1300);
        assert_eq!(info.imports.len(), 1);
        assert_eq!(info.imports[0].dll, "kernel32.dll");
        assert_eq!(info.imports[0].entries[0].iat_slot_rva, 0x1140);
        assert_eq!(info.tls.as_ref().unwrap().callback_rvas, [0x1300]);

        let bytes = &prepared.bytes;
        assert_eq!(bytes.len(), original.len());
        let patched = ImageInfo::parse(bytes).unwrap();
        assert!(patched.is_dll());
        assert_eq!(patched.entry_rva, 0);
        assert!(patched.imports.is_empty());
        let opt = layout::optional_header_offset(bytes).unwrap();
        assert_eq!(layout::read_u32(bytes, opt + layout::OPT_CHECKSUM), Some(0));
        assert_eq!(
            patched.dll_characteristics,
            dll_flags::DYNAMIC_BASE | dll_flags::NX_COMPAT
        );
        // The import directory is just the null descriptor after ours.
        let import = layout::data_directory_offset(bytes, dir::IMPORT).unwrap();
        assert_eq!(layout::read_u32(bytes, import), Some(0x1014));
        assert_eq!(
            layout::read_u32(bytes, import + 4),
            Some(IMPORT_DESCRIPTOR_SIZE)
        );
        let bound = layout::data_directory_offset(bytes, dir::BOUND_IMPORT).unwrap();
        assert_eq!(bytes[bound..bound + 8], [0; 8]);
        // The IAT and its slot are left for us to bind.
        let iat = layout::data_directory_offset(bytes, dir::IAT).unwrap();
        assert_eq!(bytes[iat..iat + 8], original[iat..iat + 8]);
        assert_eq!(
            bytes[at(0x1140)..at(0x1148)],
            original[at(0x1140)..at(0x1148)]
        );
        // The callback pointer is cleared, and so is only its relocation.
        let tls = patched.tls.as_ref().unwrap();
        assert_eq!(tls.callbacks_va, 0);
        assert!(tls.callback_rvas.is_empty());
        assert_eq!(dir64_relocations(bytes, &patched), [(at(0x1388), 0x1210)]);
        assert_eq!(bytes[at(0x138A)..at(0x138C)], [0, 0]);
        // Nothing else changed: the callback array itself is intact.
        assert_eq!(
            bytes[at(0x1240)..at(0x1248)],
            original[at(0x1240)..at(0x1248)]
        );
    }

    #[test]
    fn a_title_without_tls_callbacks_or_imports_keeps_its_relocations() {
        let mut original = synthetic_exe();
        write_u64(&mut original, at(0x1218), 0).unwrap();
        let import = layout::data_directory_offset(&original, dir::IMPORT).unwrap();
        write(&mut original, import, &[0; 8]).unwrap();
        let prepared = prepare(&original).unwrap();
        assert!(prepared.info.imports.is_empty());
        assert_eq!(prepared.info.tls.as_ref().unwrap().callbacks_va, 0);
        // An absent import directory stays absent.
        assert_eq!(prepared.bytes[import..import + 8], [0; 8]);
        assert_eq!(dir64_relocations(&prepared.bytes, &prepared.info).len(), 2);
    }

    #[test]
    fn import_sizes_that_miss_or_overshoot_the_terminator_are_normalised() {
        // Some linkers count only the real descriptors; others pad. The
        // parser walks to the null descriptor either way, so the patched
        // directory is the same: that terminator, one descriptor long.
        let original = synthetic_exe();
        let import = layout::data_directory_offset(&original, dir::IMPORT).unwrap();
        let expected = prepare(&original).unwrap().bytes;
        for size in [0, IMPORT_DESCRIPTOR_SIZE, 3 * IMPORT_DESCRIPTOR_SIZE, 0x100] {
            let mut sized = original.clone();
            write_u32(&mut sized, import + 4, size).unwrap();
            let prepared = prepare(&sized).unwrap();
            assert_eq!(prepared.info.import_count(), 1, "size {size:#x}");
            assert_eq!(
                prepared.info.imports[0].entries[0].iat_slot_rva, 0x1140,
                "size {size:#x}"
            );
            assert_eq!(prepared.bytes, expected, "size {size:#x}");
            // Different originals, different prepared files.
            assert_ne!(
                prepared.file_name(),
                prepare(&original).unwrap().file_name()
            );
        }
        // The bound IAT slot survives whatever the recorded size.
        assert_eq!(
            expected[at(0x1140)..at(0x1148)],
            original[at(0x1140)..at(0x1148)]
        );
    }

    #[test]
    fn a_tls_directory_outside_the_file_is_refused() {
        let mut original = synthetic_exe();
        // Past the raw data but inside the section's 0x400: not file-backed
        // once the raw size shrinks to the first 0x200 bytes.
        let sec = 0x58 + 240;
        write_u32(&mut original, sec + 16, 0x200).unwrap();
        write_u32(&mut original, sec + 8, 0x200).unwrap();
        assert!(matches!(
            prepare(&original),
            Err(PrepareError::Pe(PeError::Malformed(_)))
        ));
    }

    #[test]
    fn file_name_follows_the_original_bytes() {
        let original = synthetic_exe();
        let a = prepare(&original).unwrap();
        assert_eq!(a.file_name(), prepare(&original).unwrap().file_name());
        assert!(a.file_name().starts_with("victory-image-"));
        assert!(a.file_name().ends_with(".dll"));
        let mut rebuilt = original.clone();
        rebuilt[at(0x1301)] = 0xCC;
        assert_ne!(a.file_name(), prepare(&rebuilt).unwrap().file_name());
    }

    #[test]
    fn non_pe_input_is_a_pe_error() {
        assert!(matches!(prepare(b""), Err(PrepareError::Pe(_))));
        assert!(matches!(
            prepare(b"\x7fELF\x02\x01\x01\0"),
            Err(PrepareError::Pe(_))
        ));
        let mut wrong_machine = synthetic_exe();
        write_u16(&mut wrong_machine, 0x44, 0xAA64).unwrap(); // ARM64
        assert!(matches!(
            prepare(&wrong_machine),
            Err(PrepareError::Pe(PeError::Unsupported(_)))
        ));
    }
}
