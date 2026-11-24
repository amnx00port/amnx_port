//! `prepare` on a PE32+ executable built byte by byte here, so the patching
//! is covered on every host. The unit tests in `src/prepare.rs` use the test
//! binary itself, which is a PE image only on Windows.

use victory_loader::prepare;
use victory_pe::ImageInfo;
use victory_pe::layout::{self, dir, dll_flags, file_flags};

const IMAGE_BASE: u64 = 0x1_4000_0000;
const SECTION_RVA: u32 = 0x1000;
const SECTION_FILE: usize = 0x200;
const SECTION_SIZE: usize = 0x400;

// RVAs inside the one section.
const IMPORTS: u32 = 0x1000; // one descriptor, then the null terminator
const INT: u32 = 0x1040;
const IAT: u32 = 0x1060;
const HINT_NAME: u32 = 0x1080;
const DLL_NAME: u32 = 0x10A0;
const TLS: u32 = 0x1100;
const TLS_INDEX: u32 = 0x1180;
const CALLBACKS: u32 = 0x1140;
const CODE: u32 = 0x1200;
const RELOCS: u32 = 0x1300;

const NT: usize = 0x40;
const FILE_HEADER: usize = NT + 4;
const OPT: usize = FILE_HEADER + 20;
const SECTIONS: usize = OPT + 240;

const CHECKSUM: u32 = 0x0001_2345;
const DLL_CHARACTERISTICS: u16 = dll_flags::HIGH_ENTROPY_VA
    | dll_flags::DYNAMIC_BASE
    | dll_flags::NX_COMPAT
    | dll_flags::FORCE_INTEGRITY;
/// The bound-import directory as an old linker might leave it.
const BOUND: [u8; 8] = [0xF0, 0x01, 0, 0, 0x10, 0, 0, 0];

fn put(b: &mut [u8], at: usize, v: &[u8]) {
    b[at..at + v.len()].copy_from_slice(v);
}
fn put16(b: &mut [u8], at: usize, v: u16) {
    put(b, at, &v.to_le_bytes());
}
fn put32(b: &mut [u8], at: usize, v: u32) {
    put(b, at, &v.to_le_bytes());
}
fn put64(b: &mut [u8], at: usize, v: u64) {
    put(b, at, &v.to_le_bytes());
}
fn file(rva: u32) -> usize {
    SECTION_FILE + (rva - SECTION_RVA) as usize
}
fn set_dir(b: &mut [u8], index: usize, rva: u32, size: u32) {
    let at = OPT + layout::OPT_DATA_DIRECTORY + index * layout::DATA_DIRECTORY_SIZE;
    put32(b, at, rva);
    put32(b, at + 4, size);
}

/// An executable with one import module (`ExitProcess` by name, ordinal 5),
/// one TLS callback and DIR64 relocations for the TLS directory's pointers.
fn image() -> Vec<u8> {
    let mut b = vec![0u8; SECTION_FILE + SECTION_SIZE];
    put(&mut b, 0, b"MZ");
    put32(&mut b, layout::DOS_E_LFANEW, NT as u32);
    put(&mut b, NT, b"PE\0\0");

    put16(&mut b, FILE_HEADER, 0x8664); // AMD64
    put16(&mut b, FILE_HEADER + 2, 1); // sections
    put32(&mut b, FILE_HEADER + 4, 0x6000_0000); // timestamp
    put16(&mut b, FILE_HEADER + 16, 240); // optional header size
    put16(&mut b, FILE_HEADER + 18, 0x0022); // executable, large address aware

    put16(&mut b, OPT, 0x20B); // PE32+
    put32(&mut b, OPT + layout::OPT_ADDRESS_OF_ENTRY_POINT, CODE);
    put64(&mut b, OPT + 24, IMAGE_BASE);
    put32(&mut b, OPT + 32, 0x1000); // section alignment
    put32(&mut b, OPT + 36, 0x200); // file alignment
    put16(&mut b, OPT + 40, 6); // OS version
    put16(&mut b, OPT + 48, 6); // subsystem version
    put32(&mut b, OPT + 56, 0x2000); // size of image
    put32(&mut b, OPT + 60, SECTION_FILE as u32); // size of headers
    put32(&mut b, OPT + layout::OPT_CHECKSUM, CHECKSUM);
    put16(&mut b, OPT + 68, layout::subsystem::WINDOWS_CUI);
    put16(
        &mut b,
        OPT + layout::OPT_DLL_CHARACTERISTICS,
        DLL_CHARACTERISTICS,
    );
    put64(&mut b, OPT + 72, 0x10_0000); // stack reserve
    put64(&mut b, OPT + 80, 0x1000); // stack commit
    put32(&mut b, OPT + layout::OPT_NUMBER_OF_RVA_AND_SIZES, 16);
    set_dir(&mut b, dir::IMPORT, IMPORTS, 2 * 20);
    set_dir(&mut b, dir::BASERELOC, RELOCS, 16);
    set_dir(&mut b, 9, TLS, 40); // TLS
    put(
        &mut b,
        OPT + layout::OPT_DATA_DIRECTORY + dir::BOUND_IMPORT * layout::DATA_DIRECTORY_SIZE,
        &BOUND,
    );
    set_dir(&mut b, dir::IAT, IAT, 24);

    put(&mut b, SECTIONS, b".data\0\0\0");
    put32(&mut b, SECTIONS + 8, SECTION_SIZE as u32); // virtual size
    put32(&mut b, SECTIONS + 12, SECTION_RVA);
    put32(&mut b, SECTIONS + 16, SECTION_SIZE as u32); // raw size
    put32(&mut b, SECTIONS + 20, SECTION_FILE as u32);
    put32(&mut b, SECTIONS + 36, 0xE000_0060); // code, data, rwx

    // Import descriptor: name table, DLL name, IAT. The terminator is zeros.
    put32(&mut b, file(IMPORTS), INT);
    put32(&mut b, file(IMPORTS) + 12, DLL_NAME);
    put32(&mut b, file(IMPORTS) + 16, IAT);
    for table in [INT, IAT] {
        put64(&mut b, file(table), u64::from(HINT_NAME));
        put64(&mut b, file(table) + 8, 0x8000_0000_0000_0005);
    }
    put16(&mut b, file(HINT_NAME), 7);
    put(&mut b, file(HINT_NAME) + 2, b"ExitProcess\0");
    put(&mut b, file(DLL_NAME), b"KERNEL32.dll\0");

    // TLS directory: no template, an index slot and one callback.
    put64(&mut b, file(TLS) + 16, IMAGE_BASE + u64::from(TLS_INDEX));
    put64(
        &mut b,
        file(TLS) + layout::TLS64_ADDRESS_OF_CALLBACKS,
        IMAGE_BASE + u64::from(CALLBACKS),
    );
    put64(&mut b, file(CALLBACKS), IMAGE_BASE + u64::from(CODE));
    b[file(CODE)] = 0xC3; // ret

    // One block of base relocations for the page at SECTION_RVA.
    put32(&mut b, file(RELOCS), SECTION_RVA);
    put32(&mut b, file(RELOCS) + 4, 16);
    for (i, field) in [TLS + 16, callbacks_field(), CALLBACKS]
        .into_iter()
        .enumerate()
    {
        put16(&mut b, file(RELOCS) + 8 + i * 2, dir64(field));
    }
    b
}

fn callbacks_field() -> u32 {
    TLS + layout::TLS64_ADDRESS_OF_CALLBACKS as u32
}
fn dir64(rva: u32) -> u16 {
    (10 << 12) | (rva - SECTION_RVA) as u16
}
/// File offset of the relocation entry for the TLS callbacks pointer.
fn callbacks_reloc() -> usize {
    file(RELOCS) + 8 + 2
}

fn dir_offset(index: usize) -> usize {
    OPT + layout::OPT_DATA_DIRECTORY + index * layout::DATA_DIRECTORY_SIZE
}

#[test]
fn the_synthetic_image_parses_as_built() {
    let info = ImageInfo::parse(&image()).unwrap();
    assert!(!info.is_dll());
    assert_eq!(info.entry_rva, CODE);
    assert_eq!(info.import_count(), 2);
    assert_eq!(info.imports[0].dll, "KERNEL32.dll");
    let tls = info.tls.as_ref().unwrap();
    assert_eq!(tls.directory_rva, TLS);
    assert_eq!(tls.callback_rvas, [CODE]);
    assert!(info.has_relocations);
}

#[test]
fn prepare_maps_an_inert_dll_and_keeps_what_the_title_expects() {
    let original = image();
    let prepared = prepare(&original).unwrap();

    // `info` describes the original image.
    assert!(!prepared.info.is_dll());
    assert_eq!(prepared.info.entry_rva, CODE);
    assert_eq!(prepared.info.import_count(), 2);
    let slots: Vec<u32> = prepared.info.imports[0]
        .entries
        .iter()
        .map(|e| e.iat_slot_rva)
        .collect();
    assert_eq!(slots, [IAT, IAT + 8]);
    assert_eq!(prepared.info.tls.as_ref().unwrap().callback_rvas, [CODE]);
    assert_eq!(prepared.original_hash, blake3::hash(&original));

    let patched = ImageInfo::parse(&prepared.bytes).unwrap();
    assert!(patched.is_dll());
    assert_eq!(
        patched.characteristics,
        prepared.info.characteristics | file_flags::DLL
    );
    assert_eq!(patched.entry_rva, 0);
    assert!(patched.imports.is_empty());
    assert_eq!(
        patched.dll_characteristics,
        DLL_CHARACTERISTICS & !dll_flags::FORCE_INTEGRITY
    );
    assert_eq!(
        layout::read_u32(&prepared.bytes, OPT + layout::OPT_CHECKSUM),
        Some(0)
    );
    let tls = patched.tls.as_ref().unwrap();
    assert_eq!(tls.directory_rva, TLS);
    assert_eq!(tls.callbacks_va, 0);
    assert!(tls.callback_rvas.is_empty());

    // The import directory points at the empty terminator.
    let import = dir_offset(dir::IMPORT);
    assert_eq!(
        layout::read_u32(&prepared.bytes, import),
        Some(IMPORTS + 20)
    );
    assert_eq!(layout::read_u32(&prepared.bytes, import + 4), Some(20));
    assert_eq!(&prepared.bytes[file(IMPORTS + 20)..][..20], &[0; 20]);

    // Only the callbacks pointer's relocation became a no-op.
    assert_eq!(&prepared.bytes[callbacks_reloc()..][..2], &[0, 0]);
    assert_ne!(&original[callbacks_reloc()..][..2], &[0, 0]);

    // Nothing else changed: the IAT, its directory, the name tables, the
    // callback array and the other relocations are as the title left them.
    assert_eq!(prepared.bytes.len(), original.len());
    let expected = expected_changes();
    let differing: Vec<usize> = (0..original.len())
        .filter(|&i| original[i] != prepared.bytes[i])
        .collect();
    assert!(
        differing
            .iter()
            .all(|i| expected.iter().any(|r| r.contains(i))),
        "unexpected bytes changed: {:#x?}",
        differing
            .iter()
            .filter(|i| !expected.iter().any(|r| r.contains(i)))
            .collect::<Vec<_>>()
    );
}

/// Byte ranges `prepare` may write.
fn expected_changes() -> Vec<std::ops::Range<usize>> {
    let field = |at: usize, len: usize| at..at + len;
    vec![
        field(FILE_HEADER + layout::FILE_HEADER_CHARACTERISTICS, 2),
        field(OPT + layout::OPT_ADDRESS_OF_ENTRY_POINT, 4),
        field(OPT + layout::OPT_CHECKSUM, 4),
        field(OPT + layout::OPT_DLL_CHARACTERISTICS, 2),
        field(dir_offset(dir::IMPORT), layout::DATA_DIRECTORY_SIZE),
        field(dir_offset(dir::BOUND_IMPORT), layout::DATA_DIRECTORY_SIZE),
        field(file(callbacks_field()), 8),
        field(callbacks_reloc(), 2),
    ]
}

#[test]
fn prepare_clears_the_bound_import_directory() {
    let original = image();
    let at = dir_offset(dir::BOUND_IMPORT);
    assert_eq!(&original[at..][..8], &BOUND);
    let prepared = prepare(&original).unwrap();
    assert_eq!(&prepared.bytes[at..][..8], &[0; 8]);
}

#[test]
fn import_size_that_excludes_the_terminator_still_finds_it() {
    // Some linkers count the descriptors for real modules only.
    let mut original = image();
    put32(&mut original, dir_offset(dir::IMPORT) + 4, 20);
    let prepared = prepare(&original).unwrap();
    assert_eq!(prepared.info.import_count(), 2);
    let import = dir_offset(dir::IMPORT);
    assert_eq!(
        layout::read_u32(&prepared.bytes, import),
        Some(IMPORTS + 20)
    );
    assert_eq!(layout::read_u32(&prepared.bytes, import + 4), Some(20));
    assert!(
        ImageInfo::parse(&prepared.bytes)
            .unwrap()
            .imports
            .is_empty()
    );
}

#[test]
fn an_image_without_imports_or_tls_only_changes_its_headers() {
    let mut original = image();
    set_dir(&mut original, dir::IMPORT, 0, 0);
    set_dir(&mut original, dir::IAT, 0, 0);
    set_dir(&mut original, 9, 0, 0);
    let prepared = prepare(&original).unwrap();
    assert!(prepared.info.imports.is_empty());
    assert!(prepared.info.tls.is_none());
    let import = dir_offset(dir::IMPORT);
    assert_eq!(&prepared.bytes[import..][..8], &[0; 8]);
    // The section, relocations included, is untouched.
    assert_eq!(&prepared.bytes[SECTION_FILE..], &original[SECTION_FILE..]);
    assert!(ImageInfo::parse(&prepared.bytes).unwrap().is_dll());
}

#[test]
fn file_name_names_the_original_build() {
    let original = image();
    let a = prepare(&original).unwrap();
    let b = prepare(&original).unwrap();
    assert_eq!(a.file_name(), b.file_name());
    let hex = &blake3::hash(&original).to_hex()[..16];
    assert_eq!(a.file_name(), format!("victory-image-{hex}.dll"));

    let mut other = original;
    put32(&mut other, FILE_HEADER + 4, 0x6000_0001); // another timestamp
    assert_ne!(prepare(&other).unwrap().file_name(), a.file_name());
}

#[test]
fn what_is_not_a_pe32_plus_image_is_an_error() {
    assert!(prepare(&[]).is_err());
    assert!(prepare(b"MZ").is_err());
    let original = image();
    for len in [0x40, 0x100, SECTIONS + 20] {
        assert!(prepare(&original[..len]).is_err(), "truncated to {len:#x}");
    }
    let mut x86 = original;
    put16(&mut x86, FILE_HEADER, 0x014C);
    assert!(prepare(&x86).is_err());
}
