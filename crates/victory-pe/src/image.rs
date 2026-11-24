use object::LittleEndian as LE;
use object::Object;
use object::pe::{self, ImageNtHeaders64};
use object::read::pe::{
    ExportTarget, ImageNtHeaders, ImageOptionalHeader, Import, ImportTable, ImportThunkList,
    PeFile64,
};

use crate::layout::{self, dir};
use crate::{MACHINE_AMD64, PeError};

/// Everything we need to know about a PE32+ image, extracted in one pass.
#[derive(Debug, Clone)]
pub struct ImageInfo {
    pub machine: u16,
    /// `IMAGE_FILE_HEADER::Characteristics`.
    pub characteristics: u16,
    pub timestamp: u32,
    pub subsystem: u16,
    /// `IMAGE_OPTIONAL_HEADER64::DllCharacteristics`.
    pub dll_characteristics: u16,
    pub image_base: u64,
    pub size_of_image: u32,
    pub entry_rva: u32,
    pub stack_reserve: u64,
    pub stack_commit: u64,
    pub sections: Vec<SectionInfo>,
    pub imports: Vec<ImportModule>,
    pub delay_imports: Vec<ImportModule>,
    pub exports: Vec<ExportInfo>,
    pub tls: Option<TlsInfo>,
    pub has_relocations: bool,
    pub has_exception_table: bool,
    pub has_load_config: bool,
    pub codeview: Option<CodeViewInfo>,
}

#[derive(Debug, Clone)]
pub struct SectionInfo {
    pub name: String,
    pub rva: u32,
    pub virtual_size: u32,
    pub file_offset: u32,
    pub file_size: u32,
    pub characteristics: u32,
}

/// One imported DLL and the functions taken from it.
#[derive(Debug, Clone)]
pub struct ImportModule {
    /// DLL name exactly as written in the image (case preserved).
    pub dll: String,
    pub entries: Vec<ImportEntry>,
}

#[derive(Debug, Clone)]
pub struct ImportEntry {
    pub symbol: ImportSymbol,
    /// RVA of the 8-byte IAT slot the loader must fill in.
    pub iat_slot_rva: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ImportSymbol {
    Name { hint: u16, name: String },
    Ordinal(u16),
}

impl std::fmt::Display for ImportSymbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImportSymbol::Name { name, .. } => f.write_str(name),
            ImportSymbol::Ordinal(o) => write!(f, "#{o}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExportInfo {
    pub name: Option<String>,
    pub ordinal: u16,
    /// `None` for forwarded exports.
    pub rva: Option<u32>,
    /// `"dll.name"` or `"dll.#ordinal"` for forwarded exports.
    pub forward: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TlsInfo {
    /// RVA of `IMAGE_TLS_DIRECTORY64`.
    pub directory_rva: u32,
    /// VA (at the preferred base) of the callback array, 0 if none.
    pub callbacks_va: u64,
    /// Callback RVAs, in call order.
    pub callback_rvas: Vec<u32>,
}

/// CodeView (RSDS) record: identifies the exact build that produced an image.
#[derive(Debug, Clone)]
pub struct CodeViewInfo {
    pub pdb_path: String,
    pub guid: [u8; 16],
    pub age: u32,
}

impl CodeViewInfo {
    /// The key symbol servers use: GUID (Windows byte order) + age.
    pub fn symbol_key(&self) -> String {
        let g = &self.guid;
        let d1 = u32::from_le_bytes([g[0], g[1], g[2], g[3]]);
        let d2 = u16::from_le_bytes([g[4], g[5]]);
        let d3 = u16::from_le_bytes([g[6], g[7]]);
        let mut s = format!("{d1:08X}{d2:04X}{d3:04X}");
        for b in &g[8..] {
            s.push_str(&format!("{b:02X}"));
        }
        s.push_str(&format!("{:X}", self.age));
        s
    }
}

/// Upper bound on TLS callbacks we are willing to walk (defends against garbage).
const MAX_TLS_CALLBACKS: usize = 1024;

impl ImageInfo {
    /// Parses a complete PE32+ x86-64 image from its file bytes.
    pub fn parse(data: &[u8]) -> Result<Self, PeError> {
        let kind = object::FileKind::parse(data)?;
        if kind != object::FileKind::Pe64 {
            return Err(PeError::Unsupported(format!(
                "expected PE32+, found {kind:?}"
            )));
        }
        let file = PeFile64::parse(data)?;
        let nt = file.nt_headers();
        let fh = nt.file_header();
        let opt = nt.optional_header();

        let machine = fh.machine.get(LE).0;
        if machine != MACHINE_AMD64 {
            return Err(PeError::Unsupported(format!(
                "machine 0x{machine:04x} is not AMD64"
            )));
        }

        let sections = file
            .section_table()
            .iter()
            .map(|s| SectionInfo {
                name: String::from_utf8_lossy(s.raw_name()).into_owned(),
                rva: s.virtual_address.get(LE),
                virtual_size: s.virtual_size.get(LE),
                file_offset: s.pointer_to_raw_data.get(LE),
                file_size: s.size_of_raw_data.get(LE),
                characteristics: s.characteristics.get(LE).0,
            })
            .collect();

        let dir_present = |index: usize| {
            file.data_directory(index)
                .is_some_and(|d| d.virtual_address.get(LE) != 0 && d.size.get(LE) != 0)
        };

        let imports = match file.import_table()? {
            Some(table) => parse_imports(&table, opt.size_of_image())?,
            None => Vec::new(),
        };
        let delay_imports = parse_delay_imports(&file, opt.size_of_image())?;
        let exports = parse_exports(&file)?;
        let tls = parse_tls(&file, data, opt.image_base())?;
        let codeview = file.pdb_info()?.map(|cv| CodeViewInfo {
            pdb_path: String::from_utf8_lossy(cv.path()).into_owned(),
            guid: cv.guid(),
            age: cv.age(),
        });

        Ok(ImageInfo {
            machine,
            characteristics: fh.characteristics.get(LE).0,
            timestamp: fh.time_date_stamp.get(LE),
            subsystem: opt.subsystem().0,
            dll_characteristics: opt.dll_characteristics().0,
            image_base: opt.image_base(),
            size_of_image: opt.size_of_image(),
            entry_rva: opt.address_of_entry_point(),
            stack_reserve: opt.size_of_stack_reserve(),
            stack_commit: opt.size_of_stack_commit(),
            sections,
            imports,
            delay_imports,
            exports,
            tls,
            has_relocations: dir_present(dir::BASERELOC)
                && !has_flag(
                    fh.characteristics.get(LE).0,
                    layout::file_flags::RELOCS_STRIPPED,
                ),
            has_exception_table: dir_present(dir::EXCEPTION),
            has_load_config: dir_present(dir::LOAD_CONFIG),
            codeview,
        })
    }

    pub fn is_dll(&self) -> bool {
        has_flag(self.characteristics, layout::file_flags::DLL)
    }

    /// Maps an RVA to an offset in the file, if it is backed by file data.
    pub fn rva_to_file_offset(&self, rva: u32) -> Option<u32> {
        self.sections.iter().find_map(|s| {
            let delta = rva.checked_sub(s.rva)?;
            // Raw data may be padded past the virtual size; that padding is
            // not part of the section in memory.
            let backed = match s.virtual_size {
                0 => s.file_size,
                vs => vs.min(s.file_size),
            };
            // Header values are untrusted: an offset near 4 GiB overflows.
            (delta < backed).then(|| s.file_offset.checked_add(delta))?
        })
    }

    pub fn import_count(&self) -> usize {
        self.imports.iter().map(|m| m.entries.len()).sum()
    }

    pub fn subsystem_name(&self) -> &'static str {
        use layout::subsystem::*;
        match self.subsystem {
            WINDOWS_GUI => "windows-gui",
            WINDOWS_CUI => "windows-cui",
            XBOX => "xbox",
            XBOX_CODE_CATALOG => "xbox-code-catalog",
            _ => "other",
        }
    }
}

fn has_flag(value: u16, flag: u16) -> bool {
    value & flag != 0
}

fn utf8(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn convert_import(import: Import<'_>) -> ImportSymbol {
    match import {
        Import::Ordinal(o) => ImportSymbol::Ordinal(o),
        Import::Name(hint, name) => ImportSymbol::Name {
            hint,
            name: utf8(name),
        },
    }
}

fn parse_imports(
    table: &ImportTable<'_>,
    size_of_image: u32,
) -> Result<Vec<ImportModule>, PeError> {
    let mut modules = Vec::new();
    let mut descriptors = table.descriptors()?;
    while let Some(desc) = descriptors.next()? {
        let iat_rva = desc.first_thunk.get(LE);
        // Prefer the unbound name table; fall back to the IAT for old linkers.
        let int_rva = match desc.original_first_thunk.get(LE) {
            0 => iat_rva,
            rva => rva,
        };
        modules.push(ImportModule {
            dll: utf8(table.name(desc.name.get(LE))?),
            entries: entries(table.thunks(int_rva)?, iat_rva, size_of_image, |t| {
                table.import::<ImageNtHeaders64>(t)
            })?,
        });
    }
    Ok(modules)
}

fn parse_delay_imports(
    file: &PeFile64<'_>,
    size_of_image: u32,
) -> Result<Vec<ImportModule>, PeError> {
    let Some(table) = file.delay_load_import_table()? else {
        return Ok(Vec::new());
    };
    let mut modules = Vec::new();
    let mut descriptors = table.descriptors()?;
    while let Some(desc) = descriptors.next()? {
        modules.push(ImportModule {
            dll: utf8(table.name(desc.dll_name_rva.get(LE))?),
            entries: entries(
                table.thunks(desc.import_name_table_rva.get(LE))?,
                desc.import_address_table_rva.get(LE),
                size_of_image,
                |t| table.import::<ImageNtHeaders64>(t),
            )?,
        });
    }
    Ok(modules)
}

/// Pairs each name-table entry with its IAT slot. The IAT RVA is not checked
/// by `object`, so every slot is checked to lie inside the image here.
fn entries<'data>(
    mut thunks: ImportThunkList<'data>,
    iat_rva: u32,
    size_of_image: u32,
    import: impl Fn(pe::ImageThunkData64) -> object::read::Result<Import<'data>>,
) -> Result<Vec<ImportEntry>, PeError> {
    let mut entries = Vec::new();
    while let Some(thunk) = thunks.next::<ImageNtHeaders64>()? {
        let iat_slot_rva = u32::try_from(entries.len() * 8)
            .ok()
            .and_then(|offset| iat_rva.checked_add(offset))
            .filter(|&rva| rva.checked_add(8).is_some_and(|end| end <= size_of_image))
            .ok_or_else(|| {
                PeError::Malformed(format!("IAT at {iat_rva:#x} runs outside the image"))
            })?;
        entries.push(ImportEntry {
            symbol: convert_import(import(thunk)?),
            iat_slot_rva,
        });
    }
    Ok(entries)
}

fn parse_exports(file: &PeFile64<'_>) -> Result<Vec<ExportInfo>, PeError> {
    let Some(table) = file.export_table()? else {
        return Ok(Vec::new());
    };
    Ok(table
        .exports()?
        .into_iter()
        // An address of 0 marks an unused slot in the address table (a gap
        // in the ordinals), not an export at the image base.
        .filter(|e| !matches!(e.target, ExportTarget::Address(0)))
        .map(|e| {
            let (rva, forward) = match e.target {
                ExportTarget::Address(rva) => (Some(rva), None),
                ExportTarget::ForwardByName(d, n) => {
                    (None, Some(format!("{}.{}", utf8(d), utf8(n))))
                }
                ExportTarget::ForwardByOrdinal(d, o) => {
                    (None, Some(format!("{}.#{}", utf8(d), o.0)))
                }
            };
            ExportInfo {
                name: e.name.map(utf8),
                ordinal: e.ordinal.0,
                rva,
                forward,
            }
        })
        .collect())
}

fn parse_tls(
    file: &PeFile64<'_>,
    data: &[u8],
    image_base: u64,
) -> Result<Option<TlsInfo>, PeError> {
    let Some(dd) = file.data_directory(pe::IMAGE_DIRECTORY_ENTRY_TLS) else {
        return Ok(None);
    };
    let directory_rva = dd.virtual_address.get(LE);
    if directory_rva == 0 {
        return Ok(None);
    }
    let sections = file.section_table();
    let tls = sections
        .pe_data_at(data, directory_rva)
        .ok_or_else(|| PeError::Malformed("TLS directory outside of sections".into()))?;
    let callbacks_va = layout::read_u64(tls, layout::TLS64_ADDRESS_OF_CALLBACKS)
        .ok_or_else(|| PeError::Malformed("truncated TLS directory".into()))?;

    let mut callback_rvas = Vec::new();
    if callbacks_va != 0 {
        let array_rva = va_to_rva(callbacks_va, image_base)?;
        let array = sections
            .pe_data_at(data, array_rva)
            .ok_or_else(|| PeError::Malformed("TLS callback array outside of sections".into()))?;
        for slot in array.as_chunks::<8>().0.iter().take(MAX_TLS_CALLBACKS) {
            let va = u64::from_le_bytes(*slot);
            if va == 0 {
                break;
            }
            callback_rvas.push(va_to_rva(va, image_base)?);
        }
    }
    Ok(Some(TlsInfo {
        directory_rva,
        callbacks_va,
        callback_rvas,
    }))
}

fn va_to_rva(va: u64, image_base: u64) -> Result<u32, PeError> {
    va.checked_sub(image_base)
        .and_then(|rva| u32::try_from(rva).ok())
        .ok_or_else(|| PeError::Malformed(format!("VA 0x{va:x} is outside the image")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The test binary itself is a real MSVC-linked PE32+ image.
    fn own_image() -> Vec<u8> {
        std::fs::read(std::env::current_exe().unwrap()).unwrap()
    }

    #[test]
    #[cfg(windows)] // reads its own PE test binary / Windows path rules
    fn parses_own_test_binary() {
        let info = ImageInfo::parse(&own_image()).unwrap();
        assert_eq!(info.machine, MACHINE_AMD64);
        assert!(!info.is_dll());
        assert!(info.entry_rva != 0);
        assert!(info.has_exception_table, "x64 images always carry .pdata");
        assert!(info.sections.iter().any(|s| s.name.starts_with(".text")));
        let kernel32 = info
            .imports
            .iter()
            .find(|m| m.dll.eq_ignore_ascii_case("kernel32.dll"))
            .expect("Rust std imports kernel32");
        assert!(kernel32.entries.iter().all(|e| e.iat_slot_rva != 0));
        // Rust std uses static TLS, so a TLS directory must be present (its
        // callback array may legitimately be empty).
        assert!(info.tls.is_some());
    }

    #[test]
    #[cfg(windows)] // reads its own PE test binary / Windows path rules
    fn rva_to_file_offset_maps_entry_point() {
        let data = own_image();
        let info = ImageInfo::parse(&data).unwrap();
        let off = info.rva_to_file_offset(info.entry_rva).unwrap() as usize;
        assert!(off < data.len());
    }

    #[test]
    #[cfg(windows)] // reads its own PE test binary / Windows path rules
    fn rva_to_file_offset_rejects_offsets_past_4_gib() {
        let mut info = ImageInfo::parse(&own_image()).unwrap();
        let rva = info.sections[0].rva + 0x20;
        assert!(info.rva_to_file_offset(rva).is_some());
        // Used to overflow (a panic in debug builds, a wrong offset in
        // release ones).
        info.sections[0].file_offset = u32::MAX - 0x10;
        assert_eq!(info.rva_to_file_offset(rva), None);
    }

    #[test]
    fn rejects_garbage_without_panicking() {
        assert!(ImageInfo::parse(b"").is_err());
        assert!(ImageInfo::parse(b"MZ not really").is_err());
        let mut truncated = own_image();
        truncated.truncate(0x200);
        assert!(ImageInfo::parse(&truncated).is_err());
    }

    #[test]
    #[cfg(windows)] // reads its own PE test binary / Windows path rules
    fn rejects_iat_outside_image() {
        let mut data = own_image();
        let dd = layout::data_directory_offset(&data, dir::IMPORT).unwrap();
        let import_rva = layout::read_u32(&data, dd).unwrap();
        let desc = ImageInfo::parse(&data)
            .unwrap()
            .rva_to_file_offset(import_rva)
            .unwrap() as usize;
        // First descriptor's FirstThunk, at the 4 GiB limit: used to overflow.
        data[desc + 16..desc + 20].copy_from_slice(&0xFFFF_FFF8u32.to_le_bytes());
        assert!(matches!(
            ImageInfo::parse(&data),
            Err(PeError::Malformed(_))
        ));
    }
    /// A minimal PE32+ image: one section at RVA 0x1000 holding `body`, with
    /// the export directory (RVA 0x1000, 0x80 bytes) pointing at it.
    fn synthetic_image(body: &[u8]) -> Vec<u8> {
        let mut d = vec![0u8; 0x200];
        d[0..2].copy_from_slice(b"MZ");
        d[0x3C..0x40].copy_from_slice(&0x40u32.to_le_bytes());
        d[0x40..0x44].copy_from_slice(b"PE\0\0");
        d[0x44..0x46].copy_from_slice(&MACHINE_AMD64.to_le_bytes());
        d[0x46..0x48].copy_from_slice(&1u16.to_le_bytes()); // sections
        d[0x54..0x56].copy_from_slice(&240u16.to_le_bytes()); // optional header size
        d[0x56..0x58].copy_from_slice(&0x2022u16.to_le_bytes()); // DLL, large address aware
        let opt = 0x58;
        d[opt..opt + 2].copy_from_slice(&0x20Bu16.to_le_bytes());
        d[opt + 24..opt + 32].copy_from_slice(&0x1_8000_0000u64.to_le_bytes());
        d[opt + 32..opt + 36].copy_from_slice(&0x1000u32.to_le_bytes());
        d[opt + 36..opt + 40].copy_from_slice(&0x200u32.to_le_bytes());
        d[opt + 56..opt + 60].copy_from_slice(&0x2000u32.to_le_bytes());
        d[opt + 60..opt + 64].copy_from_slice(&0x200u32.to_le_bytes());
        d[opt + 108..opt + 112].copy_from_slice(&16u32.to_le_bytes());
        d[opt + 112..opt + 116].copy_from_slice(&0x1000u32.to_le_bytes());
        d[opt + 116..opt + 120].copy_from_slice(&0x80u32.to_le_bytes());
        let sec = opt + 240;
        d[sec..sec + 6].copy_from_slice(b".edata");
        d[sec + 8..sec + 12].copy_from_slice(&0x200u32.to_le_bytes()); // virtual size
        d[sec + 12..sec + 16].copy_from_slice(&0x1000u32.to_le_bytes());
        d[sec + 16..sec + 20].copy_from_slice(&0x200u32.to_le_bytes()); // raw size
        d[sec + 20..sec + 24].copy_from_slice(&0x200u32.to_le_bytes()); // raw offset
        d.extend_from_slice(body);
        d.resize(0x400, 0);
        d
    }

    #[test]
    fn unused_export_slots_are_not_exports() {
        // Ordinal base 1; three address slots: "alpha", an unused hole (0),
        // and "gamma". The Windows loader treats address 0 as "not exported".
        let mut body = vec![0u8; 0x100];
        let put32 =
            |b: &mut Vec<u8>, at: usize, v: u32| b[at..at + 4].copy_from_slice(&v.to_le_bytes());
        put32(&mut body, 12, 0x1040); // name of the DLL
        put32(&mut body, 16, 1); // ordinal base
        put32(&mut body, 20, 3); // number of functions
        put32(&mut body, 24, 2); // number of names
        put32(&mut body, 28, 0x1028); // address table
        put32(&mut body, 32, 0x1034); // name pointer table
        put32(&mut body, 36, 0x103C); // name ordinal table
        put32(&mut body, 0x28, 0x1200);
        put32(&mut body, 0x2C, 0);
        put32(&mut body, 0x30, 0x1204);
        put32(&mut body, 0x34, 0x1044);
        put32(&mut body, 0x38, 0x104A);
        body[0x3C..0x3E].copy_from_slice(&0u16.to_le_bytes());
        body[0x3E..0x40].copy_from_slice(&2u16.to_le_bytes());
        body[0x40..0x44].copy_from_slice(b"t.dl");
        body[0x44..0x4A].copy_from_slice(b"alpha\0");
        body[0x4A..0x50].copy_from_slice(b"gamma\0");

        let info = ImageInfo::parse(&synthetic_image(&body)).unwrap();
        let exports: Vec<_> = info
            .exports
            .iter()
            .map(|e| (e.name.as_deref(), e.ordinal, e.rva))
            .collect();
        assert_eq!(
            exports,
            [
                (Some("alpha"), 1, Some(0x1200)),
                (Some("gamma"), 3, Some(0x1204))
            ]
        );
    }

    // The own-binary tests above only run on Windows; these cover the same
    // edges on the synthetic image, on any host.

    #[test]
    fn synthetic_image_maps_rvas_within_its_raw_data() {
        let info = ImageInfo::parse(&synthetic_image(&[0; 0x80])).unwrap();
        assert!(info.is_dll());
        assert_eq!(info.sections.len(), 1);
        assert_eq!(info.rva_to_file_offset(0x1000), Some(0x200));
        assert_eq!(info.rva_to_file_offset(0x11FF), Some(0x3FF));
        // Before the section, and past both its virtual and raw sizes.
        assert_eq!(info.rva_to_file_offset(0xFFF), None);
        assert_eq!(info.rva_to_file_offset(0x1200), None);
        assert_eq!(info.rva_to_file_offset(u32::MAX), None);
    }

    #[test]
    fn rva_mapping_stops_at_the_smaller_of_virtual_and_raw_size() {
        let mut info = ImageInfo::parse(&synthetic_image(&[0; 0x80])).unwrap();
        // Zero-fill past the raw data is in memory but not in the file.
        info.sections[0].virtual_size = 0x400;
        assert_eq!(info.rva_to_file_offset(0x11FF), Some(0x3FF));
        assert_eq!(info.rva_to_file_offset(0x1200), None);
        // Alignment padding past the virtual size is in the file only.
        info.sections[0].virtual_size = 0x100;
        assert_eq!(info.rva_to_file_offset(0x10FF), Some(0x2FF));
        assert_eq!(info.rva_to_file_offset(0x1100), None);
        // A virtual size of 0 means "the raw size".
        info.sections[0].virtual_size = 0;
        assert_eq!(info.rva_to_file_offset(0x11FF), Some(0x3FF));
    }

    #[test]
    fn synthetic_rva_mapping_rejects_offsets_past_4_gib() {
        let mut info = ImageInfo::parse(&synthetic_image(&[0; 0x80])).unwrap();
        info.sections[0].file_offset = u32::MAX - 0x10;
        assert_eq!(info.rva_to_file_offset(0x1010), Some(u32::MAX));
        // One byte further used to overflow.
        assert_eq!(info.rva_to_file_offset(0x1011), None);
    }

    #[test]
    fn synthetic_image_rejects_other_machines_and_pe32() {
        let mut arm = synthetic_image(&[0; 0x80]);
        arm[0x44..0x46].copy_from_slice(&0xAA64u16.to_le_bytes());
        assert!(matches!(
            ImageInfo::parse(&arm),
            Err(PeError::Unsupported(_))
        ));
        let mut pe32 = synthetic_image(&[0; 0x80]);
        pe32[0x58..0x5A].copy_from_slice(&0x10Bu16.to_le_bytes());
        assert!(ImageInfo::parse(&pe32).is_err());
        let mut truncated = synthetic_image(&[0; 0x80]);
        truncated.truncate(0x100);
        assert!(ImageInfo::parse(&truncated).is_err());
    }

    // The own-binary import, IAT and TLS tests only run on Windows. This
    // image carries all three by hand, so their edges run on any host.

    const SYNTHETIC_BASE: u64 = 0x1_8000_0000;

    /// The synthetic image with its export directory cleared and instead an
    /// import of `user32.dll` (`Beep` by name, then ordinal 7) and a TLS
    /// directory whose callback array holds two callbacks, a terminator and
    /// a stray entry after it. Returns the image and the file offset of RVA
    /// 0x1000, where the body starts.
    fn synthetic_image_with_imports_and_tls() -> (Vec<u8>, usize) {
        let mut b = vec![0u8; 0x200];
        let p16 =
            |b: &mut Vec<u8>, at: usize, v: u16| b[at..at + 2].copy_from_slice(&v.to_le_bytes());
        let p32 =
            |b: &mut Vec<u8>, at: usize, v: u32| b[at..at + 4].copy_from_slice(&v.to_le_bytes());
        let p64 =
            |b: &mut Vec<u8>, at: usize, v: u64| b[at..at + 8].copy_from_slice(&v.to_le_bytes());
        // One import descriptor, then the null one.
        p32(&mut b, 0x00, 0x1100); // name table
        p32(&mut b, 0x0C, 0x1180); // DLL name
        p32(&mut b, 0x10, 0x1140); // IAT
        for table in [0x100, 0x140] {
            p64(&mut b, table, 0x1190);
            p64(&mut b, table + 8, (1 << 63) | 7);
        }
        b[0x180..0x18B].copy_from_slice(b"user32.dll\0");
        p16(&mut b, 0x190, 0x2A);
        b[0x192..0x197].copy_from_slice(b"Beep\0");
        // TLS directory; its callback array is at RVA 0x1060.
        p64(
            &mut b,
            0x1C0 + layout::TLS64_ADDRESS_OF_CALLBACKS,
            SYNTHETIC_BASE + 0x1060,
        );
        p64(&mut b, 0x60, SYNTHETIC_BASE + 0x1010);
        p64(&mut b, 0x68, SYNTHETIC_BASE + 0x1020);
        p64(&mut b, 0x78, SYNTHETIC_BASE + 0x1030); // after the terminator

        let mut d = synthetic_image(&b);
        let directory = |d: &mut Vec<u8>, index: usize, rva: u32, size: u32| {
            let at = 0x58 + 112 + index * 8;
            d[at..at + 4].copy_from_slice(&rva.to_le_bytes());
            d[at + 4..at + 8].copy_from_slice(&size.to_le_bytes());
        };
        directory(&mut d, 0, 0, 0);
        directory(&mut d, dir::IMPORT, 0x1000, 40);
        directory(&mut d, pe::IMAGE_DIRECTORY_ENTRY_TLS, 0x11C0, 40);
        (d, 0x200)
    }

    fn symbols(module: &ImportModule) -> Vec<(ImportSymbol, u32)> {
        module
            .entries
            .iter()
            .map(|e| (e.symbol.clone(), e.iat_slot_rva))
            .collect()
    }

    fn beep_then_ordinal_7(first_slot: u32) -> Vec<(ImportSymbol, u32)> {
        vec![
            (
                ImportSymbol::Name {
                    hint: 0x2A,
                    name: "Beep".into(),
                },
                first_slot,
            ),
            (ImportSymbol::Ordinal(7), first_slot + 8),
        ]
    }

    #[test]
    fn synthetic_imports_pair_names_and_ordinals_with_iat_slots() {
        let (data, _) = synthetic_image_with_imports_and_tls();
        let info = ImageInfo::parse(&data).unwrap();
        assert!(info.exports.is_empty());
        assert!(info.delay_imports.is_empty());
        assert_eq!(info.imports.len(), 1);
        assert_eq!(info.imports[0].dll, "user32.dll");
        assert_eq!(symbols(&info.imports[0]), beep_then_ordinal_7(0x1140));
        assert_eq!(info.import_count(), 2);
        assert_eq!(info.imports[0].entries[1].symbol.to_string(), "#7");
    }

    #[test]
    fn without_a_name_table_imports_are_read_from_the_iat() {
        let (mut data, body) = synthetic_image_with_imports_and_tls();
        // Old linkers leave OriginalFirstThunk 0. Clear the name table too,
        // so only a read of the IAT can find the imports.
        data[body..body + 4].fill(0);
        data[body + 0x100..body + 0x110].fill(0);
        let info = ImageInfo::parse(&data).unwrap();
        assert_eq!(symbols(&info.imports[0]), beep_then_ordinal_7(0x1140));
    }

    #[test]
    fn synthetic_iat_must_fit_inside_the_image() {
        let (data, body) = synthetic_image_with_imports_and_tls();
        let with_iat = |rva: u32, entries: usize| {
            let mut d = data.clone();
            d[body + 0x10..body + 0x14].copy_from_slice(&rva.to_le_bytes());
            d[body + 0x100 + entries * 8..body + 0x110].fill(0);
            ImageInfo::parse(&d)
        };
        // The IAT RVA is not checked by `object`: only the slots it implies.
        // The last slot may end exactly at SizeOfImage (0x2000)...
        let info = with_iat(0x1FF8, 1).unwrap();
        assert_eq!(info.imports[0].entries[0].iat_slot_rva, 0x1FF8);
        // ...but a second slot from there runs past it,
        assert!(matches!(with_iat(0x1FF8, 2), Err(PeError::Malformed(_))));
        // and one at the 4 GiB limit used to overflow.
        assert!(matches!(
            with_iat(0xFFFF_FFF8, 1),
            Err(PeError::Malformed(_))
        ));
        assert!(matches!(
            with_iat(0xFFFF_FFFC, 1),
            Err(PeError::Malformed(_))
        ));
    }

    #[test]
    fn synthetic_tls_callbacks_stop_at_the_first_null() {
        let (data, _) = synthetic_image_with_imports_and_tls();
        let tls = ImageInfo::parse(&data).unwrap().tls.unwrap();
        assert_eq!(tls.directory_rva, 0x11C0);
        assert_eq!(tls.callbacks_va, SYNTHETIC_BASE + 0x1060);
        assert_eq!(tls.callback_rvas, [0x1010, 0x1020]);
    }

    #[test]
    fn synthetic_tls_without_callbacks_is_still_a_tls_directory() {
        let (mut data, body) = synthetic_image_with_imports_and_tls();
        let callbacks = body + 0x1C0 + layout::TLS64_ADDRESS_OF_CALLBACKS;
        data[callbacks..callbacks + 8].fill(0);
        let tls = ImageInfo::parse(&data).unwrap().tls.unwrap();
        assert_eq!(tls.callbacks_va, 0);
        assert!(tls.callback_rvas.is_empty());
        // And no TLS directory at all is `None`, not an error.
        let at = 0x58 + 112 + pe::IMAGE_DIRECTORY_ENTRY_TLS * 8;
        data[at..at + 8].fill(0);
        assert!(ImageInfo::parse(&data).unwrap().tls.is_none());
    }

    #[test]
    fn synthetic_tls_pointers_outside_the_image_are_refused() {
        let (data, body) = synthetic_image_with_imports_and_tls();
        let patched = |at: usize, va: u64| {
            let mut d = data.clone();
            d[body + at..body + at + 8].copy_from_slice(&va.to_le_bytes());
            ImageInfo::parse(&d)
        };
        let array = 0x1C0 + layout::TLS64_ADDRESS_OF_CALLBACKS;
        // The array below the image base, more than 4 GiB above it, and
        // inside the image but past the one section's data.
        for va in [
            SYNTHETIC_BASE - 8,
            SYNTHETIC_BASE + (1 << 32),
            SYNTHETIC_BASE + 0x1F00,
        ] {
            assert!(
                matches!(patched(array, va), Err(PeError::Malformed(_))),
                "callback array at {va:#x}"
            );
        }
        // A callback below the image base.
        assert!(matches!(
            patched(0x68, SYNTHETIC_BASE - 0x10),
            Err(PeError::Malformed(_))
        ));
        // A TLS directory RVA past the section.
        let mut d = data.clone();
        let at = 0x58 + 112 + pe::IMAGE_DIRECTORY_ENTRY_TLS * 8;
        d[at..at + 4].copy_from_slice(&0x1F00u32.to_le_bytes());
        assert!(matches!(ImageInfo::parse(&d), Err(PeError::Malformed(_))));
    }

    #[test]
    fn forwarded_exports_name_their_target_by_name_or_ordinal() {
        // Ordinal base 1; slot 1 a real export, slots 2 and 3 forwarders.
        // An address inside the export directory (0x1000..0x1080) is a
        // "dll.name" or "dll.#ordinal" string, not code.
        let mut body = vec![0u8; 0x100];
        let put32 =
            |b: &mut Vec<u8>, at: usize, v: u32| b[at..at + 4].copy_from_slice(&v.to_le_bytes());
        put32(&mut body, 12, 0x1040); // name of the DLL
        put32(&mut body, 16, 1); // ordinal base
        put32(&mut body, 20, 3); // number of functions
        put32(&mut body, 24, 1); // number of names
        put32(&mut body, 28, 0x1028); // address table
        put32(&mut body, 32, 0x1034); // name pointer table
        put32(&mut body, 36, 0x1038); // name ordinal table
        put32(&mut body, 0x28, 0x1200);
        put32(&mut body, 0x2C, 0x1060);
        put32(&mut body, 0x30, 0x1070);
        put32(&mut body, 0x34, 0x1044);
        body[0x38..0x3A].copy_from_slice(&1u16.to_le_bytes());
        body[0x40..0x44].copy_from_slice(b"t.dl");
        body[0x44..0x4A].copy_from_slice(b"Sleep\0");
        body[0x60..0x6F].copy_from_slice(b"kernel32.Sleep\0");
        body[0x70..0x7C].copy_from_slice(b"kernel32.#5\0");

        let info = ImageInfo::parse(&synthetic_image(&body)).unwrap();
        let exports: Vec<_> = info
            .exports
            .iter()
            .map(|e| (e.name.as_deref(), e.ordinal, e.rva, e.forward.as_deref()))
            .collect();
        assert_eq!(
            exports,
            [
                (None, 1, Some(0x1200), None),
                (Some("Sleep"), 2, None, Some("kernel32.Sleep")),
                (None, 3, None, Some("kernel32.#5")),
            ]
        );

        // A forwarder with nothing after the dot is refused, not exported
        // at its string's address.
        body[0x70..0x7C].fill(0);
        body[0x70..0x7A].copy_from_slice(b"kernel32.\0");
        assert!(ImageInfo::parse(&synthetic_image(&body)).is_err());
    }

    /// Writes a delay-import descriptor for `xinput.dll` at RVA 0x1080
    /// (directory 13) that shares the name table at 0x1100 and has its own
    /// IAT at `iat`.
    fn with_delay_imports(data: &[u8], body: usize, iat: u32) -> Vec<u8> {
        let mut d = data.to_vec();
        let p32 =
            |d: &mut Vec<u8>, at: usize, v: u32| d[at..at + 4].copy_from_slice(&v.to_le_bytes());
        let desc = body + 0x80;
        p32(&mut d, desc, 1); // attributes: RVA based
        p32(&mut d, desc + 4, 0x11A0); // DLL name
        p32(&mut d, desc + 8, 0x11B0); // module handle
        p32(&mut d, desc + 12, iat);
        p32(&mut d, desc + 16, 0x1100); // name table
        d[body + 0x1A0..body + 0x1AB].copy_from_slice(b"xinput.dll\0");
        let at = 0x58 + 112 + pe::IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT * 8;
        p32(&mut d, at, 0x1080);
        p32(&mut d, at + 4, 64);
        d
    }

    #[test]
    fn synthetic_delay_imports_use_their_own_iat() {
        let (data, body) = synthetic_image_with_imports_and_tls();
        let info = ImageInfo::parse(&with_delay_imports(&data, body, 0x1160)).unwrap();
        assert_eq!(info.delay_imports.len(), 1);
        assert_eq!(info.delay_imports[0].dll, "xinput.dll");
        assert_eq!(symbols(&info.delay_imports[0]), beep_then_ordinal_7(0x1160));
        // The ordinary imports are unchanged, and `import_count` leaves the
        // delay-loaded ones out.
        assert_eq!(symbols(&info.imports[0]), beep_then_ordinal_7(0x1140));
        assert_eq!(info.import_count(), 2);
    }

    #[test]
    fn synthetic_delay_import_iat_must_fit_inside_the_image() {
        let (data, body) = synthetic_image_with_imports_and_tls();
        // Two slots from 0x1FF8 run past SizeOfImage (0x2000), and one at
        // the 4 GiB limit would overflow.
        for iat in [0x1FF8, 0xFFFF_FFF8] {
            assert!(
                matches!(
                    ImageInfo::parse(&with_delay_imports(&data, body, iat)),
                    Err(PeError::Malformed(_))
                ),
                "delay IAT at {iat:#x}"
            );
        }
    }

    #[test]
    fn header_flags_and_subsystem_on_the_synthetic_image() {
        let base = synthetic_image(&[0; 0x80]);
        let parse = |patch: &dyn Fn(&mut Vec<u8>)| {
            let mut d = base.clone();
            patch(&mut d);
            ImageInfo::parse(&d).unwrap()
        };
        let info = parse(&|_| {});
        assert_eq!(info.subsystem_name(), "other");
        assert!(!info.has_relocations);
        assert!(!info.has_exception_table);
        assert!(!info.has_load_config);
        assert!(info.codeview.is_none());

        let directory = |d: &mut Vec<u8>, index: usize| {
            let at = 0x58 + 112 + index * 8;
            d[at..at + 4].copy_from_slice(&0x1100u32.to_le_bytes());
            d[at + 4..at + 8].copy_from_slice(&8u32.to_le_bytes());
        };
        let info = parse(&|d| {
            directory(d, dir::BASERELOC);
            directory(d, dir::EXCEPTION);
            directory(d, dir::LOAD_CONFIG);
        });
        assert!(info.has_relocations && info.has_exception_table && info.has_load_config);
        // A relocation directory doesn't count once the file says its
        // relocations were stripped,
        let stripped = parse(&|d| {
            directory(d, dir::BASERELOC);
            d[0x56] |= layout::file_flags::RELOCS_STRIPPED as u8;
        });
        assert!(!stripped.has_relocations);
        // nor does one with an RVA but no size.
        let empty = parse(&|d| {
            directory(d, dir::BASERELOC);
            let at = 0x58 + 112 + dir::BASERELOC * 8 + 4;
            d[at..at + 4].fill(0);
        });
        assert!(!empty.has_relocations);

        // An executable, not a DLL.
        let exe = parse(&|d| d[0x57] &= !((layout::file_flags::DLL >> 8) as u8));
        assert!(!exe.is_dll());

        use layout::subsystem::*;
        for (value, name) in [
            (WINDOWS_GUI, "windows-gui"),
            (WINDOWS_CUI, "windows-cui"),
            (XBOX, "xbox"),
            (XBOX_CODE_CATALOG, "xbox-code-catalog"),
            (1, "other"),
        ] {
            let info = parse(&|d| d[0x58 + 68..0x58 + 70].copy_from_slice(&value.to_le_bytes()));
            assert_eq!(info.subsystem, value);
            assert_eq!(info.subsystem_name(), name);
        }
    }

    #[test]
    fn codeview_symbol_key_uses_windows_guid_byte_order() {
        let cv = CodeViewInfo {
            pdb_path: "t.pdb".into(),
            guid: [
                0x33, 0x22, 0x11, 0x00, 0x55, 0x44, 0x77, 0x66, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD,
                0xEE, 0xFF,
            ],
            age: 0x1A,
        };
        // Data1-3 little-endian, Data4 as bytes, then the age in hex with
        // no padding.
        assert_eq!(cv.symbol_key(), "00112233445566778899AABBCCDDEEFF1A");
        let cv = CodeViewInfo { age: 1, ..cv };
        assert!(cv.symbol_key().ends_with("FF1"));
    }
}
