//! Replay synthetic containers through the command a contributor will run.
use std::path::Path;
use std::process::{Command, Output};

fn container(tag: u8) -> Vec<u8> {
    // A deliberately truncated XDAT table gives an observable translation
    // refusal. Vary only the container digest, keeping the DXIL identical.
    let mut part = vec![0; 0x18];
    part.extend(b"XDAT");
    let mut bytes = b"DXBC".to_vec();
    bytes.extend([tag; 16]);
    for word in [1, 44 + part.len() as u32, 1, 36] {
        bytes.extend(word.to_le_bytes());
    }
    bytes.extend(b"DXIL");
    bytes.extend((part.len() as u32).to_le_bytes());
    bytes.extend(part);
    bytes
}

fn id(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex()[..16].to_owned()
}

fn packed(bytes: &[u8]) -> Vec<u8> {
    let deflate = miniz_oxide::deflate::compress_to_vec(bytes, 1);
    let n = deflate.len() as u32 + 12;
    let mut record = b"SGD5\x02P\x00\x32".to_vec();
    for word in [n, 0x000d_dce7, n, bytes.len() as u32] {
        record.extend(word.to_le_bytes());
    }
    record.extend(deflate);
    record
}

fn translate(path: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_victory"))
        .arg("translate")
        .arg(path)
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn replays_selected_raw_and_packed_identities_even_with_identical_dxil() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("archive");
    let (a, b, c) = (container(1), container(2), container(3));
    let mut archive = vec![0xaa; 11];
    archive.extend(&c);
    archive.extend(&a);
    archive.extend(&a);
    let mut padded = b.clone();
    padded.extend([0xcc; 9]);
    archive.extend(packed(&padded));
    archive.extend([0xbb; 19]);
    std::fs::write(&input, archive).unwrap();

    let hashes = format!("{},{},{}", id(&a).to_uppercase(), id(&b), id(&padded));
    let run = translate(&input, &["--shader-hash", &hashes, "--jobs", "2", "--list"]);
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let out = String::from_utf8(run.stdout).unwrap();
    assert!(
        out.contains("hash matches 4 (1 duplicates skipped)"),
        "{out}"
    );
    for hash in [id(&a), id(&b), id(&padded)] {
        assert!(
            out.lines()
                .any(|l| l.contains(&hash) && l.contains("truncated")),
            "{out}"
        );
    }
    assert!(!out.contains(&id(&c)), "{out}");

    let run = translate(&input, &[]);
    assert!(run.status.success());
    let out = String::from_utf8(run.stdout).unwrap();
    assert!(
        out.contains("containers   4 (3 duplicates skipped)"),
        "{out}"
    );
}

#[test]
fn padded_file_length_is_a_candidate_only_at_the_start_of_a_raw_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shader");
    let declared = container(1);
    let mut padded = declared.clone();
    padded.extend([0xaa; 16]);
    std::fs::write(&path, &padded).unwrap();
    let hashes = format!("{},{}", id(&declared), id(&padded));
    let run = translate(&path, &["--shader-hash", &hashes]);
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(String::from_utf8_lossy(&run.stdout).contains("hash matches 2 (0 duplicates skipped)"));

    let mut embedded = vec![0xbb; 8];
    embedded.extend(&padded);
    std::fs::write(&path, embedded).unwrap();
    let run = translate(&path, &["--shader-hash", &hashes]);
    assert!(!run.status.success());
    assert!(String::from_utf8_lossy(&run.stdout).contains("hash matches 1 (0 duplicates skipped)"));
    assert!(String::from_utf8_lossy(&run.stderr).contains(&id(&padded)));
}

#[test]
fn missing_requests_are_reported_after_the_matches() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shader");
    let bytes = container(1);
    std::fs::write(&path, &bytes).unwrap();
    let run = translate(
        &path,
        &[
            "--shader-hash",
            &id(&bytes),
            "--shader-hash",
            "ffffffffffffffff,0000000000000000",
        ],
    );
    assert!(!run.status.success());
    assert!(String::from_utf8_lossy(&run.stdout).contains(&id(&bytes)));
    let err = String::from_utf8_lossy(&run.stderr);
    assert!(
        err.contains("0000000000000000") && err.contains("ffffffffffffffff"),
        "{err}"
    );
}

#[test]
fn rejects_malformed_hashes() {
    for hash in [
        "0",
        "1234567890abcdef0",
        "+abcdef012345678",
        "xyz0000000000000",
    ] {
        let run = translate(Path::new("unused"), &["--shader-hash", hash]);
        assert!(!run.status.success());
        assert!(String::from_utf8_lossy(&run.stderr).contains("16 hexadecimal digits"));
    }
}
