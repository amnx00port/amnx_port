//! Two `translate --list` outputs, program by program: what the second run
//! gained, lost, or stopped for another reason. A change is measured against
//! `main` with it, each run on the tester's own copy of the title.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Who refuses a module, in the order it meets them: our translator, the
/// validator (`--validate`), the driver (`--pipelines`).
const TRANSLATOR: usize = 0;
const VALIDATOR: usize = 1;
const DRIVER: usize = 2;

const STAGES: [&str; 8] = ["Ls", "Hs", "Es", "Gs", "Vs", "Ps", "Ms", "Cs"];

/// Every program of one run, by key.
type Programs = BTreeMap<String, Outcome>;

/// What one run did with a program.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Outcome {
    /// The refusal at each stage that gave one.
    refused: [Option<String>; 3],
    /// The validator's and the driver's verdicts, where they gave one.
    verdicts: [Option<bool>; 2],
}

impl Outcome {
    /// Adds one line of the list. False if it contradicts what the program
    /// has: another program under the same key (two of a stage in one
    /// container), which the report then counts as one.
    fn add(&mut self, line: &str) -> bool {
        // A program refused by the translator never reaches the checks.
        let translated = self.refused[TRANSLATOR].is_none();
        let (stage, ok) = match line {
            "validated" => (VALIDATOR, true),
            "pipeline created" => (DRIVER, true),
            l if l.starts_with("validator: ") => (VALIDATOR, false),
            l if l.starts_with("pipeline: ") => (DRIVER, false),
            l => {
                let first = translated && self.verdicts == [None; 2];
                if translated {
                    self.refused[TRANSLATOR] = Some(l.to_owned());
                }
                return first;
            }
        };
        if !ok {
            self.refused[stage] = Some(line.to_owned());
        }
        let verdict = &mut self.verdicts[stage - 1];
        let first = verdict.is_none() && translated;
        *verdict = Some(ok);
        first
    }

    /// The validator's or the driver's verdict, where the run asked for it
    /// and the program got that far.
    fn taken(&self, stage: usize) -> Option<bool> {
        self.verdicts[stage - 1]
    }

    /// For the report: the first refusal, the translator's before the
    /// validator's before the driver's; without one, how far the module got.
    fn text(&self) -> String {
        match self.refused.iter().flatten().next() {
            Some(r) if self.taken(DRIVER) == Some(true) => format!("{r}; pipeline created"),
            Some(r) => r.clone(),
            None if self.taken(DRIVER) == Some(true) => "pipeline created".into(),
            None if self.taken(VALIDATOR) == Some(true) => "validated".into(),
            None => "emitted".into(),
        }
    }
}

/// One line of the list into the program's key and the line's outcome.
fn split(line: &str) -> Option<(String, &str)> {
    // A validator's message may hold two spaces itself: the outcome starts
    // at the first pair that ends a program's label.
    let cut = line
        .match_indices("  ")
        .map(|(at, _)| at)
        .find(|&at| ends_a_label(&line[..at]))?;
    let (label, outcome) = line.split_at(cut);
    Some((key(label)?, &outcome[2..]))
}

/// Whether `head` ends in a stage, as a program's label does, with the
/// instruction offset a refusal adds or without one.
fn ends_a_label(head: &str) -> bool {
    let mut words = head.rsplit(' ');
    let Some(last) = words.next() else {
        return false;
    };
    let stage = match (last.starts_with("0x"), words.next()) {
        (true, Some("at")) => words.next(),
        _ => Some(last),
    };
    stage.is_some_and(|s| STAGES.contains(&s))
}

/// A label, `<file>@<offset>[ <record>] <Stage>[ at 0x<offset>]` as
/// `translate` writes it, into its key: the file's name and the rest, so
/// that copies of the title in other folders compare. `None` if the label
/// names no program.
fn key(label: &str) -> Option<String> {
    // A refusal's instruction offset is no part of which program it is.
    let label = match label.rsplit_once(" at 0x") {
        Some((head, offset)) if offset.chars().all(|c| c.is_ascii_hexdigit()) => head,
        _ => label,
    };
    let (at, stage) = label.rsplit_once(' ')?;
    let (file, rest) = at.rsplit_once("@0x")?;
    let offset = rest.chars().take_while(|c| c.is_ascii_hexdigit()).count();
    if offset == 0 {
        return None;
    }
    let file = file.rsplit(['/', '\\']).next()?;
    Some(format!(
        "{file}@0x{}{} {stage}",
        &rest[..offset],
        &rest[offset..]
    ))
}

/// Every program of a `translate --list` output, and how many of its keys
/// two programs share. `None` if no line names a program.
fn parse(text: &str) -> Option<(Programs, usize)> {
    let mut programs = Programs::new();
    let mut shared = 0;
    for line in text.lines() {
        let Some((key, outcome)) = split(line) else {
            continue;
        };
        if !programs.entry(key).or_default().add(outcome) {
            shared += 1;
        }
    }
    (!programs.is_empty()).then_some((programs, shared))
}

fn read(path: &Path) -> Result<(Programs, usize)> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    parse(&text).with_context(|| {
        format!(
            "{} names no program: not a translate --list output",
            path.display()
        )
    })
}

/// How one check fared: each run's own totals, and the gains and losses
/// over the programs both runs have.
struct Check {
    /// Whether each run asked for it.
    asked: [bool; 2],
    made: [usize; 2],
    gained: Vec<String>,
    lost: Vec<String>,
}

fn check(stage: usize, runs: [&Programs; 2], both: &[(&String, &Outcome, &Outcome)]) -> Check {
    let mut c = Check {
        asked: runs.map(|p| p.values().any(|o| o.taken(stage).is_some())),
        made: runs.map(|p| p.values().filter(|o| o.taken(stage) == Some(true)).count()),
        gained: Vec::new(),
        lost: Vec::new(),
    };
    if c.asked != [true; 2] {
        return c;
    }
    for &(key, a, b) in both {
        // A program a run stopped earlier never reached the check.
        match (a.taken(stage) == Some(true), b.taken(stage) == Some(true)) {
            (false, true) => c.gained.push(key.clone()),
            (true, false) => c.lost.push(key.clone()),
            _ => {}
        }
    }
    c
}

/// `after` against `before`, as the command prints it.
fn report(before: &Programs, after: &Programs) -> String {
    let mut lines = Vec::new();
    let keys: BTreeSet<&String> = before.keys().chain(after.keys()).collect();
    let mut both = Vec::new();
    let mut only = [Vec::new(), Vec::new()];
    for key in keys {
        match (before.get(key), after.get(key)) {
            (Some(a), Some(b)) => both.push((key, a, b)),
            (Some(_), None) => only[0].push(key.as_str()),
            (None, Some(_)) => only[1].push(key.as_str()),
            (None, None) => unreachable!("a key of one of the two"),
        }
    }
    lines.push(format!(
        "programs     {} in both, {} only before, {} only after",
        both.len(),
        only[0].len(),
        only[1].len()
    ));
    let runs = [before, after];
    let mut checks = Vec::new();
    for (name, stage, flag) in [
        ("pipelines", DRIVER, "--pipelines"),
        ("validated", VALIDATOR, "--validate"),
    ] {
        let c = check(stage, runs, &both);
        if c.asked != [true; 2] {
            lines.push(format!("{name:<12} not asked of both runs ({flag})"));
        } else {
            lines.push(format!(
                "{name:<12} {} before, {} after: {} gained, {} lost",
                c.made[0],
                c.made[1],
                c.gained.len(),
                c.lost.len()
            ));
        }
        checks.push(c);
    }
    // The driver takes modules the validator refuses; a gain among them
    // runs, maybe wrongly, so it is said apart.
    if checks.iter().all(|c| c.asked == [true; 2]) {
        let unvalidated = |programs: &Programs| {
            programs
                .values()
                .filter(|o| o.taken(VALIDATOR) == Some(false) && o.taken(DRIVER) == Some(true))
                .count()
        };
        let gained = checks[0]
            .gained
            .iter()
            .filter(|&k| after[k].taken(VALIDATOR) == Some(false))
            .count();
        lines.push(format!(
            "unvalidated  {} before, {} after: pipelines from modules the validator refuses; \
             {} of the pipelines gained",
            unvalidated(before),
            unvalidated(after),
            gained
        ));
    }
    let mut moves: BTreeMap<(String, String), usize> = BTreeMap::new();
    for (_, a, b) in &both {
        let (x, y) = (a.text(), b.text());
        if x != y {
            *moves.entry((x, y)).or_default() += 1;
        }
    }
    lines.push(format!(
        "moves        {} programs end differently",
        moves.values().sum::<usize>()
    ));
    let mut moves: Vec<_> = moves.into_iter().collect();
    moves.sort_by(|x, y| y.1.cmp(&x.1).then_with(|| x.0.cmp(&y.0)));
    for ((x, y), n) in moves {
        lines.push(format!("{n:>12}  {x}  ->  {y}"));
    }
    let lost: Vec<_> = both
        .iter()
        .copied()
        .filter(|&(_, a, b)| {
            [(DRIVER, &checks[0]), (VALIDATOR, &checks[1])]
                .iter()
                .any(|&(s, c)| {
                    c.asked == [true; 2] && a.taken(s) == Some(true) && b.taken(s) != Some(true)
                })
        })
        .collect();
    lines.push(format!("lost         {}", lost.len()));
    for (key, a, b) in lost {
        lines.push(format!("  {key}"));
        lines.push(format!("    before: {}", a.text()));
        lines.push(format!("    after:  {}", b.text()));
    }
    for (name, keys) in ["only before", "only after"].iter().zip(&only) {
        if keys.is_empty() {
            continue;
        }
        lines.push(format!(
            "{name:<12} {}: {}",
            keys.len(),
            keys.iter().take(10).copied().collect::<Vec<_>>().join(", ")
        ));
    }
    lines.join("\n")
}

pub fn run(before: PathBuf, after: PathBuf) -> Result<()> {
    println!("before       {}", before.display());
    println!("after        {}", after.display());
    let (a, shared_a) = read(&before)?;
    let (b, shared_b) = read(&after)?;
    if shared_a + shared_b > 0 {
        println!("shared keys  {shared_a} before, {shared_b} after: compared as one program");
    }
    println!("{}", report(&a, &b));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const BEFORE: &str = "validator    dxil.dll\n\
containers   6 (0 duplicates skipped)\n\
D:\\game\\shaders.rpf@0x10 A_Wrapped Cs  pipeline created\n\
D:\\game\\shaders.rpf@0x10 A_Wrapped Cs  validated\n\
D:\\game\\shaders.rpf@0x20 B_Wrapped Cs  emit: a select on an untracked mask\n\
D:\\game\\shaders.rpf@0x30 C_Wrapped Ps  pipeline created\n\
D:\\game\\shaders.rpf@0x30 C_Wrapped Ps  validator: Function: main: error: Entry function performs some operation that is incompatible.  See other errors for details.\n\
D:\\game\\shaders.rpf@0x40 D_Wrapped Cs at 0x2c  resources: an address in an SGPR that is no root parameter\n\
D:\\game\\shaders.rpf@0x50 E_Wrapped Cs  pipeline: El parámetro no es correcto.\n\
D:\\game\\shaders.rpf@0x50 E_Wrapped Cs  validated\n\
D:\\game\\shaders.rpf@0x60 F_Wrapped Ms  validated\n\
      1  emit: a select on an untracked mask  (e.g. D:\\game\\shaders.rpf@0x20 B_Wrapped Cs)\n";

    /// `BEFORE` on another machine: another path, and B now runs, D is
    /// stopped elsewhere, A lost its pipeline, C and E are gone, G is new.
    const AFTER: &str = "D:/other copy/shaders.rpf@0x10 A_Wrapped Cs  validated\n\
D:/other copy/shaders.rpf@0x20 B_Wrapped Cs  pipeline created\n\
D:/other copy/shaders.rpf@0x20 B_Wrapped Cs  validated\n\
shaders.rpf@0x40 D_Wrapped Cs at 0x44  emit: a descriptor read only as data\n\
shaders.rpf@0x60 F_Wrapped Ms  validated\n\
shaders.rpf@0x70 G_Wrapped Vs  pipeline created\n";

    fn parse_both() -> (Programs, Programs) {
        (
            parse(BEFORE).expect("programs").0,
            parse(AFTER).expect("programs").0,
        )
    }

    #[test]
    fn lines_of_one_program_make_one_outcome() {
        let (before, _) = parse_both();
        let a = &before["shaders.rpf@0x10 A_Wrapped Cs"];
        assert_eq!(
            (a.taken(VALIDATOR), a.taken(DRIVER)),
            (Some(true), Some(true))
        );
        assert_eq!(a.text(), "pipeline created");
        let c = &before["shaders.rpf@0x30 C_Wrapped Ps"];
        assert_eq!(c.taken(VALIDATOR), Some(false));
        assert_eq!(
            c.text(),
            "validator: Function: main: error: Entry function performs some \
             operation that is incompatible.  See other errors for details.; pipeline \
             created"
        );
        let e = &before["shaders.rpf@0x50 E_Wrapped Cs"];
        assert_eq!(
            (e.taken(VALIDATOR), e.taken(DRIVER)),
            (Some(true), Some(false))
        );
        assert_eq!(e.text(), "pipeline: El parámetro no es correcto.");
    }

    #[test]
    fn refusals_read_translator_first() {
        let mut o = Outcome::default();
        assert!(o.add("pipeline: refused"));
        assert!(o.add("validator: refused too"));
        assert_eq!(o.text(), "validator: refused too");
        let mut p = Outcome::default();
        assert!(p.add("I/O: an interpolation with computed barycentrics"));
        assert_eq!(p.text(), "I/O: an interpolation with computed barycentrics");
        // A refused program never reaches the checks, and is refused once.
        assert!(!p.add("validated"));
        assert!(!p.add("emit: another program under the key"));
        assert_eq!(
            p.refused[TRANSLATOR].as_deref(),
            Some("I/O: an interpolation with computed barycentrics")
        );
    }

    #[test]
    fn keys_name_the_file_not_its_path() {
        let (before, after) = parse_both();
        for key in [
            "shaders.rpf@0x10 A_Wrapped Cs",
            "shaders.rpf@0x40 D_Wrapped Cs",
        ] {
            assert!(before.contains_key(key) && after.contains_key(key), "{key}");
        }
        assert_eq!((before.len(), after.len()), (6, 5));
        // A summary's example is no program line, nor is its header.
        assert_eq!(
            parse("      1  emit: x  (e.g. D:\\g\\shaders.rpf@0x20 B Cs)\n"),
            None
        );
        assert_eq!(parse("containers   6\n"), None);
    }

    #[test]
    fn the_report_counts_gains_losses_and_moves() {
        let (before, after) = parse_both();
        let report = report(&before, &after);
        let lines: Vec<&str> = report.lines().collect();
        assert_eq!(
            lines[0],
            "programs     4 in both, 2 only before, 1 only after"
        );
        assert_eq!(lines[1], "pipelines    2 before, 2 after: 1 gained, 1 lost");
        assert_eq!(lines[2], "validated    3 before, 3 after: 1 gained, 0 lost");
        assert_eq!(
            lines[3],
            "unvalidated  1 before, 0 after: pipelines from modules the validator refuses; \
             0 of the pipelines gained"
        );
        assert_eq!(lines[4], "moves        3 programs end differently");
        assert!(report.contains("emit: a select on an untracked mask  ->  pipeline created"));
        assert!(report.contains("resources: an address in an SGPR that is no root parameter  ->  emit: a descriptor read only as data"));
        assert!(report.contains("lost         1\n  shaders.rpf@0x10 A_Wrapped Cs\n    before: pipeline created\n    after:  validated"));
        assert!(report.contains(
            "only before  2: shaders.rpf@0x30 C_Wrapped Ps, shaders.rpf@0x50 E_Wrapped Cs"
        ));
        assert!(report.contains("only after   1: shaders.rpf@0x70 G_Wrapped Vs"));
    }

    #[test]
    fn two_programs_under_one_key_are_counted() {
        let text = "shaders.rpf@0x10 A_Wrapped Cs  pipeline created\n\
                    shaders.rpf@0x10 A_Wrapped Cs  emit: another program\n";
        let (programs, shared) = parse(text).unwrap();
        assert_eq!((programs.len(), shared), (1, 1));
    }
}
