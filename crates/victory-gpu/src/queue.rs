//! Command queues. Submitted lists run on the host GPU and are waited for,
//! so fence signals complete at once.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, QUEUE, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const QUEUE_IIDS: [Iid; 4] = [
    guid(
        0x0ec870a6,
        0x5d7e,
        0x4c22,
        [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDesc {
    kind: u32,
    priority: i32,
    flags: u32,
    nodes: u32,
}

/// The title reads the list type right after the header.
#[repr(C)]
struct Queue {
    object: Object,
    kind: u32,
    desc: QueueDesc,
    schedule:std::sync::Arc<Schedule>,
}
const _: () = assert!(std::mem::offset_of!(Queue, kind) == 0x10);


// Own references while operations are waiting in
// the FIFO. Addresses are only used while pinned.
struct Retained {
    address:usize,
}

impl Retained {
    fn new(this:This)->Self {
        object::add_ref(this);
        Self{address:this as usize}
    }

    fn this(&self)->This {
        self.address as This
    }
}

impl Drop for Retained {
    fn drop(&mut self) {
        object::release_retained(self.this());
    }
}

enum Scheduled {
    Wait(Retained,u64),
    Work(Box<dyn FnOnce()+Send+'static>),
}

#[derive(Default)]
struct ScheduleState {
    running:bool,
    pending:std::collections::VecDeque<Scheduled>,
}

#[derive(Default)]
struct Schedule {
    state:std::sync::Mutex<ScheduleState>,
}

impl Schedule {
    fn push(self:&std::sync::Arc<Self>,op:Scheduled) {
        let mut s=self.state.lock()
            .unwrap_or_else(|e|e.into_inner());

        s.pending.push_back(op);

        if s.running {
            return;
        }

        s.running=true;
        drop(s);

        // Preserve the original synchronous fast path
        // until a future fence introduces a dependency.
        self.drive(false);
    }

    fn drive(self:&std::sync::Arc<Self>,worker:bool) {
        loop {
            let op={
                let mut s=self.state.lock()
                    .unwrap_or_else(|e|e.into_inner());

                match s.pending.pop_front() {
                    Some(op)=>op,
                    None=>{
                        s.running=false;
                        return;
                    }
                }
            };

            match op {
                Scheduled::Work(job)=>job(),

                Scheduled::Wait(fence,target)=>{
                    let ticket=crate::fence::queue_ticket(
                        fence.this(),target
                    );

                    let ready=||{
                        ticket.load(Ordering::Acquire)
                        ||crate::fence::completed(
                            fence.this()
                        )>=target
                    };

                    if ready() {
                        continue;
                    }

                    if worker {
                        while !ready() {
                            std::thread::sleep(
                                std::time::Duration::from_millis(1)
                            );
                        }
                    } else {
                        let schedule=self.clone();

                        drop(std::thread::spawn(move||{
                            while !ticket.load(Ordering::Acquire)
                                && crate::fence::completed(
                                    fence.this()
                                )<target
                            {
                                std::thread::sleep(
                                    std::time::Duration::from_millis(1)
                                );
                            }

                            drop(fence);
                            schedule.drive(true);
                        }));

                        return;
                    }
                }
            }
        }
    }
}


fn queue<'a>(this: This) -> &'a Queue {
    // SAFETY: called from queue methods only.
    unsafe { &*this.cast::<Queue>() }
}

pub(crate) extern "system" fn create(
    _: This,
    desc: *const QueueDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "command queue");
    let queue = Box::new(Queue {
        object: if crate::device::newer() {
            Object::new(&NEWER_FUNCTIONS)
        } else {
            Object::new(&FUNCTIONS)
        },
        kind: desc.kind,
        desc,
        schedule:std::sync::Arc::new(Schedule::default()),
    });
    object::hand_out(QUEUE, queue, iid, &QUEUE_IIDS, out)
}

/// The console's queue description: an engine or pipe and a queue on it
/// in place of PC's priority and node mask.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDescX {
    kind: u32,
    flags: u32,
    engine_or_pipe: u32,
    queue: u32,
}

/// Device method `CreateCommandQueueX`. The host picks engines itself, so
/// the queue is created as on PC.
pub(crate) extern "system" fn create_x(
    this: This,
    desc: *const QueueDescX,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&x) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(desc = ?x, "console command queue");
    let desc = QueueDesc {
        kind: x.kind,
        priority: 0,
        flags: x.flags,
        nodes: 1,
    };
    create(this, &desc, iid, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        QUEUE,
        iid,
        object::is(iid, &QUEUE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Queue>(this)
}

/// Runs the lists' host lists and waits for them (decision 0007).
fn execute_now(_: This, count: u32, lists: *const This) {
    if lists.is_null() {
        return;
    }
    #[cfg(windows)]
    if let Some(host) = crate::host::host() {
        // SAFETY: the title passes `count` lists.
        let lists = unsafe { std::slice::from_raw_parts(lists, count as usize) };
        let host_lists: Vec<_> = lists
            .iter()
            .filter_map(|&l| crate::list::host_list(l))
            .collect();
        host.execute(&host_lists);
        return;
    }
    static ONCE: AtomicBool = AtomicBool::new(false);
    if !ONCE.swap(true, Ordering::Relaxed) {
        tracing::warn!(count, "command lists submitted; no host GPU runs them");
    }
}


extern "system" fn execute(
    this:This,count:u32,lists:*const This
) {
    if count==0 || lists.is_null() {
        return;
    }

    // SAFETY: the queue ABI requires count readable list pointers for this call;
    // the null and zero-count cases were handled above.
    let lists=unsafe{
        std::slice::from_raw_parts(lists,count as usize)
    };

    let held:Vec<_>=lists.iter()
        .copied()
        .filter(|p|!p.is_null())
        .map(crate::list::PendingList::new)
        .collect();

    queue(this).schedule.push(
        Scheduled::Work(Box::new(move||{
            let addresses:Vec<This>=held.iter()
                .map(|list|list.this())
                .collect();

            execute_now(
                std::ptr::null(),
                addresses.len() as u32,
                addresses.as_ptr()
            );
        }))
    );
}


/// The start of a present plane: frame token, then the plane's resources.
#[repr(C)]
pub(crate) struct Plane {
    token: u64,
    count: u32,
    resources: *const This,
}
/// Submission tagged with the frame it belongs to `(count, lists, frame
/// token, flags)`; frames are paced elsewhere, so it runs as a plain one.
extern "system" fn execute_x(this: This, count: u32, lists: *const This, _token: u64, _flags: u32) {
    execute(this, count, lists);
}
/// Signal with sync settings for the top and bottom of the pipe; our
/// submissions are complete when they return, so it signals at once.
extern "system" fn signal2_x(
    this: This,
    fence: This,
    value: u64,
    _bottom: *const u8,
    _top: *const u8,
) -> HRESULT {
    signal(this, fence, value)
}
/// Markers and events only matter to profilers.
extern "system" fn marker(_: This) {}
extern "system" fn signal(
    this:This,fence:This,value:u64
)->HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }

    let held=Retained::new(fence);

    queue(this).schedule.push(
        Scheduled::Work(Box::new(move||{
            let _=crate::fence::signal(
                held.this(),value
            );
        }))
    );

    S_OK
}
/// Nothing runs on the queue, so waiting for a fence never blocks it.
extern "system" fn wait(
    this:This,fence:This,value:u64
)->HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }

    queue(this).schedule.push(
        Scheduled::Wait(
            Retained::new(fence),value
        )
    );

    S_OK
}
/// Ticks per second of GPU timestamps (see [`Written::GpuTimestamp`]).
extern "system" fn timestamp_frequency(_: This, out: *mut u64) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null.
    unsafe { out.write(TIMESTAMP_FREQUENCY) };
    S_OK
}
const TIMESTAMP_FREQUENCY: u64 = 100_000_000;

/// What a list's bottom-of-pipe write stores when the list has run.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Written {
    Value(u64),
    /// The GPU clock: [`TIMESTAMP_FREQUENCY`] ticks per second since the
    /// process started (our GPU work is done on the CPU's timeline).
    GpuTimestamp,
    /// The CPU's performance counter.
    SystemTimestamp,
}

impl Written {
    /// Stores the low `bytes` (4 or 8) of the value at `at`.
    pub(crate) fn write(self, at: *mut u8, bytes: usize) {
        let value = match self {
            Written::Value(v) => v,
            Written::GpuTimestamp => {
                static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
                let elapsed = START.get_or_init(std::time::Instant::now).elapsed();
                (elapsed.as_nanos() / (1_000_000_000 / u128::from(TIMESTAMP_FREQUENCY))) as u64
            }
            Written::SystemTimestamp => system_timestamp(),
        };
        // SAFETY: the title names writable GPU memory (unified with ours).
        unsafe {
            if bytes == 4 {
                at.cast::<u32>().write_unaligned(value as u32);
            } else {
                at.cast::<u64>().write_unaligned(value);
            }
        }
    }
}

fn system_timestamp() -> u64 {
    #[cfg(windows)]
    {
        let mut count = 0;
        // SAFETY: a valid out pointer.
        unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut count) };
        count as u64
    }
    #[cfg(not(windows))]
    0
}

/// Frames presented so far.
static FRAMES: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "diagnostics")]
pub(crate) fn presented() -> u64 {
    FRAMES.load(Ordering::Relaxed)
}

/// Shows the first plane's first resource in our window.
///
/// shortcut: one plane, no scaling or placement, the source expected in
/// the common state.
fn present_now(
    _: This,
    planes: u32,
    plane_parameters: *const Plane,
    _parameters: *const u8,
) -> HRESULT {
    let frame = FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
    if frame.is_power_of_two() {
        let (draws, dispatches) = crate::list::take_counts();
        #[cfg(windows)]
        let (drawn, dispatched) = {
            static DRAWN: AtomicU64 = AtomicU64::new(0);
            static DISPATCHED: AtomicU64 = AtomicU64::new(0);
            let since = |last: &AtomicU64, now: u64| now - last.swap(now, Ordering::Relaxed);
            (
                since(&DRAWN, crate::draw::drawn()),
                since(&DISPATCHED, crate::draw::dispatched()),
            )
        };
        #[cfg(not(windows))]
        let (drawn, dispatched) = (0, 0);
        tracing::info!(
            frame,
            draws,
            drawn,
            dispatches,
            dispatched,
            "frame presented (since the last line: draws and dispatches, and those run on the host)"
        );
        #[cfg(windows)]
        if frame >= 256 {
            let (hits, misses) = crate::shader_cache::counts();
            tracing::info!(hits, misses, "shaders from the cache and translated so far");
        }
    }
    #[cfg(all(windows, feature = "diagnostics"))]
    if crate::capture::listed(frame) {
        crate::capture::dump_targets(frame);
    }
    #[cfg(windows)]
    if planes > 0
        // SAFETY: the title passes `planes` plane descriptions.
        && let Some(plane) = unsafe { plane_parameters.as_ref() }
        && plane.count > 0
        && !plane.resources.is_null()
        // SAFETY: the plane lists `count` resources.
        && let resource = unsafe { *plane.resources }
        && let Some(texture) = crate::resource::host_resource(resource)
        && let Some(host) = crate::host::host()
    {
        const COPY_SOURCE: u32 = 0x800;
        let state = crate::resource::host_transition(resource, COPY_SOURCE)
            .map_or(COPY_SOURCE, |(_, before)| before);
        let shown = host.present(texture, state, frame);
        crate::resource::host_transition(resource, state);
        #[cfg(feature = "diagnostics")]
        blinks(frame, resource, state, shown);
        #[cfg(not(feature = "diagnostics"))]
        let _ = shown;
    }
    #[cfg(not(windows))]
    let _ = (planes, plane_parameters);
    S_OK
}

extern "system" fn present(
    this:This,planes:u32,
    parameters:*const Plane,
    _extra:*const u8
)->HRESULT {
    let resource=if planes>0 && !parameters.is_null() {
        // SAFETY: the present ABI supplies at least one readable Plane here.
        let plane=unsafe{&*parameters};

        if plane.count>0 && !plane.resources.is_null() {
            // SAFETY: the plane ABI supplies count readable resource pointers.
            let resource=unsafe{*plane.resources};

            if resource.is_null() {
                None
            } else {
                Some(Retained::new(resource))
            }
        } else {
            None
        }
    } else {
        None
    };

    queue(this).schedule.push(
        Scheduled::Work(Box::new(move||{
            if let Some(resource)=resource {
                let address=resource.this();

                let plane=Plane {
                    token:0,
                    count:1,
                    resources:&address,
                };

                let _=present_now(
                    std::ptr::null(),1,&plane,
                    std::ptr::null()
                );
            } else {
                let _=present_now(
                    std::ptr::null(),0,
                    std::ptr::null(),
                    std::ptr::null()
                );
            }
        }))
    );

    S_OK
}

/// Diagnostics: logs a presented frame that is black after one that was
/// not, with what was presented and how many draws ran since.
#[cfg(all(windows, feature = "diagnostics"))]
fn blinks(frame: u64, resource: This, state: u32, shown: Option<bool>) {
    static LAST: std::sync::Mutex<(bool, u64, u64)> = std::sync::Mutex::new((false, 0, 0));
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let drawn = crate::draw::drawn();
    let black = shown.unwrap_or(false);
    let draws = crate::draw::take_frame();
    if black && !last.0 {
        last.2 += 1;
        tracing::warn!(
            frame,
            resource = ?resource,
            state = format_args!("{state:#x}"),
            draws = drawn - last.1,
            copied = shown.is_some(),
            black_frames = last.2,
            levels = ?crate::capture::recent_levels(),
            "a black frame after a frame that was not"
        );
        for d in draws {
            tracing::warn!("black frame {frame}: {d}");
        }
        AFTER.store(frame + 6, Ordering::Relaxed);
    }
    // Six frames on: the levels around it (a fade or a single drop).
    static AFTER: AtomicU64 = AtomicU64::new(0);
    if AFTER.load(Ordering::Relaxed) == frame {
        tracing::warn!(frame, levels = ?crate::capture::recent_levels(), "levels after the black frame");
    }
    *last = (black, drawn, last.2);
}

/// A run of destination pages: how they are mapped (0 pool pages from
/// `start` on, 1 pool page `start` each, 2 unmapped, 3 left as they are)
/// and how many.
#[repr(C)]
#[derive(Clone,Copy)]
pub(crate) struct PageRange {
    kind: u32,
    start: u32,
    count: u32,
}

/// The ranges mapped from a destination address on.
#[repr(C)]
pub(crate) struct PageBatch {
    destination: u64,
    count: u32,
    ranges: *const PageRange,
}

/// Bytes per mapped page.
const PAGE: u64 = 64 << 10;

/// `CopyPageMappingsBatchX(count, batches, pool, pool pages, flags)`: maps
/// pages of a pool (memory the title mapped physical pages into) at the
/// batches' addresses. Grand Theft Auto VI streams textures into pool pages
/// and maps them into its tiled heap this way (title+0xf2c470, ranges of
/// one constant page or of unmapped pages). Done at once: every earlier
/// submission has run.
fn copy_page_mappings_now(
    _: This,
    count: u32,
    batches: *const PageBatch,
    pool: u64,
    pool_pages: u32,
    _flags: u32,
) -> HRESULT {
    let Some(map) = crate::resource::page_mapper() else {
        return E_INVALIDARG;
    };
    if batches.is_null() && count > 0 {
        return E_INVALIDARG;
    }
    let batches = (count > 0).then(|| {
        // SAFETY: the title passes `count` batches.
        unsafe { std::slice::from_raw_parts(batches, count as usize) }
    });
    let mut ok = true;
    for b in batches.into_iter().flatten() {
        if b.ranges.is_null() {
            continue;
        }
        // SAFETY: each batch holds `count` ranges.
        let ranges = unsafe { std::slice::from_raw_parts(b.ranges, b.count as usize) };
        let mut at = b.destination;
        for r in ranges {
            let len = u64::from(r.count) * PAGE;
            let inside = |pages: u32| r.start.checked_add(pages).is_some_and(|e| e <= pool_pages);
            let source = pool + u64::from(r.start) * PAGE;
            ok &= match r.kind {
                0 if inside(r.count) => map(at, source, len),
                1 if inside(1) => (0..u64::from(r.count)).all(|k| map(at + k * PAGE, source, PAGE)),
                2 => map(at, 0, len),
                3 => true,
                _ => false,
            };
            at += len;
        }
    }
    if !ok {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!(pool = format_args!("{pool:#x}"), "a page mapping failed"));
    }
    S_OK
}


extern "system" fn copy_page_mappings(
    this:This,count:u32,
    batches:*const PageBatch,
    pool:u64,pool_pages:u32,flags:u32
)->HRESULT {
    if crate::resource::page_mapper().is_none()
        || (count>0 && batches.is_null())
    {
        return E_INVALIDARG;
    }

    let mut owned=Vec::new();

    if count>0 {
        // SAFETY: the mapping ABI supplies count readable PageBatch values;
        // batches is non-null when count is positive.
        for batch in unsafe{
            std::slice::from_raw_parts(
                batches,count as usize
            )
        } {
            if batch.ranges.is_null() {
                continue;
            }

            // SAFETY: each batch supplies count readable ranges, copied before return.
            let ranges=unsafe{
                std::slice::from_raw_parts(
                    batch.ranges,batch.count as usize
                )
            }.to_vec();

            owned.push((batch.destination,ranges));
        }
    }

    queue(this).schedule.push(
        Scheduled::Work(Box::new(move||{
            let views:Vec<PageBatch>=owned.iter()
                .map(|(destination,ranges)|PageBatch{
                    destination:*destination,
                    count:ranges.len() as u32,
                    ranges:ranges.as_ptr(),
                })
                .collect();

            let _=copy_page_mappings_now(
                std::ptr::null(),
                views.len() as u32,
                views.as_ptr(),
                pool,pool_pages,flags
            );
        }))
    );

    S_OK
}

extern "system" fn desc(this: This, out: *mut QueueDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(queue(this).desc) };
    }
}

/// The queue methods by their slot in older titles' queues (the 2020-era
/// layout): IUnknown 3, object 4, device child 1, queue 26 (12 as on PC,
/// then console ones).
const METHODS: [(usize, *const ()); 15] = [
    (0, query as *const ()),
    (2, release as *const ()),
    (7, object::parent_device as *const ()), // 0x38
    (10, execute as *const ()),              // 0x50
    (11, marker as *const ()),               // 0x58 set marker
    (12, marker as *const ()),               // 0x60 begin event
    (13, marker as *const ()),               // 0x68 end event
    (14, signal as *const ()),               // 0x70
    (15, wait as *const ()),                 // 0x78
    (16, timestamp_frequency as *const ()),  // 0x80
    (26, desc as *const ()),                 // 0xd0
    (29, copy_page_mappings as *const ()),   // 0xe8 page mapping batch
    (30, present as *const ()),              // 0xf0
    (33, execute_x as *const ()),            // 0x108
    (32, signal2_x as *const ()),            // 0x100
];

/// Newer titles' queues have two more methods before Signal, which moved
/// from 0x70 to 0x80: every slot from 0x70 on is two later (taken here as
/// starting exactly at 0x70, the methods before it kept in place).
/// Inferred from Grand Theft Auto VI's calls: Signal 0x80 `(fence,
/// value)`, timestamp frequency 0x90 (title+0x24841a6), suspend and resume
/// 0xb8/0xc0, PresentX 0x100, the frame-token submission 0x118 `(count,
/// lists, token, flags)` (title+0x24821dd).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 14 { slot + 2 } else { slot }
}

const fn queue_table(newer: bool) -> [*const (); 36] {
    let mut f = object::d3d12_object(object::missing_table::<QUEUE, 36>());
    let mut i = 0;
    while i < METHODS.len() {
        let (slot, method) = METHODS[i];
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    f
}

static FUNCTIONS: Table<36> = Table(queue_table(false));
static NEWER_FUNCTIONS: Table<36> = Table(queue_table(true));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    /// Where a queue's signal is: newer layouts have it two slots later.
    /// Another test may switch layouts while this one runs, so the queue's
    /// own table decides.
    fn signal_slot(q: This) -> usize {
        // SAFETY: one of our queues; its first field is the table.
        let table = unsafe { *q.cast::<*const *const ()>() };
        if table == NEWER_FUNCTIONS.0.as_ptr() {
            0x80
        } else {
            0x70
        }
    }

    #[test]
    fn queue_signals_complete_at_once() {
        let null = std::ptr::null();
        let desc = QueueDesc {
            kind: 0,
            priority: 0,
            flags: 0,
            nodes: 0,
        };
        let mut q = null;
        assert_eq!(create(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        // SAFETY: the title reads the type at +0x10.
        assert_eq!(unsafe { q.cast::<u8>().add(0x10).cast::<u32>().read() }, 0);
        let mut fence = null;
        let fence_iid = &crate::fence::FENCE_IIDS[0];
        assert_eq!(
            crate::fence::create(null, 0, 0, fence_iid, &mut fence),
            S_OK
        );
        let signal = slot::<extern "system" fn(This, This, u64) -> HRESULT>(q, signal_slot(q));
        assert_eq!(signal(q, fence, 7), S_OK);
        assert_eq!(
            slot::<extern "system" fn(This) -> u64>(fence, 0x40)(fence),
            7
        );
        assert_eq!(slot::<extern "system" fn(This) -> u32>(q, 0x10)(q), 0);
    }

    /// Grand Theft Auto VI's compute queue: `{COMPUTE, 0, pipe 1, queue 0}`.
    #[test]
    fn console_queues_are_created_as_on_pc() {
        let null = std::ptr::null();
        let desc = QueueDescX {
            kind: 2,
            flags: 0,
            engine_or_pipe: 1,
            queue: 0,
        };
        let mut q = null;
        assert_eq!(create_x(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        assert_eq!(queue(q).desc.kind, 2);
        let none = std::ptr::null();
        assert_eq!(create_x(null, none, &QUEUE_IIDS[0], &mut q), E_INVALIDARG);
    }

    #[test]
    fn newer_queues_have_their_methods_two_slots_later() {
        let (old, new) = (queue_table(false), queue_table(true));
        assert_eq!(old[10], new[10]); // submission stays at 0x50
        assert_eq!(old[14], new[16]); // signal 0x70 → 0x80
        assert_eq!(old[30], new[32]); // present 0xf0 → 0x100
        assert_eq!(new[35], execute_x as *const ()); // 0x118
    }

    #[test]
    fn timestamps_and_values_are_written() {
        let mut at = [0u64; 2];
        Written::Value(0x1_0000_0002).write(at.as_mut_ptr().cast(), 4);
        assert_eq!(at[0], 2);
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        let first = at[1];
        std::thread::sleep(std::time::Duration::from_millis(2));
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        assert!(at[1] > first);
    }
}


#[cfg(test)]
mod systems_final_queue {
    use super::*;
    use std::time::{Duration,Instant};

    fn make_queue()->This {
        let null=std::ptr::null();
        let mut q=null;

        let desc=QueueDesc{
            kind:0,priority:0,flags:0,nodes:0
        };

        assert_eq!(
            create(null,&desc,&QUEUE_IIDS[0],&mut q),
            S_OK
        );

        q
    }

    fn make_fence()->This {
        let null=std::ptr::null();
        let mut f=null;

        assert_eq!(
            crate::fence::create(
                null,0,0,
                &crate::fence::FENCE_IIDS[0],
                &mut f
            ),S_OK
        );

        f
    }

    fn reached(f:This,target:u64)->bool {
        let deadline=Instant::now()+Duration::from_secs(5);

        while Instant::now()<deadline {
            if crate::fence::completed(f)>=target {
                return true;
            }

            std::thread::sleep(Duration::from_millis(2));
        }

        crate::fence::completed(f)>=target
    }

    fn free(this:This) {
        object::slot::<
            extern "system" fn(This)->u32
        >(this,0x10)(this);
    }

    #[test]
    fn future_signal_orders_fifo() {
        let a=make_queue();
        let b=make_queue();
        let gate=make_fence();
        let done=make_fence();

        assert_eq!(wait(a,gate,5),S_OK);
        assert_eq!(signal(a,done,7),S_OK);
        assert_eq!(crate::fence::completed(done),0);

        assert_eq!(signal(b,gate,5),S_OK);
        assert!(reached(done,7));

        for p in [a,b,gate,done] {
            free(p);
        }

        println!("FINAL_QUEUE_FIFO_PASS");
    }

    #[test]
    fn activated_wait_survives_rewind() {
        let q=make_queue();
        let gate=make_fence();
        let done=make_fence();

        assert_eq!(wait(q,gate,5),S_OK);
        assert_eq!(signal(q,done,1),S_OK);

        assert_eq!(
            crate::fence::signal(gate,5),S_OK
        );

        assert_eq!(
            crate::fence::signal(gate,0),S_OK
        );

        assert!(reached(done,1));

        for p in [q,gate,done] {
            free(p);
        }

        println!("FINAL_QUEUE_REWIND_PASS");
    }

    #[test]
    fn later_wait_activates_in_order() {
        let q=make_queue();
        let first=make_fence();
        let second=make_fence();
        let done=make_fence();

        assert_eq!(wait(q,first,1),S_OK);
        assert_eq!(wait(q,second,1),S_OK);
        assert_eq!(signal(q,done,1),S_OK);

        let _=crate::fence::signal(second,1);
        let _=crate::fence::signal(second,0);
        let _=crate::fence::signal(first,1);

        std::thread::sleep(Duration::from_millis(100));

        let early=crate::fence::completed(done);

        let _=crate::fence::signal(second,1);

        assert!(reached(done,1));
        assert_eq!(early,0);

        for p in [q,first,second,done] {
            free(p);
        }

        println!("FINAL_QUEUE_ACTIVATION_PASS");
    }

    #[cfg(windows)]
    #[test]
    fn command_list_remains_pending_until_gate() {
        let null=std::ptr::null();

        let q=make_queue();
        let gate=make_fence();
        let done=make_fence();

        let mut allocator=null;
        let mut list=null;

        let allocator_iid=guid(
            0x6102dee4,0xaf59,0x4b09,
            [0xb9,0x99,0xb4,0x4d,0x73,0xf0,0x9b,0x24]
        );

        let list_iid=guid(
            0x7116d91c,0xe7e4,0x47ce,
            [0xb8,0xc6,0xec,0x81,0x68,0xf4,0x37,0xe5]
        );

        assert_eq!(
            crate::list::create_allocator(
                null,0,&allocator_iid,&mut allocator
            ),S_OK
        );

        assert_eq!(
            crate::list::create_list(
                null,0,0,allocator,null,
                &list_iid,&mut list
            ),S_OK
        );

        assert!(
            crate::list::host_list(list).is_some(),
            "Host GPU command list unavailable"
        );

        let close=object::slot::<
            extern "system" fn(This)->HRESULT
        >(list,0x48);

        let reset=object::slot::<
            extern "system" fn(This,This,This)->HRESULT
        >(list,0x50);

        assert_eq!(close(list),S_OK);

        assert_eq!(wait(q,gate,1),S_OK);

        execute(q,1,&list);

        assert_eq!(signal(q,done,1),S_OK);

        let pending_reset=reset(list,allocator,null);
        let early=crate::fence::completed(done);

        let _=crate::fence::signal(gate,1);

        assert!(reached(done,1));

        assert_eq!(pending_reset,E_INVALIDARG);
        assert_eq!(early,0);
        assert_eq!(reset(list,allocator,null),S_OK);
        assert_eq!(close(list),S_OK);

        for p in [list,allocator,q,gate,done] {
            free(p);
        }

        println!("FINAL_QUEUE_HOST_LIST_PASS");
    }
}
