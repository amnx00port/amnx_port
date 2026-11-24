//! XTaskQueue and XAsync: the runtime's callback queues and async-call
//! blocks, served as the task-queue provider's object from `QueryApiImpl`.
//!
//! As documented for the public API: a queue has a work port and a
//! completion port, each dispatched manually (the title calls dispatch),
//! on a thread pool, serialised on one thread, or immediately on the
//! submitting thread. A null queue handle means the process queue. An
//! async block's system area holds its status and result size, so both
//! stay readable after our state for the call is gone.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, MutexGuard};
use std::time::Duration;

use super::xgameruntime::{Guid, HRESULT, Interface, Methods, Object, S_OK, add_ref, missing};

/// The task-queue provider; it is also the interface id titles ask for.
pub const PROVIDER: Guid = Guid {
    data1: 0x073b_7dcb,
    data2: 0x1fcf,
    data3: 0x4030,
    data4: [0x94, 0xbe, 0xe3, 0xc9, 0xeb, 0x62, 0x34, 0x28],
};

pub const E_PENDING: HRESULT = 0x8000_000A_u32 as i32;
pub const E_ABORT: HRESULT = 0x8000_4004_u32 as i32;
pub const E_NOTIMPL: HRESULT = 0x8000_4001_u32 as i32;
pub const E_INVALIDARG: HRESULT = 0x8007_0057_u32 as i32;
pub const E_NOT_SUFFICIENT_BUFFER: HRESULT = 0x8007_007A_u32 as i32;
pub const E_ILLEGAL_METHOD_CALL: HRESULT = 0x8000_000E_u32 as i32;

const WORK: u32 = 0;
const COMPLETION: u32 = 1;
const INFINITE: u32 = u32::MAX;

type Callback = extern "system" fn(*mut c_void, bool);
type MonitorCallback = extern "system" fn(*mut c_void, *const Queue, u32);
type TerminatedCallback = extern "system" fn(*mut c_void);
type CompletionRoutine = extern "system" fn(*mut AsyncBlock);
type Provider = extern "system" fn(u32, *const ProviderData) -> HRESULT;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Manual,
    ThreadPool,
    Serialized,
    Immediate,
}

impl Mode {
    fn from_raw(mode: u32) -> Option<Mode> {
        [
            Mode::Manual,
            Mode::ThreadPool,
            Mode::Serialized,
            Mode::Immediate,
        ]
        .get(mode as usize)
        .copied()
    }
}

/// A title callback and its context (addresses, so tasks can cross threads).
#[derive(Clone, Copy)]
struct Task {
    context: usize,
    callback: usize,
}

impl Task {
    fn new(context: *mut c_void, callback: Callback) -> Task {
        Task {
            context: context as usize,
            callback: callback as usize,
        }
    }

    fn run(self, canceled: bool) {
        // SAFETY: built from a `Callback` in `new`.
        let callback: Callback = unsafe { std::mem::transmute(self.callback) };
        callback(self.context as *mut c_void, canceled);
    }
}



enum DispatchKind {
    Normal(u64),
    Cancel(Task, Arc<Stage>),
    Notify(usize, TerminatedCallback, Arc<Stage>),
}

struct Stage {
    remaining: Mutex<usize>,
    changed: Condvar,
}

impl Stage {
    fn new(count: usize) -> Arc<Self> {
        Arc::new(Self {
            remaining: Mutex::new(count),
            changed: Condvar::new(),
        })
    }

    fn done(&self) {
        let mut remaining=lock(&self.remaining);

        *remaining -= 1;

        if *remaining == 0 {
            self.changed.notify_all();
        }
    }

    fn wait(&self) {
        let remaining=lock(&self.remaining);

        drop(
            self.changed.wait_while(
                remaining, |n| *n != 0
            ).unwrap_or_else(|e| e.into_inner())
        );
    }
}

struct Tracked {
    owner: Arc<Queue>,
    kind: DispatchKind,
}

impl Tracked {
    fn run(self) {
        let Tracked { owner, kind }=self;

        match kind {
            DispatchKind::Normal(id) => {
                let (port,task)={
                    let mut state=lock(&owner.state);

                    let Some((port,task))=
                        state.pending.remove(&id)
                    else {
                        return;
                    };

                    state.active[port as usize] += 1;

                    (port,task)
                };

                let _activity=Activity(
                    &owner,port
                );

                task.run(false);
            }

            DispatchKind::Cancel(task,stage) => {
                task.run(true);
                stage.done();
            }

            DispatchKind::Notify(
                context,callback,stage
            ) => {
                callback(context as *mut c_void);
                stage.done();
            }
        }
    }
}

struct Activity<'a>(&'a Queue,u32);

impl Drop for Activity<'_> {
    fn drop(&mut self) {
        let mut state=lock(&self.0.state);

        state.active[self.1 as usize] -= 1;

        self.0.changed.notify_all();
    }
}

struct Port {
    mode: Mode,
    tasks: Mutex<VecDeque<Tracked>>,
    ready: Condvar,
}

impl Port {
    fn new(mode: Mode) -> Arc<Port> {
        let port = Arc::new(Port {
            mode,
            tasks: Mutex::default(),
            ready: Condvar::new(),
        });

        if mode == Mode::Serialized {
            let worker = port.clone();

            std::thread::spawn(move || {
                loop {
                    match worker.take(1000, || false) {
                        Some(task) => task.run(),
                        None if Arc::strong_count(&worker) == 1 => break,
                        None => {}
                    }
                }
            });
        }

        port
    }

    fn submit(&self, task: Tracked) {
        match self.mode {
            Mode::Immediate => task.run(),
            Mode::ThreadPool => POOL.submit(task),
            Mode::Manual | Mode::Serialized => {
                lock(&self.tasks).push_back(task);
                self.ready.notify_one();
            }
        }
    }

    fn take(
        &self,
        timeout_ms: u32,
        ended: impl Fn() -> bool,
    ) -> Option<Tracked> {
        let waiting = |t: &mut VecDeque<Tracked>| {
            t.is_empty() && !ended()
        };

        let tasks = lock(&self.tasks);

        let mut tasks = if timeout_ms == INFINITE {
            self.ready
                .wait_while(tasks, waiting)
                .unwrap_or_else(|e| e.into_inner())
        } else {
            self.ready
                .wait_timeout_while(
                    tasks,
                    Duration::from_millis(timeout_ms.into()),
                    waiting,
                )
                .unwrap_or_else(|e| e.into_inner())
                .0
        };

        // A terminated dispatcher must not consume
        // work that belongs to a surviving composite queue.
        if ended() {
            return None;
        }

        tasks.pop_front()
    }

    fn discard_owner(&self, owner: &Arc<Queue>) {
        let mut tasks = lock(&self.tasks);

        tasks.retain(|task| {
            !Arc::ptr_eq(&task.owner, owner)
        });

        self.ready.notify_all();
    }
}

/// Preserve the upstream pool policy for this experiment.
/// Bounding the pool requires separate starvation tests.
struct Pool {
    state: Mutex<(VecDeque<Tracked>, usize)>,
    ready: Condvar,
}

static POOL: Pool = Pool {
    state: Mutex::new((VecDeque::new(), 0)),
    ready: Condvar::new(),
};

impl Pool {
    fn submit(&'static self, task: Tracked) {
        let mut state = lock(&self.state);

        state.0.push_back(task);

        if state.0.len() > state.1 {
            std::thread::spawn(move || self.work());
        } else {
            self.ready.notify_one();
        }
    }

    fn work(&self) {
        let mut state = lock(&self.state);

        loop {
            while state.0.is_empty() {
                state.1 += 1;

                let (next, timeout) = self.ready.wait_timeout(
                    state, Duration::from_secs(2)
                ).unwrap_or_else(|e| e.into_inner());

                state = next;

                if timeout.timed_out() && state.0.is_empty() {
                    state.1 -= 1;
                    return;
                }

                state.1 -= 1;
            }

            let task = state.0.pop_front();

            drop(state);

            if let Some(task) = task {
                task.run();
            }

            state = lock(&self.state);
        }
    }

    fn discard_owner(&self, owner: &Arc<Queue>) {
        lock(&self.state).0.retain(|task| {
            !Arc::ptr_eq(&task.owner, owner)
        });
    }
}

#[derive(Default)]
struct QueueState {
    next: u64,

    // Ordered by submission ID for deterministic
    // cancellation within each port.
    pending: std::collections::BTreeMap<u64, (u32, Task)>,

    active: [usize; 2],
    termination_complete: bool,
}

/// A queue handle owns a reference to this object.
/// Accepted dispatch tokens also hold references.
pub struct Queue {
    ports: [Arc<Port>; 2],
    monitors: Mutex<Vec<(u64, usize, usize)>>,
    terminated: AtomicBool,
    finished: AtomicBool,
    state: Mutex<QueueState>,
    changed: Condvar,
}


type TimerEntries = std::collections::BTreeMap<
    (std::time::Instant, u64),
    (Arc<Queue>, u64, u32),
>;

// Shared scheduler for delayed callbacks.
// It holds no queue lock while delivering callbacks.
struct SharedTimer {
    entries: Mutex<TimerEntries>,
    changed: Condvar,
    sequence: AtomicU64,
}

static SHARED_TIMER:
    std::sync::OnceLock<Arc<SharedTimer>>=
    std::sync::OnceLock::new();

fn shared_timer()->&'static Arc<SharedTimer> {
    SHARED_TIMER.get_or_init(||{
        let timer=Arc::new(SharedTimer {
            entries:Mutex::default(),
            changed:Condvar::new(),
            sequence:AtomicU64::new(0),
        });

        let worker=timer.clone();

        drop(std::thread::spawn(
            move || worker.run()
        ));

        timer
    })
}


// Elastic delivery workers are reused between timer
// expirations. Idle workers leave after 500 ms.
// No hard limit on active callbacks: one callback may
// wait for another callback that has not started.
type DeliveryJob=Box<dyn FnOnce()+Send+'static>;

#[derive(Default)]
struct DeliveryState {
    jobs:VecDeque<DeliveryJob>,
    idle:usize,
    starting:usize,
    workers:usize,
}

struct DeliveryPool {
    state:Mutex<DeliveryState>,
    ready:Condvar,
}

static TIMER_DELIVERY:LazyLock<DeliveryPool>=
    LazyLock::new(||DeliveryPool {
        state:Mutex::default(),
        ready:Condvar::new(),
    });

impl DeliveryPool {
    fn submit(&'static self,job:DeliveryJob) {
        let mut s=lock(&self.state);
        s.jobs.push_back(job);

        if s.jobs.len()>s.idle+s.starting {
            s.starting+=1;
            s.workers+=1;

            let spawned=std::thread::Builder::new()
                .name("victory-timer-worker".into())
                .spawn(move||self.work());

            if spawned.is_err() {
                s.starting-=1;
                s.workers-=1;

                if s.workers==0 {
                    let job=s.jobs.pop_front();
                    drop(s);

                    if let Some(job)=job {
                        job();
                    }
                    return;
                }
            }
        }

        self.ready.notify_one();
    }

    fn work(&self) {
        let mut s=lock(&self.state);
        s.starting-=1;

        loop {
            if let Some(job)=s.jobs.pop_front() {
                drop(s);
                job();
                s=lock(&self.state);
                continue;
            }

            s.idle+=1;

            let (next,timeout)=self.ready.wait_timeout(
                s,Duration::from_millis(500)
            ).unwrap_or_else(|e|e.into_inner());

            s=next;
            s.idle-=1;

            if timeout.timed_out() && s.jobs.is_empty() {
                s.workers-=1;
                return;
            }
        }
    }
}

impl SharedTimer {
    fn schedule(
        &self,
        owner:&Arc<Queue>,
        id:u64,
        port:u32,
        delay_ms:u32
    ) {
        // Serialize insertion against termination
        // and cancellation of this pending callback.
        let queue_state=lock(&owner.state);

        if !queue_state.pending.contains_key(&id) {
            return;
        }

        let deadline=std::time::Instant::now()
            +Duration::from_millis(
                delay_ms.into()
            );

        let order=self.sequence.fetch_add(
            1,Ordering::Relaxed
        );

        let mut entries=lock(&self.entries);

        entries.insert(
            (deadline,order),
            (owner.clone(),id,port)
        );

        self.changed.notify_one();

        drop(entries);
        drop(queue_state);
    }

    fn discard_owner(&self,owner:&Arc<Queue>) {
        let mut entries=lock(&self.entries);

        entries.retain(
            |_,(queue,_,_)|
                !Arc::ptr_eq(queue,owner)
        );

        self.changed.notify_one();
    }

    fn run(&self) {
        loop {
            let (owner,id,port)={
                let mut entries=lock(
                    &self.entries
                );

                loop {
                    let deadline=entries
                        .first_key_value()
                        .map(|(key,_)|key.0);

                    match deadline {
                        None=>{
                            entries=self.changed
                                .wait(entries)
                                .unwrap_or_else(
                                    |e|e.into_inner()
                                );
                        }

                        Some(at)=>{
                            let now=
                                std::time::Instant::now();

                            if at<=now {
                                break entries
                                    .pop_first()
                                    .unwrap().1;
                            }

                            entries=self.changed
                                .wait_timeout(
                                    entries,at-now
                                )
                                .unwrap_or_else(
                                    |e|e.into_inner()
                                ).0;
                        }
                    }
                }
            };

            // Avoid starting a delivery that has
            // already been canceled.
            if !lock(&owner.state)
                .pending.contains_key(&id)
            {
                continue;
            }

            // An Immediate callback or monitor
            // may block. Do not run it on the
            // shared timer scheduler thread.
            TIMER_DELIVERY.submit(Box::new(move||{
                owner.deliver(id,port);
            }));
        }
    }
}

impl Queue {
    fn new(ports: [Arc<Port>; 2]) -> Arc<Queue> {
        Arc::new(Queue {
            ports,
            monitors: Mutex::default(),
            terminated: AtomicBool::new(false),
            finished: AtomicBool::new(false),
            state: Mutex::default(),
            changed: Condvar::new(),
        })
    }

    fn submit(
        self: &Arc<Self>,
        port: u32,
        delay_ms: u32,
        task: Task,
    ) -> HRESULT {
        if port > COMPLETION {
            return E_INVALIDARG;
        }

        let id = {
            let mut state = lock(&self.state);

            if self.terminated.load(Ordering::Acquire) {
                return E_ABORT;
            }

            let Some(next) = state.next.checked_add(1)
            else {
                return E_ABORT;
            };

            state.next = next;
            state.pending.insert(next, (port, task));

            next
        };

        if delay_ms == 0 {
            self.deliver(id, port);
        } else {
            shared_timer().schedule(
                self, id, port, delay_ms
            );
        }

        S_OK
    }

    fn deliver(self: &Arc<Self>, id: u64, port: u32) {
        {
            let mut state = lock(&self.state);

            if !state.pending.contains_key(&id) {
                return;
            }

            // Count delivery and monitor activity as well
            // as execution of the actual callback.
            state.active[port as usize] += 1;
        }

        let _activity = Activity(self,port);

        self.ports[port as usize].submit(Tracked {
            owner: self.clone(),
            kind: DispatchKind::Normal(id),
        });

        // Never call application code with queue locks held.
        let monitors = lock(&self.monitors).clone();

        for (_, context, callback) in monitors {
            // SAFETY: The address was stored from a MonitorCallback during registration.
            let callback: MonitorCallback = unsafe {
                std::mem::transmute(callback)
            };

            callback(
                context as *mut c_void,
                Arc::as_ptr(self),
                port,
            );
        }
    }


    fn wait_idle(&self, port:u32) {
        let state=lock(&self.state);

        drop(
            self.changed.wait_while(
                state,
                |s|s.active[port as usize]!=0
            ).unwrap_or_else(|e|e.into_inner())
        );
    }

    fn cancel_stage(
        self:&Arc<Self>,
        port:u32,
        pending:&std::collections::BTreeMap<
            u64,(u32,Task)
        >,
    ) {
        let count=pending.values()
            .filter(|(p,_)|*p==port)
            .count();

        let stage=Stage::new(count);

        for &(p,task) in pending.values() {
            if p==port {
                self.ports[port as usize].submit(
                    Tracked {
                        owner:self.clone(),
                        kind:DispatchKind::Cancel(
                            task,stage.clone()
                        ),
                    }
                );
            }
        }

        stage.wait();

        // Existing work already claimed by another
        // dispatcher must also finish.
        self.wait_idle(port);
    }

    fn discard_tokens(self: &Arc<Self>) {
        for port in &self.ports {
            port.discard_owner(self);
        }

        POOL.discard_owner(self);
    }

    fn terminate(
        self: &Arc<Self>,
        wait: bool,
        context: *mut c_void,
        callback: Option<TerminatedCallback>,
    ) -> HRESULT {
        let pending = {
            let mut state = lock(&self.state);

            if self.terminated.swap(
                true, Ordering::AcqRel
            ) {
                // Experimental repeated-termination policy.
                // Full GDK behavior needs separate validation.
                return E_ABORT;
            }

            let pending = std::mem::take(
                &mut state.pending
            );

            // Wake delay workers whose tasks were canceled.
            self.changed.notify_all();

            pending
        };

        // Wake dispatchers and remove only this queue's
        // tokens, not those of composite siblings.
        if let Some(timer)=SHARED_TIMER.get() {
            timer.discard_owner(self);
        }
        self.discard_tokens();

        let owner = self.clone();
        let context = context as usize;


        let finish=move || {
            // Cancel through the actual dispatch ports.
            owner.cancel_stage(WORK,&pending);
            owner.cancel_stage(COMPLETION,&pending);

            // No application notification before
            // both stages have completed.
            if let Some(callback)=callback {
                let stage=Stage::new(1);

                owner.ports[COMPLETION as usize]
                    .submit(Tracked {
                        owner:owner.clone(),
                        kind:DispatchKind::Notify(
                            context,
                            callback,
                            stage.clone(),
                        ),
                    });

                stage.wait();
            }

            {
                let mut state=lock(&owner.state);

                state.termination_complete=true;

                owner.finished.store(
                    true,Ordering::Release
                );

                owner.changed.notify_all();
            }

            // Remove stale tokens and wake any
            // dispatchers waiting for termination.
            owner.discard_tokens();
        };

        if wait {
            finish();
        } else {
            std::thread::spawn(finish);
        }

        S_OK
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The process queue: `None` until first used (then a thread-pool queue),
/// `Some(None)` once the title clears it.
static PROCESS_QUEUE: Mutex<Option<Option<Arc<Queue>>>> = Mutex::new(None);

fn process_queue() -> Option<Arc<Queue>> {
    lock(&PROCESS_QUEUE)
        .get_or_insert_with(|| {
            Some(Queue::new([
                Port::new(Mode::ThreadPool),
                Port::new(Mode::ThreadPool),
            ]))
        })
        .clone()
}

/// A new reference to the queue behind `handle` (null: the process queue).
fn queue(handle: *const Queue) -> Option<Arc<Queue>> {
    if handle.is_null() {
        return process_queue();
    }
    // SAFETY: a handle we returned, which the title has not closed.
    unsafe {
        Arc::increment_strong_count(handle);
        Some(Arc::from_raw(handle))
    }
}

pub fn XTaskQueueCreate(work: u32, completion: u32, out: *mut *const Queue) -> HRESULT {
    let (Some(work), Some(completion)) = (Mode::from_raw(work), Mode::from_raw(completion)) else {
        return E_INVALIDARG;
    };
    if out.is_null() {
        return E_INVALIDARG;
    }
    let queue = Queue::new([Port::new(work), Port::new(completion)]);
    // SAFETY: checked non-null; the title passes room for a handle.
    unsafe { out.write(Arc::into_raw(queue)) };
    S_OK
}

/// A port handle points at the port, which its queue keeps alive.
pub fn XTaskQueueGetPort(handle: *const Queue, port: u32, out: *mut *const c_void) -> HRESULT {
    let Some(queue) = queue(handle) else {
        return E_INVALIDARG;
    };
    if port > COMPLETION || out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null.
    unsafe { out.write(Arc::as_ptr(&queue.ports[port as usize]).cast()) };
    S_OK
}

pub fn XTaskQueueCreateComposite(
    work: *const c_void,
    completion: *const c_void,
    out: *mut *const Queue,
) -> HRESULT {
    if work.is_null() || completion.is_null() || out.is_null() {
        return E_INVALIDARG;
    }
    let port = |p: *const c_void| {
        // SAFETY: a port handle from `XTaskQueueGetPort`, whose queue is open.
        unsafe {
            Arc::increment_strong_count(p.cast::<Port>());
            Arc::from_raw(p.cast::<Port>())
        }
    };
    let queue = Queue::new([port(work), port(completion)]);
    // SAFETY: checked non-null.
    unsafe { out.write(Arc::into_raw(queue)) };
    S_OK
}

pub fn XTaskQueueDuplicateHandle(handle: *const Queue, out: *mut *const Queue) -> HRESULT {
    match queue(handle) {
        Some(queue) if !out.is_null() => {
            // SAFETY: checked non-null.
            unsafe { out.write(Arc::into_raw(queue)) };
            S_OK
        }
        _ => E_INVALIDARG,
    }
}

pub fn XTaskQueueCloseHandle(handle: *const Queue) {
    if !handle.is_null() {
        // SAFETY: a handle we returned; this drops its reference.
        unsafe { Arc::decrement_strong_count(handle) };
    }
}

/// Runs one queued callback of a manual port; false if none came in time,
/// or none can come: the queue was terminated (which ends a wait under way,
/// as a dispatcher thread's loop expects).
pub fn XTaskQueueDispatch(handle: *const Queue, port: u32, timeout_ms: u32) -> bool {
    let Some(queue) = queue(handle) else {
        return false;
    };
    let Some(port) = queue.ports.get(port as usize) else {
        return false;
    };
    if port.mode != Mode::Manual {
        return false;
    }
    match port.take(timeout_ms, || queue.finished.load(Ordering::Acquire)) {
        Some(task) => {
            task.run();
            true
        }
        None => false,
    }
}

pub fn XTaskQueueSubmitDelayedCallback(
    handle: *const Queue,
    port: u32,
    delay_ms: u32,
    context: *mut c_void,
    callback: Option<Callback>,
) -> HRESULT {
    match (queue(handle), callback) {
        (Some(queue), Some(callback)) => queue.submit(port, delay_ms, Task::new(context, callback)),
        _ => E_INVALIDARG,
    }
}

/// Cancels what is still queued (callbacks see `canceled`), refuses new
/// work, then calls `callback`.
pub fn XTaskQueueTerminate(
    handle: *const Queue,
    wait: bool,
    context: *mut c_void,
    callback: Option<TerminatedCallback>,
) -> HRESULT {
    let Some(queue) = queue(handle) else {
        return E_INVALIDARG;
    };

    queue.terminate(wait, context, callback)
}

static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);

pub fn XTaskQueueRegisterMonitor(
    handle: *const Queue,
    context: *mut c_void,
    callback: Option<MonitorCallback>,
    token: *mut u64,
) -> HRESULT {
    let (Some(queue), Some(callback)) = (queue(handle), callback) else {
        return E_INVALIDARG;
    };
    if token.is_null() {
        return E_INVALIDARG;
    }
    let id = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed);
    lock(&queue.monitors).push((id, context as usize, callback as usize));
    // SAFETY: checked non-null.
    unsafe { token.write(id) };
    S_OK
}

pub fn XTaskQueueUnregisterMonitor(handle: *const Queue, token: u64) {
    if let Some(queue) = queue(handle) {
        lock(&queue.monitors).retain(|m| m.0 != token);
    }
}

/// A new handle to the process queue, if there is one.
pub fn XTaskQueueGetCurrentProcessTaskQueue(out: *mut *const Queue) -> bool {
    let found = process_queue();
    if !out.is_null() {
        // SAFETY: checked non-null.
        unsafe { out.write(found.clone().map_or(std::ptr::null(), Arc::into_raw)) };
    }
    found.is_some()
}

pub fn XTaskQueueSetCurrentProcessTaskQueue(handle: *const Queue) {
    let new = if handle.is_null() {
        None
    } else {
        queue(handle)
    };
    *lock(&PROCESS_QUEUE) = Some(new);
}

/// An async block: what the title fills in, then the system area.
#[repr(C)]
pub struct AsyncBlock {
    pub queue: *const Queue,
    pub context: *mut c_void,
    pub callback: Option<CompletionRoutine>,
    /// Ours: [unused, status, result size, unused].
    pub internal: [usize; 4],
}

#[repr(C)]
pub struct ProviderData {
    pub block: *mut AsyncBlock,
    pub buffer_size: usize,
    pub buffer: *mut c_void,
    pub context: *mut c_void,
}

const OP_BEGIN: u32 = 0;
const OP_DO_WORK: u32 = 1;
const OP_GET_RESULT: u32 = 2;
const OP_CANCEL: u32 = 3;
const OP_CLEANUP: u32 = 4;

/// One call in flight, until its result is taken or it fails.
struct Call {
    provider: Provider,
    context: usize,
    identity: usize,
    done: Mutex<bool>,
    finished: Condvar,
}

impl Call {
    fn provider(
        &self,
        op: u32,
        block: *mut AsyncBlock,
        size: usize,
        buffer: *mut c_void,
    ) -> HRESULT {
        let data = ProviderData {
            block,
            buffer_size: size,
            buffer,
            context: self.context as *mut c_void,
        };
        (self.provider)(op, &data)
    }
}

/// Calls by block address.
static CALLS: LazyLock<Mutex<HashMap<usize, Arc<Call>>>> = LazyLock::new(Default::default);

fn call(block: *mut AsyncBlock) -> Option<Arc<Call>> {
    lock(&CALLS).get(&(block as usize)).cloned()
}

/// The block's status and result-size words.
fn words(block: *mut AsyncBlock) -> (&'static AtomicUsize, &'static AtomicUsize) {
    // SAFETY: the title's block outlives the call; its system area is ours.
    unsafe {
        let internal = std::ptr::addr_of_mut!((*block).internal).cast::<AtomicUsize>();
        (&*internal.add(1), &*internal.add(2))
    }
}

fn status(block: *mut AsyncBlock) -> HRESULT {
    words(block).0.load(Ordering::Acquire) as u32 as i32
}

fn release(block: *mut AsyncBlock) {
    // Unlocked before the provider runs: it may call back into us.
    let removed = lock(&CALLS).remove(&(block as usize));
    if let Some(call) = removed {
        call.provider(OP_CLEANUP, block, 0, std::ptr::null_mut());
    }
}

pub fn XAsyncBegin(
    block: *mut AsyncBlock,
    context: *mut c_void,
    identity: *const c_void,
    _identity_name: *const u8,
    provider: Option<Provider>,
) -> HRESULT {
    let Some(provider) = provider else {
        return E_INVALIDARG;
    };
    if block.is_null() || call(block).is_some() {
        return E_INVALIDARG;
    }
    let (status, size) = words(block);
    status.store(E_PENDING as u32 as usize, Ordering::Release);
    size.store(0, Ordering::Release);
    let call = Arc::new(Call {
        provider,
        context: context as usize,
        identity: identity as usize,
        done: Mutex::new(false),
        finished: Condvar::new(),
    });
    lock(&CALLS).insert(block as usize, call.clone());
    // A failed begin fails the call, so the completion routine still runs.
    let hr = call.provider(OP_BEGIN, block, 0, std::ptr::null_mut());
    if hr < 0 {
        XAsyncComplete(block, hr, 0);
    }
    S_OK
}

/// Queues the provider's `DoWork` on the block's work port.
pub fn XAsyncSchedule(block: *mut AsyncBlock, delay_ms: u32) -> HRESULT {
    extern "system" fn work(block: *mut c_void, canceled: bool) {
        let block = block.cast::<AsyncBlock>();
        let Some(call) = call(block) else { return };
        let hr = if canceled {
            E_ABORT
        } else {
            call.provider(OP_DO_WORK, block, 0, std::ptr::null_mut())
        };
        if hr != E_PENDING {
            XAsyncComplete(block, hr, 0);
        }
    }
    if block.is_null() || call(block).is_none() {
        return E_INVALIDARG;
    }
    // SAFETY: a block in flight is valid.
    let handle = unsafe { (*block).queue };
    match queue(handle) {
        Some(queue) => queue.submit(WORK, delay_ms, Task::new(block.cast(), work)),
        None => E_INVALIDARG,
    }
}

/// Records the result (the first one wins), wakes waiters and queues the
/// completion routine on the block's completion port.
pub fn XAsyncComplete(block: *mut AsyncBlock, result: HRESULT, required_size: usize) {
    extern "system" fn complete(block: *mut c_void, _canceled: bool) {
        let block = block.cast::<AsyncBlock>();
        // SAFETY: read before the status was published; see below.
        if let Some(routine) = unsafe { (*block).callback } {
            routine(block);
        }
    }
    let Some(call) = call(block) else { return };
    if result == E_PENDING {
        return;
    }
    let mut done = lock(&call.done);
    if *done {
        return;
    }
    // Read before publishing: once the status is set, the title may free
    // the block.
    // SAFETY: a block in flight is valid.
    let (handle, routine) = unsafe { ((*block).queue, (*block).callback) };
    let (status, size) = words(block);
    if result < 0 || required_size == 0 {
        release(block);
    }
    size.store(required_size, Ordering::Release);
    status.store(result as u32 as usize, Ordering::Release);
    *done = true;
    call.finished.notify_all();
    drop(done);
    if routine.is_some() {
        let task = Task::new(block.cast(), complete);
        match queue(handle) {
            Some(queue) if queue.submit(COMPLETION, 0, task) == S_OK => {}
            _ => task.run(true),
        }
    }
}

pub fn XAsyncGetStatus(block: *mut AsyncBlock, wait: bool) -> HRESULT {
    if block.is_null() {
        return E_INVALIDARG;
    }
    if wait && let Some(call) = call(block) {
        let done = lock(&call.done);
        drop(call.finished.wait_while(done, |d| !*d));
    }
    status(block)
}

pub fn XAsyncGetResultSize(block: *mut AsyncBlock, out: *mut usize) -> HRESULT {
    if block.is_null() || out.is_null() {
        return E_INVALIDARG;
    }
    let hr = status(block);
    // Any nonnegative completion carries a result, not just S_OK: S_FALSE
    // is a success code too and a provider may return data with it.
    if hr >= 0 {
        // SAFETY: checked non-null.
        unsafe { out.write(words(block).1.load(Ordering::Acquire)) };
    }
    hr
}

/// Asks the provider to cancel; the call fails with `E_ABORT` unless the
/// provider completed it first.
pub fn XAsyncCancel(block: *mut AsyncBlock) {
    if let Some(call) = call(block) {
        call.provider(OP_CANCEL, block, 0, std::ptr::null_mut());
        XAsyncComplete(block, E_ABORT, 0);
    }
}

/// Has the provider write the result into `buffer`, then ends the call.
pub fn XAsyncGetResult(
    block: *mut AsyncBlock,
    identity: *const c_void,
    buffer_size: usize,
    buffer: *mut c_void,
    used: *mut usize,
) -> HRESULT {
    if block.is_null() {
        return E_INVALIDARG;
    }
    let hr = status(block);
    let size = words(block).1.load(Ordering::Acquire);
    if hr < 0 || size == 0 {
        if hr >= 0 && !used.is_null() {
            // SAFETY: checked non-null.
            unsafe { used.write(0) };
        }
        return hr;
    }
    let Some(call) = call(block) else {
        return E_ILLEGAL_METHOD_CALL;
    };
    if call.identity != identity as usize {
        return E_INVALIDARG;
    }
    if buffer_size < size || buffer.is_null() {
        return E_NOT_SUFFICIENT_BUFFER;
    }
    let hr = call.provider(OP_GET_RESULT, block, buffer_size, buffer);
    if !used.is_null() {
        // SAFETY: checked non-null.
        unsafe { used.write(size) };
    }
    release(block);
    hr
}

/// Starts a call on `block` that is already over: it completes with `hr`
/// and no result data, through the block's completion port as usual.
pub fn finished(block: *mut AsyncBlock, identity: *const c_void, hr: HRESULT) -> HRESULT {
    finished_with(
        block,
        identity,
        if hr < 0 { Err(hr) } else { Ok(Vec::new()) },
    )
}

/// Like [`finished`], with the bytes `XAsyncGetResult` hands out for
/// `identity` (an error has none).
pub fn finished_with(
    block: *mut AsyncBlock,
    identity: *const c_void,
    result: Result<Vec<u8>, HRESULT>,
) -> HRESULT {
    type Data = Result<Vec<u8>, HRESULT>;
    extern "system" fn provider(op: u32, data: *const ProviderData) -> HRESULT {
        // SAFETY: the runtime passes valid provider data; the context is the
        // boxed result until cleanup frees it.
        let data = unsafe { &*data };
        let result = data.context.cast::<Data>();
        match op {
            // SAFETY: boxed in `finished_with`; nothing runs after cleanup.
            OP_CLEANUP => drop(unsafe { Box::from_raw(result) }),
            OP_BEGIN => {
                // SAFETY: as above. Copied out: completing may clean up.
                let (hr, len) = match unsafe { &*result } {
                    Ok(bytes) => (S_OK, bytes.len()),
                    Err(hr) => (*hr, 0),
                };
                XAsyncComplete(data.block, hr, len);
            }
            // SAFETY: as above; the runtime checked the buffer's size.
            OP_GET_RESULT => unsafe {
                if let Ok(bytes) = &*result {
                    data.buffer
                        .cast::<u8>()
                        .copy_from(bytes.as_ptr(), bytes.len());
                }
            },
            _ => {}
        }
        S_OK
    }
    let context = Box::into_raw(Box::new(result));
    let hr = XAsyncBegin(
        block,
        context.cast(),
        identity,
        std::ptr::null(),
        Some(provider),
    );
    if hr < 0 {
        // Refused before the call existed, so no cleanup will come.
        // SAFETY: still ours.
        drop(unsafe { Box::from_raw(context) });
    }
    hr
}

/// Like [`finished_with`], but the result comes from `work`, run on a
/// thread of ours; the call stays pending until it returns (a dialog, say).
pub fn pending_with(
    block: *mut AsyncBlock,
    identity: *const c_void,
    work: impl FnOnce() -> Result<Vec<u8>, HRESULT> + Send + 'static,
) -> HRESULT {
    type Data = Mutex<(Option<Box<dyn FnOnce() -> Result<Vec<u8>, HRESULT> + Send>>, Vec<u8>)>;
    extern "system" fn provider(op: u32, data: *const ProviderData) -> HRESULT {
        // SAFETY: the runtime passes valid provider data; the context is the
        // boxed state until cleanup frees it.
        let data = unsafe { &*data };
        let state = data.context.cast::<Data>();
        match op {
            // SAFETY: boxed in `pending_with`; nothing runs after cleanup.
            OP_CLEANUP => drop(unsafe { Box::from_raw(state) }),
            OP_BEGIN => {
                // SAFETY: as above; cleanup waits for completion.
                let work = unsafe { &*state }.lock().unwrap_or_else(|e| e.into_inner()).0.take();
                let (block, state) = (data.block as usize, state as usize);
                std::thread::spawn(move || {
                    let result = work.map_or(Err(E_ABORT), |work| work());
                    let (hr, len) = match &result {
                        Ok(bytes) => (S_OK, bytes.len()),
                        Err(hr) => (*hr, 0),
                    };
                    if let Ok(bytes) = result {
                        // SAFETY: the call is pending, so its state lives.
                        let state = unsafe { &*(state as *const Data) };
                        state.lock().unwrap_or_else(|e| e.into_inner()).1 = bytes;
                    }
                    XAsyncComplete(block as *mut AsyncBlock, hr, len);
                });
            }
            // SAFETY: as above; the runtime checked the buffer's size.
            OP_GET_RESULT => unsafe {
                let state = (*state).lock().unwrap_or_else(|e| e.into_inner());
                data.buffer
                    .cast::<u8>()
                    .copy_from(state.1.as_ptr(), state.1.len());
            },
            _ => {}
        }
        S_OK
    }
    let state: Box<Data> = Box::new(Mutex::new((Some(Box::new(work)), Vec::new())));
    let context = Box::into_raw(state);
    let hr = XAsyncBegin(block, context.cast(), identity, std::ptr::null(), Some(provider));
    if hr < 0 {
        // SAFETY: refused before the call existed; still ours.
        drop(unsafe { Box::from_raw(context) });
    }
    hr
}

/// The result of a call begun with [`finished_with`] for `identity`, as a
/// `T` written through `out`.
pub fn result<T>(block: *mut AsyncBlock, identity: *const c_void, out: *mut T) -> HRESULT {
    XAsyncGetResult(
        block,
        identity,
        size_of::<T>(),
        out.cast(),
        std::ptr::null_mut(),
    )
}

/// The bytes of a value, for [`finished_with`].
pub fn bytes<T: Copy>(value: T) -> Vec<u8> {
    // SAFETY: a `Copy` value is plain data; padding bytes are never read.
    unsafe { std::slice::from_raw_parts((&raw const value).cast::<u8>(), size_of::<T>()) }.to_vec()
}

type Work = extern "system" fn(*mut AsyncBlock) -> HRESULT;

/// Runs `work` on the block's work port as an async call of its own; its
/// return value is the call's result.
pub fn XAsyncRun(block: *mut AsyncBlock, work: Option<Work>) -> HRESULT {
    extern "system" fn provider(op: u32, data: *const ProviderData) -> HRESULT {
        // SAFETY: the runtime passes valid provider data.
        let data = unsafe { &*data };
        match op {
            OP_BEGIN => XAsyncSchedule(data.block, 0),
            OP_DO_WORK => {
                // SAFETY: the context is the `Work` given to `XAsyncRun`.
                let work = unsafe { std::mem::transmute::<*mut c_void, Work>(data.context) };
                work(data.block)
            }
            _ => S_OK,
        }
    }
    let Some(work) = work else {
        return E_INVALIDARG;
    };
    let identity = XAsyncRun as *const c_void;
    XAsyncBegin(
        block,
        work as *mut c_void,
        identity,
        std::ptr::null(),
        Some(provider),
    )
}

// The object `QueryApiImpl` returns. Each method takes the object first,
// then the API's arguments; slot offsets as titles call them.

pub static OBJECT: Object = Object::new("task queue", &VTABLE);

/// Its interface id is the provider's.
pub const INTERFACES: &[Interface] = &[(PROVIDER, PROVIDER, &OBJECT)];

type This = *const Object;
extern "system" fn m_get_status(_: This, b: *mut AsyncBlock, wait: u8) -> HRESULT {
    XAsyncGetStatus(b, wait != 0)
}
extern "system" fn m_get_result_size(_: This, b: *mut AsyncBlock, out: *mut usize) -> HRESULT {
    XAsyncGetResultSize(b, out)
}
extern "system" fn m_cancel(_: This, b: *mut AsyncBlock) {
    XAsyncCancel(b)
}
extern "system" fn m_run(_: This, b: *mut AsyncBlock, work: Option<Work>) -> HRESULT {
    XAsyncRun(b, work)
}
extern "system" fn m_begin(
    _: This,
    b: *mut AsyncBlock,
    context: *mut c_void,
    identity: *const c_void,
    name: *const u8,
    provider: Option<Provider>,
) -> HRESULT {
    XAsyncBegin(b, context, identity, name, provider)
}
extern "system" fn m_schedule(_: This, b: *mut AsyncBlock, delay: u32) -> HRESULT {
    XAsyncSchedule(b, delay)
}
extern "system" fn m_complete(_: This, b: *mut AsyncBlock, hr: HRESULT, size: usize) {
    XAsyncComplete(b, hr, size)
}
extern "system" fn m_get_result(
    _: This,
    b: *mut AsyncBlock,
    identity: *const c_void,
    size: usize,
    buffer: *mut c_void,
    used: *mut usize,
) -> HRESULT {
    XAsyncGetResult(b, identity, size, buffer, used)
}
extern "system" fn m_create(
    _: This,
    work: u32,
    completion: u32,
    out: *mut *const Queue,
) -> HRESULT {
    XTaskQueueCreate(work, completion, out)
}
extern "system" fn m_create_composite(
    _: This,
    work: *const c_void,
    completion: *const c_void,
    out: *mut *const Queue,
) -> HRESULT {
    XTaskQueueCreateComposite(work, completion, out)
}
extern "system" fn m_get_port(
    _: This,
    q: *const Queue,
    port: u32,
    out: *mut *const c_void,
) -> HRESULT {
    XTaskQueueGetPort(q, port, out)
}
extern "system" fn m_duplicate(_: This, q: *const Queue, out: *mut *const Queue) -> HRESULT {
    XTaskQueueDuplicateHandle(q, out)
}
extern "system" fn m_dispatch(_: This, q: *const Queue, port: u32, timeout: u32) -> bool {
    XTaskQueueDispatch(q, port, timeout)
}
extern "system" fn m_close(_: This, q: *const Queue) {
    XTaskQueueCloseHandle(q)
}
extern "system" fn m_submit(
    _: This,
    q: *const Queue,
    port: u32,
    context: *mut c_void,
    callback: Option<Callback>,
) -> HRESULT {
    XTaskQueueSubmitDelayedCallback(q, port, 0, context, callback)
}
extern "system" fn m_submit_delayed(
    _: This,
    q: *const Queue,
    port: u32,
    delay: u32,
    context: *mut c_void,
    callback: Option<Callback>,
) -> HRESULT {
    XTaskQueueSubmitDelayedCallback(q, port, delay, context, callback)
}
extern "system" fn m_terminate(
    _: This,
    q: *const Queue,
    wait: u8,
    context: *mut c_void,
    callback: Option<TerminatedCallback>,
) -> HRESULT {
    XTaskQueueTerminate(q, wait != 0, context, callback)
}
extern "system" fn m_register_monitor(
    _: This,
    q: *const Queue,
    context: *mut c_void,
    callback: Option<MonitorCallback>,
    token: *mut u64,
) -> HRESULT {
    XTaskQueueRegisterMonitor(q, context, callback, token)
}
extern "system" fn m_unregister_monitor(_: This, q: *const Queue, token: u64) {
    XTaskQueueUnregisterMonitor(q, token)
}
extern "system" fn m_get_process_queue(_: This, out: *mut *const Queue) -> bool {
    XTaskQueueGetCurrentProcessTaskQueue(out)
}
extern "system" fn m_set_process_queue(_: This, q: *const Queue) {
    XTaskQueueSetCurrentProcessTaskQueue(q)
}
/// The runtime's debug check that the caller is not a time-critical thread,
/// which every runtime call in a title makes first.
extern "system" fn m_check_thread(_: This) {}

static VTABLE: Methods<31> = Methods([
    missing::<0> as *const (),         // 0x00 QueryInterface
    add_ref as *const (),              // 0x08 AddRef
    add_ref as *const (),              // 0x10 Release: the object is static
    m_get_status as *const (),         // 0x18
    m_get_result_size as *const (),    // 0x20
    m_cancel as *const (),             // 0x28
    m_run as *const (),                // 0x30
    m_begin as *const (),              // 0x38
    missing::<8> as *const (),         // 0x40
    m_schedule as *const (),           // 0x48
    m_complete as *const (),           // 0x50
    m_get_result as *const (),         // 0x58
    m_create as *const (),             // 0x60
    m_create_composite as *const (),   // 0x68
    m_get_port as *const (),           // 0x70
    m_duplicate as *const (),          // 0x78
    m_dispatch as *const (),           // 0x80
    m_close as *const (),              // 0x88
    m_submit as *const (),             // 0x90
    m_submit_delayed as *const (),     // 0x98
    missing::<20> as *const (),        // 0xa0 register a wait handle
    missing::<21> as *const (),        // 0xa8 unregister it
    m_terminate as *const (),          // 0xb0
    m_register_monitor as *const (),   // 0xb8
    m_unregister_monitor as *const (), // 0xc0
    m_get_process_queue as *const (),  // 0xc8
    m_set_process_queue as *const (),  // 0xd0
    missing::<27> as *const (),        // 0xd8 thread time-sensitivity
    missing::<28> as *const (),        // 0xe0
    m_check_thread as *const (),       // 0xe8
    missing::<30> as *const (),        // 0xf0
]);

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    fn slot<F: Copy>(offset: usize) -> F {
        crate::modules::xgameruntime::method(&OBJECT, offset)
    }

    static RAN: AtomicU32 = AtomicU32::new(0);
    extern "system" fn count(context: *mut c_void, canceled: bool) {
        RAN.fetch_add(
            if canceled { 100 } else { context as u32 },
            Ordering::SeqCst,
        );
    }

    #[test]
    fn manual_ports_run_callbacks_when_dispatched() {
        type Create = extern "system" fn(This, u32, u32, *mut *const Queue) -> HRESULT;
        type Submit = extern "system" fn(This, *const Queue, u32, *mut c_void, Callback) -> HRESULT;
        type Dispatch = extern "system" fn(This, *const Queue, u32, u32) -> bool;
        type Close = extern "system" fn(This, *const Queue);
        let this: This = &OBJECT;
        let mut q = std::ptr::null();
        assert_eq!(slot::<Create>(0x60)(this, 0, 0, &mut q), S_OK);
        assert_eq!(
            slot::<Submit>(0x90)(this, q, COMPLETION, 5 as *mut c_void, count),
            S_OK
        );
        assert_eq!(RAN.load(Ordering::SeqCst), 0, "manual: nothing runs yet");
        assert!(
            !slot::<Dispatch>(0x80)(this, q, WORK, 0),
            "work port is empty"
        );
        assert!(slot::<Dispatch>(0x80)(this, q, COMPLETION, 0));
        assert_eq!(RAN.load(Ordering::SeqCst), 5);
        assert!(!slot::<Dispatch>(0x80)(this, q, COMPLETION, 0));
        slot::<Close>(0x88)(this, q);
    }

    /// A provider that schedules its work and returns a u64 result.
    extern "system" fn provider(op: u32, data: *const ProviderData) -> HRESULT {
        // SAFETY: the runtime passes valid provider data.
        let data = unsafe { &*data };
        match op {
            OP_BEGIN => XAsyncSchedule(data.block, 0),
            OP_DO_WORK => {
                XAsyncComplete(data.block, S_OK, 8);
                S_OK
            }
            OP_GET_RESULT => {
                // SAFETY: at least 8 bytes, checked by the runtime.
                unsafe { data.buffer.cast::<u64>().write(data.context as u64) };
                S_OK
            }
            _ => S_OK,
        }
    }

    static COMPLETED: AtomicU32 = AtomicU32::new(0);
    extern "system" fn completed(_: *mut AsyncBlock) {
        COMPLETED.fetch_add(1, Ordering::SeqCst);
    }

    #[test]
    fn async_calls_complete_and_return_results() {
        let mut q = std::ptr::null();
        // Work on a thread pool, completion dispatched by hand.
        assert_eq!(XTaskQueueCreate(1, 0, &mut q), S_OK);
        let mut block = AsyncBlock {
            queue: q,
            context: std::ptr::null_mut(),
            callback: Some(completed),
            internal: [0; 4],
        };
        let identity = 0x1d as *const c_void;
        assert_eq!(
            XAsyncBegin(
                &mut block,
                42 as *mut c_void,
                identity,
                std::ptr::null(),
                Some(provider)
            ),
            S_OK
        );
        assert_eq!(XAsyncGetStatus(&mut block, true), S_OK);
        let mut size = 0;
        assert_eq!(XAsyncGetResultSize(&mut block, &mut size), S_OK);
        assert_eq!(size, 8);
        assert_eq!(COMPLETED.load(Ordering::SeqCst), 0, "completion is manual");
        assert!(XTaskQueueDispatch(q, COMPLETION, INFINITE));
        assert_eq!(COMPLETED.load(Ordering::SeqCst), 1);

        let mut value = 0u64;
        let mut small = 0u32;
        let out = (&mut small as *mut u32).cast();
        assert_eq!(
            XAsyncGetResult(&mut block, identity, 4, out, std::ptr::null_mut()),
            E_NOT_SUFFICIENT_BUFFER
        );
        let out = (&mut value as *mut u64).cast();
        let mut used = 0;
        assert_eq!(
            XAsyncGetResult(&mut block, identity, 8, out, &mut used),
            S_OK
        );
        assert_eq!((value, used), (42, 8));
        assert!(call(&mut block).is_none(), "the call ends with its result");
        XTaskQueueCloseHandle(q);
    }

    #[test]
    fn a_nonnegative_completion_still_returns_its_payload() {
        const S_FALSE: HRESULT = 1;
        extern "system" fn provider(op: u32, data: *const ProviderData) -> HRESULT {
            // SAFETY: the runtime passes valid provider data.
            let data = unsafe { &*data };
            match op {
                OP_BEGIN => XAsyncSchedule(data.block, 0),
                OP_DO_WORK => {
                    XAsyncComplete(data.block, S_FALSE, 8);
                    S_OK
                }
                OP_GET_RESULT => {
                    // SAFETY: the caller checked eight bytes.
                    unsafe { data.buffer.cast::<u64>().write(0x1234) };
                    S_OK
                }
                _ => S_OK,
            }
        }
        let mut q = std::ptr::null();
        assert_eq!(XTaskQueueCreate(0, 0, &mut q), S_OK);
        let mut block = AsyncBlock {
            queue: q,
            context: std::ptr::null_mut(),
            callback: None,
            internal: [0; 4],
        };
        let identity = 0x77 as *const c_void;
        assert_eq!(
            XAsyncBegin(&mut block, std::ptr::null_mut(), identity, std::ptr::null(), Some(provider)),
            S_OK
        );
        assert!(XTaskQueueDispatch(q, WORK, 0));
        assert_eq!(XAsyncGetStatus(&mut block, false), S_FALSE);
        let mut size = 0;
        assert_eq!(XAsyncGetResultSize(&mut block, &mut size), S_FALSE);
        assert_eq!(size, 8);
        let mut value = 0u64;
        let mut used = 0;
        assert_eq!(
            XAsyncGetResult(&mut block, identity, 8, (&mut value as *mut u64).cast(), &mut used),
            S_OK
        );
        assert_eq!((value, used), (0x1234, 8));
        assert!(call(&raw mut block).is_none(), "the call ends with its result");
        XTaskQueueCloseHandle(q);
    }

    #[test]
    fn run_returns_the_work_result() {
        extern "system" fn work(_: *mut AsyncBlock) -> HRESULT {
            E_ABORT
        }
        let mut block = AsyncBlock {
            queue: std::ptr::null(),
            context: std::ptr::null_mut(),
            callback: None,
            internal: [0; 4],
        };
        assert_eq!(
            slot::<extern "system" fn(This, *mut AsyncBlock, Option<Work>) -> HRESULT>(0x30)(
                &OBJECT,
                &mut block,
                Some(work)
            ),
            S_OK
        );
        assert_eq!(XAsyncGetStatus(&mut block, true), E_ABORT);
    }

    #[test]
    fn pool_callbacks_may_wait_for_each_other() {
        static SECOND: AtomicBool = AtomicBool::new(false);
        static FIRST: AtomicBool = AtomicBool::new(false);
        extern "system" fn waits(_: *mut c_void, _: bool) {
            let start = std::time::Instant::now();
            while !SECOND.load(Ordering::SeqCst) && start.elapsed().as_secs() < 5 {
                std::thread::yield_now();
            }
            FIRST.store(SECOND.load(Ordering::SeqCst), Ordering::SeqCst);
        }
        extern "system" fn releases(_: *mut c_void, _: bool) {
            SECOND.store(true, Ordering::SeqCst);
        }
        let mut q = std::ptr::null();
        assert_eq!(XTaskQueueCreate(1, 1, &mut q), S_OK);
        let none = std::ptr::null_mut();
        assert_eq!(
            XTaskQueueSubmitDelayedCallback(q, WORK, 0, none, Some(waits)),
            S_OK
        );
        assert_eq!(
            XTaskQueueSubmitDelayedCallback(q, WORK, 0, none, Some(releases)),
            S_OK
        );
        let start = std::time::Instant::now();
        while !FIRST.load(Ordering::SeqCst) && start.elapsed().as_secs() < 10 {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(FIRST.load(Ordering::SeqCst));
        XTaskQueueCloseHandle(q);
    }

    /// A dispatcher thread's loop, `while (XTaskQueueDispatch(queue, port,
    /// INFINITE));`, ends when its queue is terminated: nothing can be
    /// submitted to it any more, so nothing else would ever wake it.
    #[test]
    fn terminate_ends_a_waiting_dispatch() {
        static RAN: AtomicU32 = AtomicU32::new(0);
        extern "system" fn ran(_: *mut c_void, _: bool) {
            RAN.fetch_add(1, Ordering::SeqCst);
        }
        /// Dispatches `queue`'s work until a dispatch returns false, then
        /// says so.
        fn dispatcher(queue: *const Queue) -> std::sync::mpsc::Receiver<()> {
            let (done, ended) = std::sync::mpsc::channel();
            let queue = queue as usize;
            std::thread::spawn(move || {
                while XTaskQueueDispatch(queue as *const Queue, WORK, INFINITE) {}
                let _ = done.send(());
            });
            ended
        }
        let run_one = |queue, total| {
            let none = std::ptr::null_mut();
            assert_eq!(
                XTaskQueueSubmitDelayedCallback(queue, WORK, 0, none, Some(ran)),
                S_OK
            );
            for _ in 0..500 {
                if RAN.load(Ordering::SeqCst) == total {
                    return;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            panic!("callback {total} did not run");
        };
        let (long, short) = (Duration::from_secs(5), Duration::from_millis(200));
        let none = std::ptr::null_mut();

        // Two queues on the same manual ports.
        let (mut a, mut b) = (std::ptr::null(), std::ptr::null());
        assert_eq!(XTaskQueueCreate(0, 0, &mut a), S_OK);
        let (mut work, mut completion) = (std::ptr::null(), std::ptr::null());
        assert_eq!(XTaskQueueGetPort(a, WORK, &mut work), S_OK);
        assert_eq!(XTaskQueueGetPort(a, COMPLETION, &mut completion), S_OK);
        assert_eq!(XTaskQueueCreateComposite(work, completion, &mut b), S_OK);

        let ended = dispatcher(a);
        run_one(a, 1);
        // The other queue's end is not this one's: the wait goes on and
        // still runs what comes in.
        assert_eq!(XTaskQueueTerminate(b, false, none, None), S_OK);
        assert!(
            ended.recv_timeout(short).is_err(),
            "ended with another queue"
        );
        run_one(a, 2);
        // Its own queue's end is.
        assert_eq!(XTaskQueueTerminate(a, false, none, None), S_OK);
        assert!(
            ended.recv_timeout(long).is_ok(),
            "still waiting on a terminated queue"
        );
        // And a dispatch that starts after the end returns at once.
        assert!(!XTaskQueueDispatch(a, WORK, INFINITE));
        XTaskQueueCloseHandle(a);
        XTaskQueueCloseHandle(b);
    }

    #[test]
    fn terminate_cancels_what_is_queued() {
        static TERMINATED: AtomicBool = AtomicBool::new(false);
        extern "system" fn terminated(_: *mut c_void) {
            TERMINATED.store(true, Ordering::SeqCst);
        }
        extern "system" fn note(context: *mut c_void, canceled: bool) {
            // SAFETY: the test passes a pointer to its flag.
            unsafe { *context.cast::<bool>() = canceled };
        }
        let mut q = std::ptr::null();
        assert_eq!(XTaskQueueCreate(0, 0, &mut q), S_OK);
        let mut canceled = false;
        let flag = (&mut canceled as *mut bool).cast();
        assert_eq!(
            XTaskQueueSubmitDelayedCallback(q, WORK, 0, flag, Some(note)),
            S_OK
        );
        assert_eq!(
            XTaskQueueTerminate(q, true, std::ptr::null_mut(), Some(terminated)),
            S_OK
        );
        assert!(canceled && TERMINATED.load(Ordering::SeqCst));
        assert_eq!(
            XTaskQueueSubmitDelayedCallback(q, WORK, 0, flag, Some(note)),
            E_ABORT
        );
        XTaskQueueCloseHandle(q);
    }
}


#[cfg(test)]
mod systems_hle_critical {
    use super::*;
    use std::sync::atomic::{
        AtomicUsize, AtomicBool, Ordering
    };
    use std::time::{Duration, Instant};
    use std::thread;

    const ACQ: Ordering = Ordering::Acquire;
    const REL: Ordering = Ordering::Release;
    const SEQ: Ordering = Ordering::SeqCst;

    struct Counts {
        normal: AtomicUsize,
        canceled: AtomicUsize,
        terminated: AtomicUsize,
        canceled_at_termination: AtomicUsize,
    }

    impl Counts {
        fn new() -> &'static Self {
            Box::leak(Box::new(Self {
                normal: AtomicUsize::new(0),
                canceled: AtomicUsize::new(0),
                terminated: AtomicUsize::new(0),
                canceled_at_termination:
                    AtomicUsize::new(0),
            }))
        }

        fn ptr(&self) -> *mut c_void {
            (self as *const Self).cast_mut().cast()
        }
    }

    extern "system" fn count(
        context: *mut c_void,
        canceled: bool
    ) {
        // SAFETY: The test passes a leaked Counts allocation as the callback context.
        let c = unsafe {
            &*context.cast::<Counts>()
        };

        if canceled {
            c.canceled.fetch_add(1,SEQ);
        } else {
            c.normal.fetch_add(1,SEQ);
        }
    }

    extern "system" fn terminated(
        context: *mut c_void
    ) {
        // SAFETY: The termination context is the same leaked Counts allocation.
        let c = unsafe {
            &*context.cast::<Counts>()
        };

        c.canceled_at_termination.store(
            c.canceled.load(ACQ),
            REL
        );

        c.terminated.fetch_add(1,SEQ);
    }

    fn make_queue(
        work:u32,
        completion:u32
    ) -> *const Queue {
        let mut q = std::ptr::null();

        assert_eq!(
            XTaskQueueCreate(
                work,completion,&mut q
            ),
            S_OK
        );

        assert!(!q.is_null());

        q
    }

    #[test]
    fn registry_inventory() {
        let modules = crate::registry();

        let mut total = 0usize;

        for m in modules {
            println!(
                "HLE_MODULE name={} exports={}",
                m.name,
                m.exports.len()
            );

            assert!(
                m.exports.iter().all(
                    |e| e.address != 0
                )
            );

            total += m.exports.len();
        }

        println!(
            "HLE_REGISTRY modules={} exports={}",
            modules.len(),total
        );

        assert!(total > 0);

        println!("HLE_REGISTRY_PASS");
    }

    #[test]
    fn manual_ports_are_independent() {
        let q = make_queue(0,0);

        let work = Counts::new();
        let completion = Counts::new();

        assert_eq!(
            XTaskQueueSubmitDelayedCallback(
                q,WORK,0,
                work.ptr(),Some(count)
            ),S_OK
        );

        assert_eq!(
            XTaskQueueSubmitDelayedCallback(
                q,COMPLETION,0,
                completion.ptr(),Some(count)
            ),S_OK
        );

        assert_eq!(work.normal.load(ACQ),0);
        assert_eq!(completion.normal.load(ACQ),0);

        assert!(XTaskQueueDispatch(q,WORK,0));

        assert_eq!(work.normal.load(ACQ),1);
        assert_eq!(completion.normal.load(ACQ),0);

        assert!(!XTaskQueueDispatch(q,WORK,0));

        assert!(XTaskQueueDispatch(
            q,COMPLETION,0
        ));

        assert_eq!(completion.normal.load(ACQ),1);

        assert!(!XTaskQueueDispatch(
            q,COMPLETION,0
        ));

        XTaskQueueCloseHandle(q);

        println!("HLE_MANUAL_PORTS_PASS");
    }

    #[test]
    fn duplicated_handle_preserves_queue() {
        let q = make_queue(0,0);

        let c = Counts::new();

        let mut duplicate = std::ptr::null();

        assert_eq!(
            XTaskQueueDuplicateHandle(
                q,&mut duplicate
            ),S_OK
        );

        assert_eq!(
            XTaskQueueSubmitDelayedCallback(
                q,WORK,0,c.ptr(),Some(count)
            ),S_OK
        );

        XTaskQueueCloseHandle(q);

        assert!(
            XTaskQueueDispatch(
                duplicate,WORK,0
            )
        );

        assert_eq!(c.normal.load(ACQ),1);

        XTaskQueueCloseHandle(duplicate);

        println!("HLE_HANDLE_LIFETIME_PASS");
    }

    #[test]
    fn delayed_callbacks_must_cancel_on_terminate() {
        // Immediate ports, with delayed submissions.
        let q = make_queue(3,3);
        let c = Counts::new();

        for port in [WORK,COMPLETION] {
            assert_eq!(
                XTaskQueueSubmitDelayedCallback(
                    q,port,1000,
                    c.ptr(),Some(count)
                ),S_OK
            );
        }

        assert_eq!(c.normal.load(ACQ),0);

        assert_eq!(
            XTaskQueueTerminate(
                q,true,
                c.ptr(),Some(terminated)
            ),S_OK
        );

        let canceled_at_return =
            c.canceled.load(ACQ);

        let terminated_at_return =
            c.terminated.load(ACQ);

        // Both the callback context and its counters
        // stay alive even if the implementation
        // incorrectly dispatches after termination.
        thread::sleep(
            Duration::from_millis(1300)
        );

        let normal = c.normal.load(ACQ);
        let canceled = c.canceled.load(ACQ);

        let at_termination =
            c.canceled_at_termination.load(ACQ);

        let refused =
            XTaskQueueSubmitDelayedCallback(
                q,WORK,0,
                c.ptr(),Some(count)
            );

        XTaskQueueCloseHandle(q);

        println!(
            "HLE_DELAYED_TERMINATION normal={} \
            canceled={} canceled_at_return={} \
            terminated_at_return={} \
            canceled_at_termination={}",
            normal,canceled,
            canceled_at_return,
            terminated_at_return,
            at_termination
        );

        assert_eq!(
            refused,E_ABORT,
            "HLE_NEW_WORK_ACCEPTED_AFTER_TERMINATION"
        );

        assert_eq!(
            (
                normal,
                canceled,
                canceled_at_return,
                terminated_at_return,
                at_termination
            ),
            (0,2,2,1,2),
            "HLE_DELAYED_TERMINATION_DEFECT"
        );

        println!("HLE_DELAYED_TERMINATION_PASS");
    }

    struct Running {
        entered: AtomicBool,
        release: AtomicBool,
        finished: AtomicBool,
    }

    extern "system" fn running(
        context:*mut c_void,
        _canceled:bool
    ) {
        // SAFETY: The test supplies a leaked Running allocation to this callback.
        let state = unsafe {
            &*context.cast::<Running>()
        };

        state.entered.store(true,REL);

        let deadline = Instant::now()
            + Duration::from_secs(5);

        while !state.release.load(ACQ)
            && Instant::now() < deadline
        {
            thread::sleep(
                Duration::from_millis(1)
            );
        }

        state.finished.store(true,REL);
    }

    #[test]
    fn termination_waits_for_running_callback() {
        let q = make_queue(1,1);

        let state:&'static Running =
            Box::leak(Box::new(Running {
                entered:AtomicBool::new(false),
                release:AtomicBool::new(false),
                finished:AtomicBool::new(false),
            }));

        let ptr = (
            state as *const Running
        ).cast_mut().cast();

        assert_eq!(
            XTaskQueueSubmitDelayedCallback(
                q,WORK,0,ptr,Some(running)
            ),S_OK
        );

        let until = Instant::now()
            + Duration::from_secs(3);

        while !state.entered.load(ACQ)
            && Instant::now() < until
        {
            thread::sleep(
                Duration::from_millis(1)
            );
        }

        assert!(
            state.entered.load(ACQ),
            "HLE_WORKER_CONTROL_FAILED"
        );

        let q_address = q as usize;

        let (sent,received) =
            std::sync::mpsc::channel();

        let (enter_tx,enter_rx) =
            std::sync::mpsc::channel();

        let terminator = thread::spawn(move || {
            enter_tx.send(()).unwrap();

            let hr = XTaskQueueTerminate(
                q_address as *const Queue,
                true,
                std::ptr::null_mut(),
                None
            );

            sent.send((
                hr,state.finished.load(ACQ)
            )).unwrap();
        });

        enter_rx.recv_timeout(
            Duration::from_secs(2)
        ).unwrap();

        let early = received.recv_timeout(
            Duration::from_millis(250)
        ).ok();

        state.release.store(true,REL);

        let final_result = match early {
            Some(value) => value,
            None => received.recv_timeout(
                Duration::from_secs(5)
            ).unwrap()
        };

        terminator.join().unwrap();

        let until = Instant::now()
            + Duration::from_secs(2);

        while !state.finished.load(ACQ)
            && Instant::now() < until
        {
            thread::sleep(
                Duration::from_millis(1)
            );
        }

        assert!(state.finished.load(ACQ));

        XTaskQueueCloseHandle(q);

        println!(
            "HLE_RUNNING_TERMINATION \
            returned_early={} \
            finished_at_return={}",
            early.is_some(),
            final_result.1
        );

        assert_eq!(final_result.0,S_OK);

        assert!(
            !early.is_some()
                && final_result.1,
            "HLE_RUNNING_TERMINATION_DEFECT"
        );

        println!("HLE_RUNNING_TERMINATION_PASS");
    }

    static IDENTITY:u8 = 1;
    static WRONG_IDENTITY:u8 = 2;

    #[test]
    fn async_result_bounds_and_identity() {
        let q = make_queue(0,0);

        let identity =
            (&raw const IDENTITY).cast();

        let wrong =
            (&raw const WRONG_IDENTITY).cast();

        let mut b = AsyncBlock {
            queue:q,
            context:std::ptr::null_mut(),
            callback:None,
            internal:[0;4],
        };

        let payload=vec![
            0x12u8,0x34,0x56,0x78,
            0x9a,0xbc,0xde,0xf0
        ];

        assert_eq!(
            finished_with(
                &mut b,identity,
                Ok(payload.clone())
            ),S_OK
        );

        assert_eq!(
            XAsyncGetStatus(
                &mut b,false
            ),S_OK
        );

        let mut size=0;

        assert_eq!(
            XAsyncGetResultSize(
                &mut b,&mut size
            ),S_OK
        );

        assert_eq!(size,8);

        let mut buf=[0xCDu8;16];

        let mut used=0;

        assert_eq!(
            XAsyncGetResult(
                &mut b,wrong,
                8,buf.as_mut_ptr().cast(),
                &mut used
            ),E_INVALIDARG
        );

        assert_eq!(buf,[0xCD;16]);

        assert_eq!(
            XAsyncGetResult(
                &mut b,identity,
                7,buf.as_mut_ptr().cast(),
                &mut used
            ),E_NOT_SUFFICIENT_BUFFER
        );

        assert_eq!(buf,[0xCD;16]);

        assert_eq!(
            XAsyncGetResult(
                &mut b,identity,
                8,buf.as_mut_ptr().cast(),
                &mut used
            ),S_OK
        );

        assert_eq!(used,8);
        assert_eq!(&buf[..8],payload.as_slice());
        assert_eq!(&buf[8..],&[0xCD;8]);

        XTaskQueueCloseHandle(q);

        println!("HLE_ASYNC_RESULT_PASS");
    }

    struct AsyncCounters {
        canceled:AtomicUsize,
        cleanup:AtomicUsize,
        callbacks:AtomicUsize,
    }

    extern "system" fn provider(
        op:u32,
        data:*const ProviderData
    ) -> HRESULT {
        // SAFETY: The runtime supplies valid ProviderData for the duration of this call.
        let data=unsafe { &*data };

        // SAFETY: XAsyncBegin receives the leaked AsyncCounters allocation as context.
        let c=unsafe {
            &*data.context.cast::<AsyncCounters>()
        };

        if op==OP_CANCEL {
            c.canceled.fetch_add(1,SEQ);
        }

        if op==OP_CLEANUP {
            c.cleanup.fetch_add(1,SEQ);
        }

        S_OK
    }

    extern "system" fn completed(
        b:*mut AsyncBlock
    ) {
        // SAFETY: The test keeps its AsyncBlock alive through dispatch; its context is leaked.
        let c=unsafe {
            &*(*b).context.cast::<AsyncCounters>()
        };

        c.callbacks.fetch_add(1,SEQ);
    }

    #[test]
    fn async_cancel_completes_exactly_once() {
        let q=make_queue(0,0);

        let c=Box::leak(Box::new(
            AsyncCounters {
                canceled:AtomicUsize::new(0),
                cleanup:AtomicUsize::new(0),
                callbacks:AtomicUsize::new(0),
            }
        ));

        let ptr=(
            c as *mut AsyncCounters
        ).cast();

        let mut b=AsyncBlock {
            queue:q,
            context:ptr,
            callback:Some(completed),
            internal:[0;4]
        };

        let identity=
            (&raw const IDENTITY).cast();

        assert_eq!(
            XAsyncBegin(
                &mut b,ptr,identity,
                std::ptr::null(),
                Some(provider)
            ),S_OK
        );

        assert_eq!(
            XAsyncGetStatus(
                &mut b,false
            ),E_PENDING
        );

        XAsyncCancel(&mut b);

        // A second cancellation must not deliver
        // another completion or cleanup.
        XAsyncCancel(&mut b);

        assert_eq!(
            XAsyncGetStatus(
                &mut b,false
            ),E_ABORT
        );

        assert!(
            XTaskQueueDispatch(
                q,COMPLETION,0
            )
        );

        assert!(
            !XTaskQueueDispatch(
                q,COMPLETION,0
            )
        );

        assert_eq!(c.canceled.load(ACQ),1);
        assert_eq!(c.cleanup.load(ACQ),1);
        assert_eq!(c.callbacks.load(ACQ),1);

        XTaskQueueCloseHandle(q);

        println!("HLE_ASYNC_CANCEL_PASS");
    }
}




#[cfg(test)]
mod systems_upstream_review {
    use super::*;
    use std::sync::atomic::{
        AtomicBool, AtomicUsize, Ordering
    };
    use std::time::{Duration,Instant};
    use std::thread;

    const ACQ:Ordering = Ordering::Acquire;
    const REL:Ordering = Ordering::Release;

    struct State {
        entered:AtomicBool,
        release:AtomicBool,
        finished:AtomicBool,
        terminated:AtomicUsize,
        early_callback:AtomicBool,
    }

    impl State {
        fn new() -> &'static State {
            Box::leak(Box::new(State {
                entered:AtomicBool::new(false),
                release:AtomicBool::new(false),
                finished:AtomicBool::new(false),
                terminated:AtomicUsize::new(0),
                early_callback:AtomicBool::new(false),
            }))
        }

        fn context(&self) -> *mut c_void {
            (self as *const Self)
                .cast_mut().cast()
        }
    }

    fn until(
        limit:Duration,
        predicate:impl Fn()->bool
    ) -> bool {
        let deadline=Instant::now()+limit;

        while Instant::now()<deadline {
            if predicate() {
                return true;
            }

            thread::sleep(
                Duration::from_millis(1)
            );
        }

        predicate()
    }

    extern "system" fn work(
        context:*mut c_void,
        _canceled:bool
    ) {
        // SAFETY: The test context points to a leaked State allocation.
        let s=unsafe {
            &*context.cast::<State>()
        };

        s.entered.store(true,REL);

        until(
            Duration::from_secs(5),
            ||s.release.load(ACQ)
        );

        s.finished.store(true,REL);
    }

    extern "system" fn on_terminated(
        context:*mut c_void
    ) {
        // SAFETY: The termination callback receives the same leaked State allocation.
        let s=unsafe {
            &*context.cast::<State>()
        };

        if !s.finished.load(ACQ) {
            s.early_callback.store(true,REL);
        }

        s.terminated.fetch_add(
            1,Ordering::AcqRel
        );
    }

    fn termination_case(
        mode:u32,
        wait:bool,
        notify:bool
    ) {
        let mut q=std::ptr::null();

        assert_eq!(
            XTaskQueueCreate(
                mode,mode,&mut q
            ),S_OK
        );

        let s=State::new();

        assert_eq!(
            XTaskQueueSubmitDelayedCallback(
                q,WORK,0,
                s.context(),Some(work)
            ),S_OK
        );

        let q_addr=q as usize;

        // Manual ports require a separate dispatcher.
        let dispatcher=if mode==0 {
            Some(thread::spawn(move || {
                XTaskQueueDispatch(
                    q_addr as *const Queue,
                    WORK,1000
                )
            }))
        } else {
            None
        };

        let entered=until(
            Duration::from_secs(3),
            ||s.entered.load(ACQ)
        );

        if !entered {
            s.release.store(true,REL);

            panic!("WORKER_DID_NOT_START");
        }

        let (tx,rx)=std::sync::mpsc::channel();

        let (started_tx,started_rx)=
            std::sync::mpsc::channel();

        let terminator=thread::spawn(move || {
            started_tx.send(()).unwrap();

            let cb:Option<TerminatedCallback>=
                if notify {
                    Some(on_terminated)
                } else {
                    None
                };

            let hr=XTaskQueueTerminate(
                q_addr as *const Queue,
                wait,s.context(),cb
            );

            tx.send((
                hr,s.finished.load(ACQ)
            )).unwrap();
        });

        started_rx.recv_timeout(
            Duration::from_secs(2)
        ).unwrap();

        let early=rx.recv_timeout(
            Duration::from_millis(250)
        ).ok();

        let callbacks_before_release=
            s.terminated.load(ACQ);

        // Always unblock the worker before assertions.
        s.release.store(true,REL);

        let returned=match early {
            Some(value)=>value,
            None=>rx.recv_timeout(
                Duration::from_secs(5)
            ).unwrap()
        };

        terminator.join().unwrap();

        if let Some(worker)=dispatcher {
            worker.join().unwrap();
        }

        assert!(
            until(
                Duration::from_secs(2),
                ||s.finished.load(ACQ)
            ),
            "WORKER_DID_NOT_FINISH"
        );

        if notify {
            let _=until(
                Duration::from_secs(2),
                || {
                    if mode==0 {
                        XTaskQueueDispatch(
                            q,COMPLETION,0
                        );
                    }

                    s.terminated.load(ACQ)>0
                }
            );
        }

        let count=s.terminated.load(ACQ);

        let callback_early=
            s.early_callback.load(ACQ);

        XTaskQueueCloseHandle(q);

        println!(
            "UPSTREAM_TERMINATION mode={} wait={} \
            notify={} returned_early={} \
            finished_at_return={} \
            callback_before_release={} \
            callback_count={} callback_early={}",
            mode,wait,notify,
            early.is_some(),returned.1,
            callbacks_before_release,
            count,callback_early
        );

        assert_eq!(returned.0,S_OK);

        if wait {
            assert!(
                early.is_none() && returned.1,
                "UPSTREAM_WAIT_RETURNED_EARLY"
            );
        } else {
            assert!(
                early.is_some(),
                "UPSTREAM_ASYNC_TERMINATE_BLOCKED"
            );
        }

        if notify {
            assert_eq!(
                callbacks_before_release,0,
                "UPSTREAM_EARLY_CALLBACK"
            );

            assert!(!callback_early);

            assert_eq!(
                count,1,
                "UPSTREAM_CALLBACK_COUNT"
            );
        }

        println!("UPSTREAM_CASE_PASS");
    }

    #[test]
    fn manual_wait_without_callback() {
        termination_case(0,true,false);
    }

    #[test]
    fn pool_wait_without_callback() {
        termination_case(1,true,false);
    }

    #[test]
    fn pool_wait_with_callback() {
        termination_case(1,true,true);
    }

    #[test]
    fn pool_async_termination() {
        termination_case(1,false,true);
    }

    #[test]
    fn serialized_wait_with_callback() {
        termination_case(2,true,true);
    }

    #[test]
    fn serialized_async_termination() {
        termination_case(2,false,true);
    }

    struct Counter {
        normal:AtomicUsize,
        canceled:AtomicUsize,
    }

    impl Counter {
        fn new()->&'static Self {
            Box::leak(Box::new(Self {
                normal:AtomicUsize::new(0),
                canceled:AtomicUsize::new(0),
            }))
        }

        fn ptr(&self)->*mut c_void {
            (self as *const Self)
                .cast_mut().cast()
        }
    }

    extern "system" fn count(
        context:*mut c_void,
        canceled:bool
    ) {
        // SAFETY: Counter::new leaks the allocation used as this callback context.
        let c=unsafe {
            &*context.cast::<Counter>()
        };

        if canceled {
            c.canceled.fetch_add(
                1,Ordering::AcqRel
            );
        } else {
            c.normal.fetch_add(
                1,Ordering::AcqRel
            );
        }
    }

    #[test]
    fn composite_queue_isolation() {
        let mut a=std::ptr::null();

        assert_eq!(
            XTaskQueueCreate(0,0,&mut a),
            S_OK
        );

        let mut work_port=std::ptr::null();
        let mut completion_port=std::ptr::null();

        assert_eq!(
            XTaskQueueGetPort(
                a,WORK,&mut work_port
            ),S_OK
        );

        assert_eq!(
            XTaskQueueGetPort(
                a,COMPLETION,
                &mut completion_port
            ),S_OK
        );

        let mut b=std::ptr::null();

        assert_eq!(
            XTaskQueueCreateComposite(
                work_port,
                completion_port,
                &mut b
            ),S_OK
        );

        let ca=Counter::new();
        let cb=Counter::new();

        assert_eq!(
            XTaskQueueSubmitDelayedCallback(
                a,WORK,0,
                ca.ptr(),Some(count)
            ),S_OK
        );

        assert_eq!(
            XTaskQueueSubmitDelayedCallback(
                b,WORK,0,
                cb.ptr(),Some(count)
            ),S_OK
        );

        assert_eq!(
            XTaskQueueTerminate(
                a,false,
                std::ptr::null_mut(),None
            ),S_OK
        );

        let _=until(
            Duration::from_secs(2),
            || {
                XTaskQueueDispatch(
                    b,WORK,0
                );

                XTaskQueueDispatch(
                    b,COMPLETION,0
                );

                ca.canceled.load(ACQ)>0
                && (
                    cb.normal.load(ACQ)>0 ||
                    cb.canceled.load(ACQ)>0
                )
            }
        );

        let observed=(
            ca.normal.load(ACQ),
            ca.canceled.load(ACQ),
            cb.normal.load(ACQ),
            cb.canceled.load(ACQ)
        );

        XTaskQueueCloseHandle(a);
        XTaskQueueCloseHandle(b);

        println!(
            "UPSTREAM_SHARED_PORTS={observed:?}"
        );

        assert_eq!(
            observed,(0,1,1,0),
            "UPSTREAM_SHARED_PORT_ISOLATION"
        );

        println!("UPSTREAM_CASE_PASS");
    }
}




#[cfg(test)]
mod systems_unified_stress {
    use super::*;
    use std::sync::atomic::{
        AtomicBool, AtomicUsize, Ordering
    };
    use std::sync::Barrier;
    use std::thread;

    struct Item {
        hits: AtomicUsize,
        accepted: AtomicBool,
    }

    impl Item {
        fn new() -> &'static Item {
            Box::leak(Box::new(Item {
                hits: AtomicUsize::new(0),
                accepted: AtomicBool::new(false),
            }))
        }

        fn ptr(&self) -> *mut c_void {
            (self as *const Self)
                .cast_mut().cast()
        }
    }

    extern "system" fn count(
        context:*mut c_void,
        _canceled:bool
    ) {
        // SAFETY: Item::new leaks each test item passed to this callback.
        let item=unsafe {
            &*context.cast::<Item>()
        };

        item.hits.fetch_add(
            1,Ordering::SeqCst
        );
    }

    fn submit(
        q:*const Queue,
        item:&Item,
        port:u32,
        delay:u32
    ) {
        let hr=XTaskQueueSubmitDelayedCallback(
            q,port,delay,
            item.ptr(),Some(count)
        );

        assert!(
            hr==S_OK || hr==E_ABORT
        );

        item.accepted.store(
            hr==S_OK,Ordering::SeqCst
        );
    }

    fn race(mode:u32) {
        let mut q=std::ptr::null();

        assert_eq!(
            XTaskQueueCreate(mode,mode,&mut q),
            S_OK
        );

        let q_addr=q as usize;

        let items:Vec<&'static Item>=
            (0..136).map(|_|Item::new()).collect();

        // Ensure a nonzero number of accepted tasks.
        for (i, &item) in items[..8].iter().enumerate() {
            submit(
                q,item,
                (i%2) as u32,
                100
            );
        }

        let barrier=Arc::new(Barrier::new(5));

        let mut workers=Vec::new();

        for worker in 0..4 {
            let start=8+worker*32;

            let group=items[start..start+32].to_vec();

            let barrier=barrier.clone();

            workers.push(thread::spawn(move || {
                barrier.wait();

                for (i,item) in group.iter().enumerate() {
                    let delay=if i%4==0 { 20 } else { 0 };

                    submit(
                        q_addr as *const Queue,
                        item,
                        (i%2) as u32,
                        delay
                    );
                }
            }));
        }

        barrier.wait();

        let dispatchers:Vec<_>=if mode==0 {
            [WORK,COMPLETION].into_iter()
                .map(|port| {
                    thread::spawn(move || {
                        while XTaskQueueDispatch(
                            q_addr as *const Queue,
                            port,INFINITE
                        ) {}
                    })
                }).collect()
        } else {
            Vec::new()
        };

        thread::yield_now();

        assert_eq!(
            XTaskQueueTerminate(
                q,true,std::ptr::null_mut(),None
            ),
            S_OK
        );

        for worker in workers {
            worker.join().unwrap();
        }

        for dispatcher in dispatchers {
            dispatcher.join().unwrap();
        }

        let mut accepted=0;
        let mut delivered=0;

        for item in &items {
            let wanted=usize::from(
                item.accepted.load(Ordering::SeqCst)
            );

            let hits=item.hits.load(Ordering::SeqCst);

            assert_eq!(
                hits,wanted,
                "UNIFIED_EXACTLY_ONCE_DEFECT"
            );

            accepted+=wanted;
            delivered+=hits;
        }

        assert!(accepted>=8);

        // No pending callbacks or active delivery.
        let owner=queue(q).unwrap();

        {
            let state=lock(&owner.state);

            assert!(state.pending.is_empty());
            assert_eq!(state.active,[0,0]);
            assert!(state.termination_complete);
        }

        // Delayed callbacks must not run a second time.
        thread::sleep(Duration::from_millis(150));

        for item in &items {
            assert_eq!(
                item.hits.load(Ordering::SeqCst),
                usize::from(
                    item.accepted.load(Ordering::SeqCst)
                )
            );
        }

        XTaskQueueCloseHandle(q);

        println!(
            "UNIFIED_RACE mode={mode} \
            accepted={accepted} delivered={delivered}"
        );
    }

    #[test]
    fn manual_race() {
        race(0);
    }

    #[test]
    fn pool_race() {
        race(1);
    }

    #[test]
    fn serialized_race() {
        race(2);
    }

    #[test]
    fn immediate_race() {
        race(3);
    }

    #[test]
    fn closing_original_handle_keeps_pending_work_alive() {
        let mut q=std::ptr::null();

        assert_eq!(
            XTaskQueueCreate(0,0,&mut q),
            S_OK
        );

        let item=Item::new();

        submit(q,item,WORK,0);

        let owner=queue(q).unwrap();

        let weak=Arc::downgrade(&owner);

        drop(owner);

        // The accepted dispatch token owns a reference.
        XTaskQueueCloseHandle(q);

        let owner=weak.upgrade()
            .expect("pending callback lost its queue");

        assert!(
            XTaskQueueDispatch(
                Arc::as_ptr(&owner),
                WORK,0
            )
        );

        assert_eq!(
            item.hits.load(Ordering::SeqCst),1
        );

        drop(owner);

        assert!(weak.upgrade().is_none());

        println!("UNIFIED_PENDING_LIFETIME_PASS");
    }
}




#[cfg(test)]
mod systems_affinity {
    use super::*;
    use std::sync::atomic::{
        AtomicBool, AtomicUsize, AtomicI32, Ordering
    };
    use std::thread;
    use std::time::{Duration, Instant};

    const SEQ: Ordering = Ordering::SeqCst;

    fn make_queue(
        work:u32,
        completion:u32
    ) -> *const Queue {
        let mut q=std::ptr::null();

        assert_eq!(
            XTaskQueueCreate(
                work,completion,&mut q
            ),
            S_OK
        );

        q
    }

    fn until(
        seconds:u64,
        mut predicate:impl FnMut()->bool
    ) -> bool {
        let deadline=Instant::now()
            +Duration::from_secs(seconds);

        while Instant::now()<deadline {
            if predicate() {
                return true;
            }

            thread::sleep(
                Duration::from_millis(1)
            );
        }

        predicate()
    }

    struct Manual {
        dispatcher:thread::ThreadId,
        work:AtomicUsize,
        completion:AtomicUsize,
        notified:AtomicUsize,
        violations:AtomicUsize,
    }

    impl Manual {
        fn new()->&'static Self {
            Box::leak(Box::new(Self {
                dispatcher:thread::current().id(),
                work:AtomicUsize::new(0),
                completion:AtomicUsize::new(0),
                notified:AtomicUsize::new(0),
                violations:AtomicUsize::new(0),
            }))
        }

        fn ptr(&self)->*mut c_void {
            (self as *const Self)
                .cast_mut().cast()
        }

        fn check(&self,condition:bool) {
            if !condition {
                self.violations.fetch_add(1,SEQ);
            }
        }
    }

    extern "system" fn cancel_work(
        context:*mut c_void,
        canceled:bool
    ) {
        // SAFETY: The work callback receives the leaked Manual test state.
        let s=unsafe {
            &*context.cast::<Manual>()
        };

        s.check(canceled);
        s.check(
            thread::current().id()==s.dispatcher
        );

        s.work.fetch_add(1,SEQ);
    }

    extern "system" fn cancel_completion(
        context:*mut c_void,
        canceled:bool
    ) {
        // SAFETY: The completion callback receives the leaked Manual test state.
        let s=unsafe {
            &*context.cast::<Manual>()
        };

        s.check(canceled);

        s.check(
            thread::current().id()==s.dispatcher
        );

        s.check(s.work.load(SEQ)==2);

        s.completion.fetch_add(1,SEQ);
    }

    extern "system" fn notify(
        context:*mut c_void
    ) {
        // SAFETY: The notification context is the leaked Manual test state.
        let s=unsafe {
            &*context.cast::<Manual>()
        };

        s.check(
            thread::current().id()==s.dispatcher
        );

        s.check(s.work.load(SEQ)==2);
        s.check(s.completion.load(SEQ)==2);

        s.notified.fetch_add(1,SEQ);
    }

    fn manual_case(with_notification:bool) {
        let q=make_queue(0,0);
        let state=Manual::new();

        // Two accepted callbacks per port:
        // one immediate submission and one delayed.
        for delay in [0,60000] {
            assert_eq!(
                XTaskQueueSubmitDelayedCallback(
                    q,WORK,delay,
                    state.ptr(),
                    Some(cancel_work)
                ),
                S_OK
            );

            assert_eq!(
                XTaskQueueSubmitDelayedCallback(
                    q,COMPLETION,delay,
                    state.ptr(),
                    Some(cancel_completion)
                ),
                S_OK
            );
        }

        let notification:Option<TerminatedCallback>=
            if with_notification {
                Some(notify)
            } else {
                None
            };

        assert_eq!(
            XTaskQueueTerminate(
                q,false,
                state.ptr(),notification
            ),
            S_OK
        );

        // The test's thread owns both manual ports.
        let completed=until(5,|| {
            XTaskQueueDispatch(q,WORK,0);

            XTaskQueueDispatch(
                q,COMPLETION,0
            );

            state.work.load(SEQ)==2
            && state.completion.load(SEQ)==2
            && (
                !with_notification
                || state.notified.load(SEQ)==1
            )
        });

        let observed=(
            state.work.load(SEQ),
            state.completion.load(SEQ),
            state.notified.load(SEQ),
            state.violations.load(SEQ)
        );

        println!(
            "AFFINITY_MANUAL notify={} \
            completed={} observed={observed:?}",
            with_notification,completed
        );

        XTaskQueueCloseHandle(q);

        assert!(completed);

        assert_eq!(
            observed,
            (
                2,2,
                usize::from(with_notification),
                0
            ),
            "AFFINITY_MANUAL_DEFECT"
        );
    }

    #[test]
    fn manual_completion_affinity() {
        manual_case(true);
    }

    #[test]
    fn manual_cancellation_without_notification() {
        manual_case(false);
    }

    struct Serial {
        entered:AtomicBool,
        release:AtomicBool,
        active:AtomicBool,
        canceled:AtomicUsize,
        overlap:AtomicBool,
        notified:AtomicBool,
    }

    extern "system" fn blocking(
        context:*mut c_void,
        canceled:bool
    ) {
        // SAFETY: The test supplies a leaked Serial allocation as callback context.
        let s=unsafe {
            &*context.cast::<Serial>()
        };

        if canceled {
            s.overlap.store(true,SEQ);
        }

        s.active.store(true,SEQ);
        s.entered.store(true,SEQ);

        until(
            5,||s.release.load(SEQ)
        );

        s.active.store(false,SEQ);
    }

    extern "system" fn serialized_cancel(
        context:*mut c_void,
        canceled:bool
    ) {
        // SAFETY: The cancellation callback uses the same leaked Serial allocation.
        let s=unsafe {
            &*context.cast::<Serial>()
        };

        if !canceled || s.active.load(SEQ) {
            s.overlap.store(true,SEQ);
        }

        s.canceled.fetch_add(1,SEQ);
    }

    extern "system" fn serialized_notify(
        context:*mut c_void
    ) {
        // SAFETY: The notification callback uses the leaked Serial allocation.
        let s=unsafe {
            &*context.cast::<Serial>()
        };

        if s.active.load(SEQ)
            || s.canceled.load(SEQ)!=1
        {
            s.overlap.store(true,SEQ);
        }

        s.notified.store(true,SEQ);
    }

    #[test]
    fn serialized_cancel_does_not_overlap_active_work() {
        let q=make_queue(2,2);

        let s:&'static Serial=
            Box::leak(Box::new(Serial {
                entered:AtomicBool::new(false),
                release:AtomicBool::new(false),
                active:AtomicBool::new(false),
                canceled:AtomicUsize::new(0),
                overlap:AtomicBool::new(false),
                notified:AtomicBool::new(false),
            }));

        let context=(
            s as *const Serial
        ).cast_mut().cast();

        assert_eq!(
            XTaskQueueSubmitDelayedCallback(
                q,WORK,0,
                context,Some(blocking)
            ),
            S_OK
        );

        assert!(until(
            3,||s.entered.load(SEQ)
        ));

        assert_eq!(
            XTaskQueueSubmitDelayedCallback(
                q,WORK,60000,
                context,Some(serialized_cancel)
            ),
            S_OK
        );

        assert_eq!(
            XTaskQueueTerminate(
                q,false,context,
                Some(serialized_notify)
            ),
            S_OK
        );

        thread::sleep(
            Duration::from_millis(200)
        );

        let premature=s.canceled.load(SEQ);

        s.release.store(true,SEQ);

        assert!(until(
            5,||s.notified.load(SEQ)
        ));

        let overlap=s.overlap.load(SEQ);
        let canceled=s.canceled.load(SEQ);

        XTaskQueueCloseHandle(q);

        println!(
            "AFFINITY_SERIAL premature={premature} \
            overlap={overlap} canceled={canceled}"
        );

        assert_eq!(
            premature,0,
            "AFFINITY_SERIAL_PREMATURE"
        );

        assert!(!overlap);
        assert_eq!(canceled,1);
    }

    struct Reentrant {
        queue:usize,
        dispatcher:thread::ThreadId,
        result:AtomicI32,
        returned:AtomicBool,
        notified:AtomicBool,
        violation:AtomicBool,
    }

    extern "system" fn reentrant_notify(
        context:*mut c_void
    ) {
        // SAFETY: The reentrant notification uses a leaked Reentrant allocation.
        let s=unsafe {
            &*context.cast::<Reentrant>()
        };

        if !s.returned.load(SEQ)
            || thread::current().id()!=s.dispatcher
        {
            s.violation.store(true,SEQ);
        }

        s.notified.store(true,SEQ);
    }

    extern "system" fn reentrant_work(
        context:*mut c_void,
        canceled:bool
    ) {
        // SAFETY: The reentrant work callback uses a leaked Reentrant allocation.
        let s=unsafe {
            &*context.cast::<Reentrant>()
        };

        if canceled {
            s.violation.store(true,SEQ);
        }

        let hr=XTaskQueueTerminate(
            s.queue as *const Queue,
            false,
            context,
            Some(reentrant_notify)
        );

        s.result.store(hr,SEQ);
        s.returned.store(true,SEQ);
    }

    #[test]
    fn callback_can_request_async_termination() {
        let q=make_queue(0,0);

        let s:&'static Reentrant=
            Box::leak(Box::new(Reentrant {
                queue:q as usize,
                dispatcher:thread::current().id(),
                result:AtomicI32::new(E_PENDING),
                returned:AtomicBool::new(false),
                notified:AtomicBool::new(false),
                violation:AtomicBool::new(false),
            }));

        let context=(
            s as *const Reentrant
        ).cast_mut().cast();

        assert_eq!(
            XTaskQueueSubmitDelayedCallback(
                q,WORK,0,
                context,Some(reentrant_work)
            ),
            S_OK
        );

        assert!(
            XTaskQueueDispatch(q,WORK,0)
        );

        assert!(s.returned.load(SEQ));
        assert_eq!(s.result.load(SEQ),S_OK);

        let notified=until(5,||{
            XTaskQueueDispatch(
                q,COMPLETION,0
            );

            s.notified.load(SEQ)
        });

        let violation=s.violation.load(SEQ);

        XTaskQueueCloseHandle(q);

        println!(
            "AFFINITY_REENTRANT \
            notified={notified} violation={violation}"
        );

        assert!(notified);

        assert!(
            !violation,
            "AFFINITY_REENTRANT_DEFECT"
        );
    }
}



#[cfg(test)]
mod systems_shared_timer {
    use super::*;
    use std::sync::atomic::{
        AtomicUsize, Ordering
    };
    use std::time::{Duration, Instant};
    use std::thread;

    const SEQ:Ordering=Ordering::SeqCst;

    #[repr(C)]
    struct ThreadEntry {
        size:u32,
        usage:u32,
        thread_id:u32,
        owner_pid:u32,
        base_priority:i32,
        delta_priority:i32,
        flags:u32,
    }

    #[link(name="kernel32")]
    unsafe extern "system" {
        fn CreateToolhelp32Snapshot(
            flags:u32,pid:u32
        )->*mut c_void;

        fn Thread32First(
            snapshot:*mut c_void,
            entry:*mut ThreadEntry
        )->i32;

        fn Thread32Next(
            snapshot:*mut c_void,
            entry:*mut ThreadEntry
        )->i32;

        fn CloseHandle(
            handle:*mut c_void
        )->i32;

        fn GetCurrentProcessId()->u32;
    }

    fn thread_count()->usize {
        // SAFETY: The snapshot handle is checked; ThreadEntry has C layout and its size is initialized. The handle is closed after enumeration.
        unsafe {
            let snapshot=
                CreateToolhelp32Snapshot(4,0);

            assert_ne!(
                snapshot as usize,
                usize::MAX
            );

            let pid=GetCurrentProcessId();

            let mut entry=ThreadEntry {
                size:std::mem::size_of::<
                    ThreadEntry
                >() as u32,
                usage:0,
                thread_id:0,
                owner_pid:0,
                base_priority:0,
                delta_priority:0,
                flags:0,
            };

            let mut count=0;

            let mut more=Thread32First(
                snapshot,&mut entry
            );

            while more!=0 {
                if entry.owner_pid==pid {
                    count+=1;
                }

                entry.size=std::mem::size_of::<
                    ThreadEntry
                >() as u32;

                more=Thread32Next(
                    snapshot,&mut entry
                );
            }

            CloseHandle(snapshot);

            assert!(count>0);

            count
        }
    }

    struct Hits {
        normal:AtomicUsize,
        canceled:AtomicUsize,
    }

    impl Hits {
        fn new()->&'static Self {
            Box::leak(Box::new(Self {
                normal:AtomicUsize::new(0),
                canceled:AtomicUsize::new(0),
            }))
        }

        fn ptr(&self)->*mut c_void {
            (self as *const Self)
                .cast_mut().cast()
        }
    }

    extern "system" fn hit(
        context:*mut c_void,
        canceled:bool
    ) {
        // SAFETY: Hits::new leaks the allocation used as the callback context.
        let s=unsafe {
            &*context.cast::<Hits>()
        };

        if canceled {
            s.canceled.fetch_add(1,SEQ);
        } else {
            s.normal.fetch_add(1,SEQ);
        }
    }

    fn make_queue(mode:u32)->*const Queue {
        let mut q=std::ptr::null();

        assert_eq!(
            XTaskQueueCreate(mode,mode,&mut q),
            S_OK
        );

        q
    }

    fn wait_until(
        mut check:impl FnMut()->bool
    )->bool {
        let deadline=Instant::now()
            +Duration::from_secs(8);

        while Instant::now()<deadline {
            if check() {
                return true;
            }

            thread::sleep(
                Duration::from_millis(2)
            );
        }

        check()
    }

    #[test]
    fn delayed_thread_budget() {
        let q=make_queue(0);
        let s=Hits::new();

        let before=thread_count();

        for _ in 0..128 {
            assert_eq!(
                XTaskQueueSubmitDelayedCallback(
                    q,WORK,60000,
                    s.ptr(),Some(hit)
                ),
                S_OK
            );
        }

        thread::sleep(
            Duration::from_millis(150)
        );

        let pending=thread_count();

        assert_eq!(
            XTaskQueueTerminate(
                q,false,std::ptr::null_mut(),None
            ),
            S_OK
        );

        let completed=wait_until(||{
            XTaskQueueDispatch(q,WORK,0);

            XTaskQueueDispatch(
                q,COMPLETION,0
            );

            queue(q).unwrap()
                .finished.load(SEQ)
        });

        let normal=s.normal.load(SEQ);
        let canceled=s.canceled.load(SEQ);

        XTaskQueueCloseHandle(q);

        let extra=pending.saturating_sub(before);

        println!(
            "TIMER_BUDGET before={before} \
            pending={pending} extra={extra} \
            normal={normal} canceled={canceled}"
        );

        assert!(completed);
        assert_eq!((normal,canceled),(0,128));

        assert!(
            extra<=16,
            "TIMER_THREAD_BUDGET_EXCEEDED"
        );
    }

    #[test]
    fn earlier_deadline_wakes_scheduler() {
        let q=make_queue(0);

        let long=Hits::new();
        let short=Hits::new();

        assert_eq!(
            XTaskQueueSubmitDelayedCallback(
                q,WORK,60000,
                long.ptr(),Some(hit)
            ),
            S_OK
        );

        thread::sleep(
            Duration::from_millis(30)
        );

        assert_eq!(
            XTaskQueueSubmitDelayedCallback(
                q,WORK,20,
                short.ptr(),Some(hit)
            ),
            S_OK
        );

        let fired=wait_until(||{
            XTaskQueueDispatch(q,WORK,0);
            short.normal.load(SEQ)==1
        });

        assert_eq!(
            XTaskQueueTerminate(
                q,false,std::ptr::null_mut(),None
            ),
            S_OK
        );

        let completed=wait_until(||{
            XTaskQueueDispatch(q,WORK,0);

            XTaskQueueDispatch(
                q,COMPLETION,0
            );

            queue(q).unwrap()
                .finished.load(SEQ)
        });

        XTaskQueueCloseHandle(q);

        assert!(fired);
        assert!(completed);

        assert_eq!(
            short.normal.load(SEQ),1
        );

        assert_eq!(
            short.canceled.load(SEQ),0
        );

        assert_eq!(
            long.normal.load(SEQ),0
        );

        assert_eq!(
            long.canceled.load(SEQ),1
        );

        println!("TIMER_EARLIER_DEADLINE_PASS");
    }

    #[test]
    fn cancellation_releases_timer_references() {
        let q=make_queue(3);
        let s=Hits::new();

        let weak=Arc::downgrade(
            &queue(q).unwrap()
        );

        for _ in 0..64 {
            assert_eq!(
                XTaskQueueSubmitDelayedCallback(
                    q,WORK,60000,
                    s.ptr(),Some(hit)
                ),
                S_OK
            );
        }

        assert_eq!(
            XTaskQueueTerminate(
                q,true,std::ptr::null_mut(),None
            ),
            S_OK
        );

        XTaskQueueCloseHandle(q);

        let released=wait_until(||{
            weak.upgrade().is_none()
        });

        assert!(released);

        assert_eq!(
            s.normal.load(SEQ),0
        );

        assert_eq!(
            s.canceled.load(SEQ),64
        );

        println!("TIMER_REFERENCES_RELEASED");
    }


    #[test]
    fn expired_burst_delivered_and_reaped() {
        let q=make_queue(0);
        let hits=Hits::new();

        let before=thread_count();

        for i in 0..128 {
            assert_eq!(
                XTaskQueueSubmitDelayedCallback(
                    q,WORK,20+(i%8)*5,
                    hits.ptr(),Some(hit)
                ),S_OK
            );
        }

        let mut peak=before;

        let delivered=wait_until(||{
            peak=peak.max(thread_count());

            XTaskQueueDispatch(q,WORK,0);

            hits.normal.load(SEQ)==128
        });

        assert_eq!(
            XTaskQueueTerminate(
                q,false,std::ptr::null_mut(),None
            ),S_OK
        );

        let finished=wait_until(||{
            XTaskQueueDispatch(q,WORK,0);
            XTaskQueueDispatch(q,COMPLETION,0);
            queue(q).unwrap().finished.load(SEQ)
        });

        XTaskQueueCloseHandle(q);

        let reaped=wait_until(||{
            let s=lock(&TIMER_DELIVERY.state);
            s.workers==0 && s.jobs.is_empty()
        });

        println!(
            "FINAL_TIMER_BURST delivered={} peak={} reaped={}",
            hits.normal.load(SEQ),peak,reaped
        );

        assert!(delivered && finished && reaped);
        assert_eq!(hits.normal.load(SEQ),128);
        assert_eq!(hits.canceled.load(SEQ),0);
    }

    #[test]
    fn delivery_does_not_block_other_timer() {
        struct Gate {
            entered:AtomicBool,
            release:AtomicBool,
        }

        extern "system" fn blocked(
            ctx:*mut c_void,_:bool
        ) {
            // SAFETY: The test leaks the Gate allocation before registering this callback.
            let gate=unsafe{&*ctx.cast::<Gate>()};
            gate.entered.store(true,SEQ);

            while !gate.release.load(SEQ) {
                thread::sleep(Duration::from_millis(1));
            }
        }

        let gate=Box::leak(Box::new(Gate{
            entered:AtomicBool::new(false),
            release:AtomicBool::new(false),
        }));

        let hits=Hits::new();

        let a=make_queue(3);
        let b=make_queue(3);

        assert_eq!(
            XTaskQueueSubmitDelayedCallback(
                a,WORK,20,
                (gate as *mut Gate).cast(),
                Some(blocked)
            ),S_OK
        );

        let entered=wait_until(||gate.entered.load(SEQ));

        assert_eq!(
            XTaskQueueSubmitDelayedCallback(
                b,WORK,20,hits.ptr(),Some(hit)
            ),S_OK
        );

        let independent=wait_until(||hits.normal.load(SEQ)==1);

        gate.release.store(true,SEQ);

        for q in [a,b] {
            assert_eq!(
                XTaskQueueTerminate(
                    q,true,std::ptr::null_mut(),None
                ),S_OK
            );
            XTaskQueueCloseHandle(q);
        }

        assert!(entered && independent);

        println!("FINAL_TIMER_INDEPENDENCE_PASS");
    }



    #[test]
    fn original_pool_reaps_idle_workers() {
        let q=make_queue(1);
        let hits=Hits::new();

        for _ in 0..64 {
            assert_eq!(
                XTaskQueueSubmitDelayedCallback(
                    q,WORK,0,
                    hits.ptr(),Some(hit)
                ),
                S_OK
            );
        }

        assert!(wait_until(||{
            hits.normal.load(SEQ)==64
        }));

        assert_eq!(
            XTaskQueueTerminate(
                q,true,std::ptr::null_mut(),None
            ),
            S_OK
        );

        XTaskQueueCloseHandle(q);

        let observed_idle=wait_until(||{
            lock(&POOL.state).1>0
        });

        let reaped=wait_until(||{
            let state=lock(&POOL.state);
            state.0.is_empty() && state.1==0
        });

        assert!(observed_idle && reaped);
        assert_eq!(hits.normal.load(SEQ),64);
        assert_eq!(hits.canceled.load(SEQ),0);

        println!("FINAL_ORIGINAL_POOL_REAPED");
    }

}
