//! A console without a network cable: the title reaches loopback only.
//!
//! Every way out of Winsock and WinHTTP a title can import or look up by
//! name is vetted here; the rest of both DLLs is the host's own.
//!
//! * Connects (also by name, by list, `ConnectEx` from `WSAIoctl`) and
//!   datagram sends to anything but 127.0.0.0/8 and ::1 fail with
//!   `WSAENETUNREACH`.
//! * Forward lookups other than `localhost` and IP literals fail with
//!   `WSAHOST_NOT_FOUND`; reverse lookups of other addresses come back
//!   numeric, or fail if a name is required, as with no network.
//! * `WinHttpConnect` to a host that is not loopback fails with
//!   `ERROR_WINHTTP_CANNOT_CONNECT`, so no request (or WebSocket) handle can
//!   exist for one. Proxies are off and proxy auto-detection (WPAD, PAC
//!   downloads) fails with `ERROR_WINHTTP_AUTODETECTION_FAILED`.
//! * A bind to all interfaces is narrowed to loopback, so nothing on the
//!   network can connect in.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::too_many_arguments)]

use std::ffi::{CStr, c_char, c_void};
use std::net::{IpAddr, SocketAddr};
use std::sync::Mutex;

use windows_sys::Win32::Foundation::SetLastError;

use crate::modules::files::host;
use crate::{HleExport, HleModule, hle_export};

const SOCKET_ERROR: i32 = -1;
const INVALID_SOCKET: Socket = !0;
const WSAEINVAL: i32 = 10022;
const WSAEADDRNOTAVAIL: i32 = 10049;
const WSAENETUNREACH: i32 = 10051;
const WSASERVICE_NOT_FOUND: i32 = 10108;
const WSAHOST_NOT_FOUND: i32 = 11001;
const ERROR_WINHTTP_CANNOT_CONNECT: u32 = 12029;
const ERROR_WINHTTP_AUTODETECTION_FAILED: u32 = 12180;
const AF_INET: u16 = 2;
const AF_INET6: u16 = 23;
const NI_NUMERICHOST: i32 = 0x02;
const NI_NAMEREQD: i32 = 0x04;
const SIO_GET_EXTENSION_FUNCTION_POINTER: u32 = 0xC800_0006;
const SIO_GET_MULTIPLE_EXTENSION_FUNCTION_POINTER: u32 = 0xC800_0024;
const WINHTTP_ACCESS_TYPE_NO_PROXY: u32 = 1;
const WINHTTP_OPTION_PROXY: u32 = 38;

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
struct Guid(u32, u16, u16, [u8; 8]);

const WSAID_CONNECTEX: Guid = Guid(
    0x25a2_07b9,
    0xddf3,
    0x4660,
    [0x8e, 0xe9, 0x76, 0xe5, 0x8c, 0x74, 0x06, 0x3e],
);
const WSAID_WSASENDMSG: Guid = Guid(
    0xa441_e712,
    0x754f,
    0x43ca,
    [0x84, 0xa7, 0x0d, 0xee, 0x44, 0xcf, 0x60, 0x6d],
);

type Socket = usize;
type Handle = *mut c_void;

/// Logs `what` for `destination` the first time (no payload, ever).
fn once(what: &'static str, destination: String) {
    static SEEN: Mutex<Vec<(&str, String)>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.iter().any(|(w, d)| *w == what && *d == destination) {
        tracing::info!(%destination, "{what}");
        seen.push((what, destination));
    }
}

fn wsa_error(code: i32) {
    host!("ws2_32.dll", "WSASetLastError": extern "system" fn(i32))(code);
}

fn narrow(text: *const c_char) -> Option<String> {
    (!text.is_null()).then(|| {
        // SAFETY: the title passes a NUL-terminated string.
        unsafe { CStr::from_ptr(text) }
            .to_string_lossy()
            .into_owned()
    })
}

fn wide(text: *const u16) -> Option<String> {
    // SAFETY: the title passes a NUL-terminated string or null.
    (!text.is_null()).then(|| unsafe {
        let len = (0..).take_while(|&i| *text.add(i) != 0).count();
        String::from_utf16_lossy(std::slice::from_raw_parts(text, len))
    })
}

/// An IPv4 or IPv6 `sockaddr` of `len` bytes.
fn socket_addr(addr: *const u8, len: i32) -> Option<SocketAddr> {
    let len = usize::try_from(len).ok()?;
    if addr.is_null() || len < 4 {
        return None;
    }
    // SAFETY: the title passes `len` readable bytes.
    let b = unsafe { std::slice::from_raw_parts(addr, len) };
    let port = u16::from_be_bytes([b[2], b[3]]);
    match u16::from_le_bytes([b[0], b[1]]) {
        AF_INET if len >= 16 => Some((<[u8; 4]>::try_from(&b[4..8]).ok()?, port).into()),
        AF_INET6 if len >= 28 => Some((<[u8; 16]>::try_from(&b[8..24]).ok()?, port).into()),
        _ => None,
    }
}

/// A raw address of family `kind` (`gethostbyaddr` style).
fn raw_ip(addr: *const c_char, len: i32, kind: i32) -> Option<IpAddr> {
    let len = usize::try_from(len).ok()?;
    if addr.is_null() {
        return None;
    }
    // SAFETY: the title passes `len` readable bytes.
    let b = unsafe { std::slice::from_raw_parts(addr.cast::<u8>(), len) };
    match (u16::try_from(kind).ok()?, len) {
        (AF_INET, 4..) => Some(<[u8; 4]>::try_from(&b[..4]).ok()?.into()),
        (AF_INET6, 16..) => Some(<[u8; 16]>::try_from(&b[..16]).ok()?.into()),
        _ => None,
    }
}

fn loopback(ip: IpAddr) -> bool {
    ip.to_canonical().is_loopback()
}

/// `localhost` or a loopback literal (bracketed IPv6 too).
fn loopback_name(name: &str) -> bool {
    let bare = name.trim_start_matches('[').trim_end_matches(']');
    bare.eq_ignore_ascii_case("localhost") || bare.parse().is_ok_and(loopback)
}

fn shown(addr: Option<impl ToString>) -> String {
    addr.map_or_else(|| "not an IP address".into(), |a| a.to_string())
}

/// True for a loopback `sockaddr`; otherwise logs it and sets
/// `WSAENETUNREACH`. Other address families count as off the machine.
fn reachable(what: &'static str, addr: *const u8, len: i32) -> bool {
    match socket_addr(addr, len) {
        Some(a) if loopback(a.ip()) => true,
        a => {
            once(what, shown(a));
            wsa_error(WSAENETUNREACH);
            false
        }
    }
}

/// Connecting by name: only to loopback. A name would need a lookup
/// (`WSAHOST_NOT_FOUND`), an address a route (`WSAENETUNREACH`).
fn reachable_name(name: Option<String>) -> bool {
    match name {
        Some(name) if !loopback_name(&name) => {
            let literal = name.parse::<IpAddr>().is_ok();
            once("network connect blocked", name);
            wsa_error(if literal {
                WSAENETUNREACH
            } else {
                WSAHOST_NOT_FOUND
            });
            false
        }
        // No name: the host rejects the call itself.
        _ => true,
    }
}

/// Names answered without asking the network: none, `localhost`, IP
/// literals (connecting to one is vetted separately).
fn resolvable(name: Option<String>) -> bool {
    match name {
        Some(name)
            if !(name.is_empty()
                || name.eq_ignore_ascii_case("localhost")
                || name.parse::<IpAddr>().is_ok()) =>
        {
            once("network name lookup blocked", name);
            wsa_error(WSAHOST_NOT_FOUND);
            false
        }
        _ => true,
    }
}

/// Reverse lookups ask DNS unless the address is loopback.
fn reverse_resolvable(ip: Option<IpAddr>) -> bool {
    match ip {
        Some(ip) if loopback(ip) => true,
        ip => {
            once("network reverse lookup blocked", shown(ip));
            false
        }
    }
}

/// `getnameinfo` flags that keep a host name lookup on the machine: the
/// numeric form, as without a network, or `None` if a name is required.
fn name_info_flags(sa: *const u8, len: i32, host_wanted: bool, flags: i32) -> Option<i32> {
    if !host_wanted
        || flags & NI_NUMERICHOST != 0
        || reverse_resolvable(socket_addr(sa, len).map(|a| a.ip()))
    {
        Some(flags)
    } else if flags & NI_NAMEREQD != 0 {
        wsa_error(WSAHOST_NOT_FOUND);
        None
    } else {
        Some(flags | NI_NUMERICHOST)
    }
}

// Connects.

pub extern "system" fn connect(s: Socket, name: *const u8, len: i32) -> i32 {
    if !reachable("network connect blocked", name, len) {
        return SOCKET_ERROR;
    }
    host!("ws2_32.dll", "connect": extern "system" fn(Socket, *const u8, i32) -> i32)(s, name, len)
}

pub extern "system" fn WSAConnect(
    s: Socket,
    name: *const u8,
    len: i32,
    caller_data: *const c_void,
    callee_data: *mut c_void,
    sqos: *const c_void,
    gqos: *const c_void,
) -> i32 {
    if !reachable("network connect blocked", name, len) {
        return SOCKET_ERROR;
    }
    host!("ws2_32.dll", "WSAConnect": extern "system" fn(Socket, *const u8, i32, *const c_void, *mut c_void, *const c_void, *const c_void) -> i32)(
        s,
        name,
        len,
        caller_data,
        callee_data,
        sqos,
        gqos,
    )
}

pub extern "system" fn WSAConnectByNameA(
    s: Socket,
    node: *const c_char,
    service: *const c_char,
    local_len: *mut u32,
    local: *mut c_void,
    remote_len: *mut u32,
    remote: *mut c_void,
    timeout: *const c_void,
    reserved: *mut c_void,
) -> i32 {
    if !reachable_name(narrow(node)) {
        return 0;
    }
    host!("ws2_32.dll", "WSAConnectByNameA": extern "system" fn(Socket, *const c_char, *const c_char, *mut u32, *mut c_void, *mut u32, *mut c_void, *const c_void, *mut c_void) -> i32)(
        s, node, service, local_len, local, remote_len, remote, timeout, reserved,
    )
}

pub extern "system" fn WSAConnectByNameW(
    s: Socket,
    node: *const u16,
    service: *const u16,
    local_len: *mut u32,
    local: *mut c_void,
    remote_len: *mut u32,
    remote: *mut c_void,
    timeout: *const c_void,
    reserved: *mut c_void,
) -> i32 {
    if !reachable_name(wide(node)) {
        return 0;
    }
    host!("ws2_32.dll", "WSAConnectByNameW": extern "system" fn(Socket, *const u16, *const u16, *mut u32, *mut c_void, *mut u32, *mut c_void, *const c_void, *mut c_void) -> i32)(
        s, node, service, local_len, local, remote_len, remote, timeout, reserved,
    )
}

/// `SOCKET_ADDRESS`; a `SOCKET_ADDRESS_LIST` is an `i32` count, then these
/// (pointer aligned).
#[repr(C)]
struct SocketAddress {
    addr: *const u8,
    len: i32,
}

/// Every candidate must be loopback.
pub extern "system" fn WSAConnectByList(
    s: Socket,
    list: *const u8,
    local_len: *mut u32,
    local: *mut c_void,
    remote_len: *mut u32,
    remote: *mut c_void,
    timeout: *const c_void,
    reserved: *mut c_void,
) -> i32 {
    if !list.is_null() {
        // SAFETY: the title passes a `SOCKET_ADDRESS_LIST`.
        let entries = unsafe {
            let count = usize::try_from(list.cast::<i32>().read()).unwrap_or(0);
            std::slice::from_raw_parts(list.add(8).cast::<SocketAddress>(), count)
        };
        if !entries
            .iter()
            .all(|e| reachable("network connect blocked", e.addr, e.len))
        {
            return 0;
        }
    }
    host!("ws2_32.dll", "WSAConnectByList": extern "system" fn(Socket, *const u8, *mut u32, *mut c_void, *mut u32, *mut c_void, *const c_void, *mut c_void) -> i32)(
        s, list, local_len, local, remote_len, remote, timeout, reserved,
    )
}

pub extern "system" fn WSAJoinLeaf(
    s: Socket,
    name: *const u8,
    len: i32,
    caller_data: *const c_void,
    callee_data: *mut c_void,
    sqos: *const c_void,
    gqos: *const c_void,
    flags: u32,
) -> Socket {
    if !reachable("network connect blocked", name, len) {
        return INVALID_SOCKET;
    }
    host!("ws2_32.dll", "WSAJoinLeaf": extern "system" fn(Socket, *const u8, i32, *const c_void, *mut c_void, *const c_void, *const c_void, u32) -> Socket)(
        s,
        name,
        len,
        caller_data,
        callee_data,
        sqos,
        gqos,
        flags,
    )
}

/// `ConnectEx` as `WSAIoctl` hands it out: vetted, then the socket's own.
extern "system" fn connect_ex(
    s: Socket,
    name: *const u8,
    len: i32,
    send: *const c_void,
    send_len: u32,
    sent: *mut u32,
    overlapped: *mut c_void,
) -> i32 {
    if !reachable("network connect blocked", name, len) {
        return 0;
    }
    let (mut real, mut size) = (0usize, 0u32);
    let ioctl = host!("ws2_32.dll", "WSAIoctl": extern "system" fn(Socket, u32, *const c_void, u32, *mut c_void, u32, *mut u32, *mut c_void, *const c_void) -> i32);
    let guid = WSAID_CONNECTEX;
    let found = ioctl(
        s,
        SIO_GET_EXTENSION_FUNCTION_POINTER,
        (&raw const guid).cast(),
        16,
        (&raw mut real).cast(),
        8,
        &mut size,
        std::ptr::null_mut(),
        std::ptr::null(),
    );
    if found != 0 || real == 0 {
        return 0;
    }
    // SAFETY: the provider's `ConnectEx` for this socket.
    let real = unsafe {
        std::mem::transmute::<
            usize,
            extern "system" fn(
                Socket,
                *const u8,
                i32,
                *const c_void,
                u32,
                *mut u32,
                *mut c_void,
            ) -> i32,
        >(real)
    };
    real(s, name, len, send, send_len, sent, overlapped)
}

/// The `ConnectEx` and `WSASendMsg` pointers it hands out are ours; the
/// multi-pointer query (Registered I/O, whose `RIOSendEx` takes any
/// address) is refused as by an old system. Other controls pass through.
pub extern "system" fn WSAIoctl(
    s: Socket,
    code: u32,
    input: *const c_void,
    input_len: u32,
    output: *mut c_void,
    output_len: u32,
    returned: *mut u32,
    overlapped: *mut c_void,
    completion: *const c_void,
) -> i32 {
    if code == SIO_GET_MULTIPLE_EXTENSION_FUNCTION_POINTER {
        wsa_error(WSAEINVAL);
        return SOCKET_ERROR;
    }
    let result = host!("ws2_32.dll", "WSAIoctl": extern "system" fn(Socket, u32, *const c_void, u32, *mut c_void, u32, *mut u32, *mut c_void, *const c_void) -> i32)(
        s, code, input, input_len, output, output_len, returned, overlapped, completion,
    );
    if result == 0 && code == SIO_GET_EXTENSION_FUNCTION_POINTER && input_len >= 16 {
        // SAFETY: the host just read a GUID from `input` and wrote a pointer to `output`.
        unsafe {
            let ours = match input.cast::<Guid>().read_unaligned() {
                WSAID_CONNECTEX => connect_ex as *const () as usize,
                WSAID_WSASENDMSG => WSASendMsg as *const () as usize,
                _ => return result,
            };
            output.cast::<usize>().write_unaligned(ours);
        }
    }
    result
}

// Datagram sends.

/// Without an address it sends to the connected peer, which `connect` vetted;
/// so does a stream socket whatever `to` says (Winsock ignores it there).
pub extern "system" fn sendto(
    s: Socket,
    buf: *const u8,
    len: i32,
    flags: i32,
    to: *const u8,
    to_len: i32,
) -> i32 {
    if !to.is_null() && !stream(s) && !reachable("network datagram blocked", to, to_len) {
        return SOCKET_ERROR;
    }
    host!("ws2_32.dll", "sendto": extern "system" fn(Socket, *const u8, i32, i32, *const u8, i32) -> i32)(
        s, buf, len, flags, to, to_len,
    )
}

pub extern "system" fn WSASendTo(
    s: Socket,
    buffers: *const c_void,
    count: u32,
    sent: *mut u32,
    flags: u32,
    to: *const u8,
    to_len: i32,
    overlapped: *mut c_void,
    completion: *const c_void,
) -> i32 {
    if !to.is_null() && !stream(s) && !reachable("network datagram blocked", to, to_len) {
        return SOCKET_ERROR;
    }
    host!("ws2_32.dll", "WSASendTo": extern "system" fn(Socket, *const c_void, u32, *mut u32, u32, *const u8, i32, *mut c_void, *const c_void) -> i32)(
        s, buffers, count, sent, flags, to, to_len, overlapped, completion,
    )
}

/// A connection-oriented socket, whose sends go to the peer `connect` vetted.
fn stream(s: Socket) -> bool {
    const SOL_SOCKET: i32 = 0xffff;
    const SO_TYPE: i32 = 0x1008;
    const SOCK_STREAM: i32 = 1;
    let (mut kind, mut len) = (0i32, 4i32);
    let getsockopt = host!("ws2_32.dll", "getsockopt": extern "system" fn(Socket, i32, i32, *mut i32, *mut i32) -> i32);
    getsockopt(s, SOL_SOCKET, SO_TYPE, &mut kind, &mut len) == 0 && kind == SOCK_STREAM
}

/// The start of a `WSAMSG`: the destination.
#[repr(C)]
struct MessageName {
    name: *const u8,
    len: i32,
}

pub extern "system" fn WSASendMsg(
    s: Socket,
    message: *const c_void,
    flags: u32,
    sent: *mut u32,
    overlapped: *mut c_void,
    completion: *const c_void,
) -> i32 {
    // SAFETY: the title passes a `WSAMSG` or null.
    if let Some(m) = unsafe { message.cast::<MessageName>().as_ref() }
        && !m.name.is_null()
        && !reachable("network datagram blocked", m.name, m.len)
    {
        return SOCKET_ERROR;
    }
    host!("ws2_32.dll", "WSASendMsg": extern "system" fn(Socket, *const c_void, u32, *mut u32, *mut c_void, *const c_void) -> i32)(
        s, message, flags, sent, overlapped, completion,
    )
}

/// All interfaces become loopback, the only one an unplugged console has;
/// any other address is not ours (`WSAEADDRNOTAVAIL`).
// shortcut: `[::]` becomes `[::1]`, so a dual-stack listener no longer hears
// 127.0.0.1; bind both families if a title relies on that.
pub extern "system" fn bind(s: Socket, name: *const u8, len: i32) -> i32 {
    let host_bind = host!("ws2_32.dll", "bind": extern "system" fn(Socket, *const u8, i32) -> i32);
    match socket_addr(name, len) {
        Some(a) if loopback(a.ip()) => host_bind(s, name, len),
        Some(a) if a.ip().is_unspecified() => {
            once("network bind narrowed to loopback", a.to_string());
            // SAFETY: `socket_addr` checked `len` readable bytes.
            let mut local = unsafe { std::slice::from_raw_parts(name, len as usize) }.to_vec();
            match a {
                SocketAddr::V4(_) => local[4..8].copy_from_slice(&[127, 0, 0, 1]),
                SocketAddr::V6(_) => local[23] = 1,
            }
            host_bind(s, local.as_ptr(), len)
        }
        a => {
            once("network bind blocked", shown(a));
            wsa_error(WSAEADDRNOTAVAIL);
            SOCKET_ERROR
        }
    }
}

// Forward lookups.

pub extern "system" fn getaddrinfo(
    node: *const c_char,
    service: *const c_char,
    hints: *const c_void,
    result: *mut *mut c_void,
) -> i32 {
    if !resolvable(narrow(node)) {
        return WSAHOST_NOT_FOUND;
    }
    host!("ws2_32.dll", "getaddrinfo": extern "system" fn(*const c_char, *const c_char, *const c_void, *mut *mut c_void) -> i32)(
        node, service, hints, result,
    )
}

pub extern "system" fn GetAddrInfoW(
    node: *const u16,
    service: *const u16,
    hints: *const c_void,
    result: *mut *mut c_void,
) -> i32 {
    if !resolvable(wide(node)) {
        return WSAHOST_NOT_FOUND;
    }
    host!("ws2_32.dll", "GetAddrInfoW": extern "system" fn(*const u16, *const u16, *const c_void, *mut *mut c_void) -> i32)(
        node, service, hints, result,
    )
}

pub extern "system" fn GetAddrInfoExA(
    name: *const c_char,
    service: *const c_char,
    namespace: u32,
    provider: *const c_void,
    hints: *const c_void,
    result: *mut *mut c_void,
    timeout: *const c_void,
    overlapped: *mut c_void,
    completion: *const c_void,
    handle: *mut Handle,
) -> i32 {
    if !resolvable(narrow(name)) {
        return WSAHOST_NOT_FOUND;
    }
    host!("ws2_32.dll", "GetAddrInfoExA": extern "system" fn(*const c_char, *const c_char, u32, *const c_void, *const c_void, *mut *mut c_void, *const c_void, *mut c_void, *const c_void, *mut Handle) -> i32)(
        name, service, namespace, provider, hints, result, timeout, overlapped, completion, handle,
    )
}

pub extern "system" fn GetAddrInfoExW(
    name: *const u16,
    service: *const u16,
    namespace: u32,
    provider: *const c_void,
    hints: *const c_void,
    result: *mut *mut c_void,
    timeout: *const c_void,
    overlapped: *mut c_void,
    completion: *const c_void,
    handle: *mut Handle,
) -> i32 {
    if !resolvable(wide(name)) {
        return WSAHOST_NOT_FOUND;
    }
    host!("ws2_32.dll", "GetAddrInfoExW": extern "system" fn(*const u16, *const u16, u32, *const c_void, *const c_void, *mut *mut c_void, *const c_void, *mut c_void, *const c_void, *mut Handle) -> i32)(
        name, service, namespace, provider, hints, result, timeout, overlapped, completion, handle,
    )
}

pub extern "system" fn gethostbyname(name: *const c_char) -> *mut c_void {
    if !resolvable(narrow(name)) {
        return std::ptr::null_mut();
    }
    host!("ws2_32.dll", "gethostbyname": extern "system" fn(*const c_char) -> *mut c_void)(name)
}

pub extern "system" fn WSAAsyncGetHostByName(
    window: Handle,
    message: u32,
    name: *const c_char,
    buf: *mut c_char,
    buf_len: i32,
) -> Handle {
    if !resolvable(narrow(name)) {
        return std::ptr::null_mut();
    }
    host!("ws2_32.dll", "WSAAsyncGetHostByName": extern "system" fn(Handle, u32, *const c_char, *mut c_char, i32) -> Handle)(
        window, message, name, buf, buf_len,
    )
}

/// Name-space queries (DNS, also network lists and Bluetooth) are refused
/// as a console without a network answers them.
// shortcut: local name spaces are refused too; pass those through if a title
// needs one.
pub extern "system" fn WSALookupServiceBeginA(
    _restrictions: *const c_void,
    _flags: u32,
    _lookup: *mut Handle,
) -> i32 {
    once("network name-space query blocked", "any".into());
    wsa_error(WSASERVICE_NOT_FOUND);
    SOCKET_ERROR
}

/// See [`WSALookupServiceBeginA`].
pub extern "system" fn WSALookupServiceBeginW(
    restrictions: *const c_void,
    flags: u32,
    lookup: *mut Handle,
) -> i32 {
    WSALookupServiceBeginA(restrictions, flags, lookup)
}

// Reverse lookups.

pub extern "system" fn gethostbyaddr(addr: *const c_char, len: i32, kind: i32) -> *mut c_void {
    if !reverse_resolvable(raw_ip(addr, len, kind)) {
        wsa_error(WSAHOST_NOT_FOUND);
        return std::ptr::null_mut();
    }
    host!("ws2_32.dll", "gethostbyaddr": extern "system" fn(*const c_char, i32, i32) -> *mut c_void)(
        addr, len, kind,
    )
}

pub extern "system" fn WSAAsyncGetHostByAddr(
    window: Handle,
    message: u32,
    addr: *const c_char,
    len: i32,
    kind: i32,
    buf: *mut c_char,
    buf_len: i32,
) -> Handle {
    if !reverse_resolvable(raw_ip(addr, len, kind)) {
        wsa_error(WSAHOST_NOT_FOUND);
        return std::ptr::null_mut();
    }
    host!("ws2_32.dll", "WSAAsyncGetHostByAddr": extern "system" fn(Handle, u32, *const c_char, i32, i32, *mut c_char, i32) -> Handle)(
        window, message, addr, len, kind, buf, buf_len,
    )
}

pub extern "system" fn getnameinfo(
    sa: *const u8,
    sa_len: i32,
    host: *mut c_char,
    host_len: u32,
    serv: *mut c_char,
    serv_len: u32,
    flags: i32,
) -> i32 {
    let Some(flags) = name_info_flags(sa, sa_len, !host.is_null(), flags) else {
        return WSAHOST_NOT_FOUND;
    };
    host!("ws2_32.dll", "getnameinfo": extern "system" fn(*const u8, i32, *mut c_char, u32, *mut c_char, u32, i32) -> i32)(
        sa, sa_len, host, host_len, serv, serv_len, flags,
    )
}

pub extern "system" fn GetNameInfoW(
    sa: *const u8,
    sa_len: i32,
    host: *mut u16,
    host_len: u32,
    serv: *mut u16,
    serv_len: u32,
    flags: i32,
) -> i32 {
    let Some(flags) = name_info_flags(sa, sa_len, !host.is_null(), flags) else {
        return WSAHOST_NOT_FOUND;
    };
    host!("ws2_32.dll", "GetNameInfoW": extern "system" fn(*const u8, i32, *mut u16, u32, *mut u16, u32, i32) -> i32)(
        sa, sa_len, host, host_len, serv, serv_len, flags,
    )
}

// WinHTTP.

/// The one door to a server: requests, uploads and WebSockets all hang off
/// the handle this returns.
pub extern "system" fn WinHttpConnect(
    session: Handle,
    server: *const u16,
    port: u16,
    reserved: u32,
) -> Handle {
    if let Some(name) = wide(server)
        && !loopback_name(&name)
    {
        once("HTTP connection blocked", format!("{name}:{port}"));
        // SAFETY: no pointers.
        unsafe { SetLastError(ERROR_WINHTTP_CANNOT_CONNECT) };
        return std::ptr::null_mut();
    }
    host!("winhttp.dll", "WinHttpConnect": extern "system" fn(Handle, *const u16, u16, u32) -> Handle)(
        session, server, port, reserved,
    )
}

/// Sessions never use a proxy: only loopback is reachable, and a proxy
/// (named, or found by auto-detection) would carry it off the machine.
pub extern "system" fn WinHttpOpen(
    agent: *const u16,
    access: u32,
    _proxy: *const u16,
    _bypass: *const u16,
    flags: u32,
) -> Handle {
    if access != WINHTTP_ACCESS_TYPE_NO_PROXY {
        once("HTTP proxy ignored", format!("access type {access}"));
    }
    host!("winhttp.dll", "WinHttpOpen": extern "system" fn(*const u16, u32, *const u16, *const u16, u32) -> Handle)(
        agent,
        WINHTTP_ACCESS_TYPE_NO_PROXY,
        std::ptr::null(),
        std::ptr::null(),
        flags,
    )
}

/// See [`WinHttpOpen`]: a proxy set later is accepted and not used. A handle
/// nobody opened is the host's to refuse.
pub extern "system" fn WinHttpSetOption(
    handle: Handle,
    option: u32,
    buffer: *const c_void,
    len: u32,
) -> i32 {
    if option == WINHTTP_OPTION_PROXY && !handle.is_null() {
        once("HTTP proxy ignored", "WINHTTP_OPTION_PROXY".into());
        return 1;
    }
    host!("winhttp.dll", "WinHttpSetOption": extern "system" fn(Handle, u32, *const c_void, u32) -> i32)(
        handle, option, buffer, len,
    )
}

/// Proxy auto-detection asks the network (DHCP, DNS, a PAC download).
fn no_auto_proxy() -> u32 {
    once("HTTP proxy auto-detection blocked", "WPAD/PAC".into());
    ERROR_WINHTTP_AUTODETECTION_FAILED
}

pub extern "system" fn WinHttpDetectAutoProxyConfigUrl(_flags: u32, _url: *mut *mut u16) -> i32 {
    // SAFETY: no pointers.
    unsafe { SetLastError(no_auto_proxy()) };
    0
}

pub extern "system" fn WinHttpGetProxyForUrl(
    _session: Handle,
    _url: *const u16,
    _options: *const c_void,
    _info: *mut c_void,
) -> i32 {
    // SAFETY: no pointers.
    unsafe { SetLastError(no_auto_proxy()) };
    0
}

pub extern "system" fn WinHttpGetProxyForUrlEx(
    _resolver: Handle,
    _url: *const u16,
    _options: *const c_void,
    _context: usize,
) -> u32 {
    no_auto_proxy()
}

pub extern "system" fn WinHttpGetProxyForUrlEx2(
    _resolver: Handle,
    _url: *const u16,
    _options: *const c_void,
    _selection_len: u32,
    _selection: *const u8,
    _context: usize,
) -> u32 {
    no_auto_proxy()
}

pub extern "system" fn WinHttpGetProxySettingsEx(
    _resolver: Handle,
    _kind: i32,
    _params: *const c_void,
    _context: usize,
) -> u32 {
    no_auto_proxy()
}

/// Winsock imports by ordinal; the Winsock 1.1 numbers are fixed.
fn ordinal(name: &'static str, f: usize) -> HleExport {
    HleExport { name, address: f }
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "net",
        dlls: &["ws2_32.dll", "winhttp.dll"],
        exports: vec![
            hle_export!(connect),
            hle_export!(WSAConnect),
            hle_export!(WSAConnectByNameA),
            hle_export!(WSAConnectByNameW),
            hle_export!(WSAConnectByList),
            hle_export!(WSAJoinLeaf),
            hle_export!(WSAIoctl),
            hle_export!(sendto),
            hle_export!(WSASendTo),
            hle_export!(WSASendMsg),
            hle_export!(bind),
            hle_export!(getaddrinfo),
            hle_export!(GetAddrInfoW),
            hle_export!(GetAddrInfoExA),
            hle_export!(GetAddrInfoExW),
            hle_export!(gethostbyname),
            hle_export!(WSAAsyncGetHostByName),
            hle_export!(WSALookupServiceBeginA),
            hle_export!(WSALookupServiceBeginW),
            hle_export!(gethostbyaddr),
            hle_export!(WSAAsyncGetHostByAddr),
            hle_export!(getnameinfo),
            hle_export!(GetNameInfoW),
            hle_export!(WinHttpConnect),
            hle_export!(WinHttpOpen),
            hle_export!(WinHttpSetOption),
            hle_export!(WinHttpDetectAutoProxyConfigUrl),
            hle_export!(WinHttpGetProxyForUrl),
            hle_export!(WinHttpGetProxyForUrlEx),
            hle_export!(WinHttpGetProxyForUrlEx2),
            hle_export!(WinHttpGetProxySettingsEx),
            ordinal("#2", bind as *const () as usize),
            ordinal("#4", connect as *const () as usize),
            ordinal("#20", sendto as *const () as usize),
            ordinal("#51", gethostbyaddr as *const () as usize),
            ordinal("#52", gethostbyname as *const () as usize),
            ordinal("#102", WSAAsyncGetHostByAddr as *const () as usize),
            ordinal("#103", WSAAsyncGetHostByName as *const () as usize),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_pe::ImportSymbol;

    // Synthetic addresses only. TEST-NET-1 (192.0.2.0/24) and the IPv6
    // documentation prefix are never routed, and no test tries them.
    const AWAY: [u8; 4] = [192, 0, 2, 1];
    const HOME: [u8; 4] = [127, 0, 0, 1];
    const AWAY6: [u8; 16] = [0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
    const HOME6: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];

    fn v4(ip: [u8; 4], port: u16) -> [u8; 16] {
        let mut addr = [0u8; 16];
        addr[..2].copy_from_slice(&AF_INET.to_le_bytes());
        addr[2..4].copy_from_slice(&port.to_be_bytes());
        addr[4..8].copy_from_slice(&ip);
        addr
    }

    fn v6(ip: [u8; 16], port: u16) -> [u8; 28] {
        let mut addr = [0u8; 28];
        addr[..2].copy_from_slice(&AF_INET6.to_le_bytes());
        addr[2..4].copy_from_slice(&port.to_be_bytes());
        addr[8..24].copy_from_slice(&ip);
        addr
    }

    /// `text` as a title passes it to a `W` function.
    fn utf16(text: &str) -> Vec<u16> {
        text.encode_utf16().chain([0]).collect()
    }

    /// Winsock starts on the first socket the process opens.
    fn up() {
        drop(std::net::TcpListener::bind("127.0.0.1:0").unwrap());
    }

    fn open(family: u16, kind: i32, protocol: i32) -> Socket {
        host!("ws2_32.dll", "socket": extern "system" fn(i32, i32, i32) -> Socket)(
            i32::from(family),
            kind,
            protocol,
        )
    }

    fn close(socket: Socket) {
        let _ = host!("ws2_32.dll", "closesocket": extern "system" fn(Socket) -> i32)(socket);
    }

    fn last_error() -> i32 {
        host!("ws2_32.dll", "WSAGetLastError": extern "system" fn() -> i32)()
    }

    fn last_error_win32() -> u32 {
        // SAFETY: no arguments.
        unsafe { windows_sys::Win32::Foundation::GetLastError() }
    }

    /// The address the kernel gave `socket`, and how many bytes of it.
    fn bound_address(socket: Socket) -> ([u8; 128], i32) {
        let mut addr = [0u8; 128];
        let mut len = addr.len() as i32;
        assert_eq!(
            host!("ws2_32.dll", "getsockname": extern "system" fn(Socket, *mut u8, *mut i32) -> i32)(
                socket,
                addr.as_mut_ptr(),
                &mut len
            ),
            0,
            "getsockname: {}",
            last_error()
        );
        (addr, len)
    }

    fn free_addrinfo(result: *mut c_void, wide: bool) {
        if wide {
            host!("ws2_32.dll", "FreeAddrInfoW": extern "system" fn(*mut c_void))(result);
        } else {
            host!("ws2_32.dll", "freeaddrinfo": extern "system" fn(*mut c_void))(result);
        }
    }

    /// A NUL-terminated narrow string the host wrote.
    fn cstr(text: &[c_char]) -> String {
        // SAFETY: the call that filled `text` terminated it.
        unsafe { CStr::from_ptr(text.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }

    /// A `SOCKET_ADDRESS_LIST`: an `i32` count, then `(pointer, length)`
    /// pairs at 8 and 16 bytes. A `Vec<u64>` keeps it aligned for them.
    fn address_list(entries: &[[u8; 16]]) -> Vec<u64> {
        let mut list = vec![0u64; 1 + 2 * entries.len()];
        list[0] = u64::from(u32::try_from(entries.len()).unwrap());
        for (i, entry) in entries.iter().enumerate() {
            list[1 + 2 * i] = entry.as_ptr() as usize as u64;
            list[2 + 2 * i] = 16;
        }
        list
    }

    /// The first `WSAMSG` fields: where a datagram is going.
    #[repr(C)]
    struct WsaBuf {
        len: u32,
        buf: *mut u8,
    }

    #[repr(C)]
    struct WsaMsg {
        name: *const u8,
        namelen: i32,
        buffers: *const WsaBuf,
        count: u32,
        control: WsaBuf,
        flags: u32,
    }

    /// `ADDRINFOW`, as the host lays it out.
    #[repr(C)]
    struct AddrInfo {
        flags: i32,
        family: i32,
        kind: i32,
        protocol: i32,
        addr_len: usize,
        canon_name: *mut u16,
        addr: *const u8,
        next: *mut AddrInfo,
    }

    #[test]
    fn sockaddrs_are_read_by_family_and_length() {
        assert_eq!(
            socket_addr(v4(HOME, 8080).as_ptr(), 16),
            Some(SocketAddr::from((HOME, 8080)))
        );
        assert_eq!(
            socket_addr(v6(HOME6, 443).as_ptr(), 28),
            Some(SocketAddr::from((HOME6, 443)))
        );
        assert_eq!(
            socket_addr(v6(AWAY6, 0x1234).as_ptr(), 28),
            Some(SocketAddr::from((AWAY6, 0x1234)))
        );
        // The port is in network order.
        let mut network = v4(HOME, 0);
        network[2] = 0x12;
        network[3] = 0x34;
        assert_eq!(socket_addr(network.as_ptr(), 16).unwrap().port(), 0x1234);

        // Too short, not a family we know, or no buffer at all.
        assert_eq!(socket_addr(std::ptr::null(), 16), None);
        assert_eq!(socket_addr(network.as_ptr(), 0), None);
        assert_eq!(socket_addr(network.as_ptr(), 3), None);
        assert_eq!(socket_addr(network.as_ptr(), 15), None);
        assert_eq!(socket_addr(network.as_ptr(), -1), None);
        assert_eq!(socket_addr(v6(HOME6, 80).as_ptr(), 27), None);
        let mut unknown = v4(HOME, 80);
        unknown[..2].copy_from_slice(&99u16.to_le_bytes());
        assert_eq!(socket_addr(unknown.as_ptr(), 16), None);
        // A `sockaddr_storage` is longer than what it holds.
        let mut storage = [0u8; 128];
        storage[..16].copy_from_slice(&v4(HOME, 80));
        assert_eq!(
            socket_addr(storage.as_ptr(), 128),
            Some(SocketAddr::from((HOME, 80)))
        );
    }

    #[test]
    fn raw_addresses_are_read_by_family() {
        assert_eq!(
            raw_ip(HOME.as_ptr().cast(), 4, i32::from(AF_INET)),
            Some(IpAddr::from(HOME))
        );
        assert_eq!(
            raw_ip(HOME6.as_ptr().cast(), 16, i32::from(AF_INET6)),
            Some(IpAddr::from(HOME6))
        );
        // Lengths and families that cannot carry an address.
        assert_eq!(raw_ip(HOME.as_ptr().cast(), 3, i32::from(AF_INET)), None);
        assert_eq!(raw_ip(HOME6.as_ptr().cast(), 15, i32::from(AF_INET6)), None);
        assert_eq!(raw_ip(HOME.as_ptr().cast(), 4, 99), None);
        assert_eq!(raw_ip(HOME.as_ptr().cast(), -4, i32::from(AF_INET)), None);
        assert_eq!(raw_ip(std::ptr::null(), 4, i32::from(AF_INET)), None);
        // A longer buffer still reads the address at its start.
        let mut long = [0u8; 16];
        long[..4].copy_from_slice(&AWAY);
        assert_eq!(
            raw_ip(long.as_ptr().cast(), 16, i32::from(AF_INET)),
            Some(IpAddr::from(AWAY))
        );
    }

    #[test]
    fn only_loopback_is_local() {
        for ip in ["127.0.0.1", "127.0.0.2", "127.255.255.255"] {
            assert!(loopback(ip.parse().unwrap()), "{ip}");
        }
        for ip in [
            "0.0.0.0",
            "126.255.255.255",
            "128.0.0.1",
            "192.0.2.1",
            "::",
            "::2",
            "2001:db8::1",
        ] {
            assert!(!loopback(ip.parse().unwrap()), "{ip}");
        }
        assert!(loopback("::1".parse().unwrap()));
        // An IPv4-mapped loopback address is a loopback address.
        assert!(loopback("::ffff:127.0.0.1".parse().unwrap()));
        assert!(!loopback("::ffff:192.0.2.1".parse().unwrap()));
    }

    #[test]
    fn names_are_local_only_for_localhost_and_loopback_literals() {
        for name in [
            "localhost",
            "LocalHost",
            "127.0.0.1",
            "127.1.2.3",
            "::1",
            "[::1]",
            "[::ffff:127.0.0.1]",
        ] {
            assert!(loopback_name(name), "{name}");
        }
        for name in [
            "",
            "example.com",
            "localhost.example.com",
            "192.0.2.1",
            "0.0.0.0",
            "::2",
            "[::2]",
        ] {
            assert!(!loopback_name(name), "{name}");
        }
    }

    #[test]
    fn names_are_read_as_narrow_or_utf16() {
        assert_eq!(narrow(std::ptr::null()), None);
        assert_eq!(narrow(c"localhost".as_ptr()).as_deref(), Some("localhost"));
        assert_eq!(narrow(c"".as_ptr()).as_deref(), Some(""));
        // ANSI is not UTF-8: an odd byte becomes a replacement character,
        // never part of a name that would match.
        let ansi = [b'c', b'a', b'f', 0xe9, 0];
        assert_eq!(narrow(ansi.as_ptr().cast()).as_deref(), Some("caf\u{fffd}"));

        assert_eq!(wide(std::ptr::null()), None);
        assert_eq!(
            wide(utf16("localhost").as_ptr()).as_deref(),
            Some("localhost")
        );
        assert_eq!(wide(utf16("").as_ptr()).as_deref(), Some(""));
        let unicode = "h\u{e9}llo/\u{65e5}\u{672c}";
        assert_eq!(wide(utf16(unicode).as_ptr()).as_deref(), Some(unicode));
        // A surrogate pair is one character, not two replacements.
        assert_eq!(
            wide(utf16("\u{1f680}").as_ptr()).as_deref(),
            Some("\u{1f680}")
        );
    }

    #[test]
    fn lookups_answer_without_asking_the_network() {
        up();
        assert!(resolvable(None));
        assert!(resolvable(Some(String::new())));
        assert!(resolvable(Some("localhost".into())));
        assert!(resolvable(Some("LocalHost".into())));
        assert!(resolvable(Some("127.0.0.1".into())));
        assert!(resolvable(Some("::1".into())));
        assert!(!resolvable(Some("example.com".into())));
        assert_eq!(last_error(), WSAHOST_NOT_FOUND);

        // Classifying does not touch the error: the caller sets it.
        wsa_error(0);
        assert!(reverse_resolvable(Some("127.0.0.1".parse().unwrap())));
        assert!(reverse_resolvable(Some("::1".parse().unwrap())));
        assert!(!reverse_resolvable(Some("192.0.2.1".parse().unwrap())));
        assert_eq!(last_error(), 0);
        assert!(!reverse_resolvable(None));
        assert_eq!(last_error(), 0);

        let home = v4(HOME, 80);
        let away = v4(AWAY, 80);
        // No host wanted: nothing to decide.
        assert_eq!(name_info_flags(home.as_ptr(), 16, false, 0), Some(0));
        // The numeric form is already local.
        assert_eq!(
            name_info_flags(home.as_ptr(), 16, true, NI_NUMERICHOST),
            Some(NI_NUMERICHOST)
        );
        // Loopback may be named; nothing else may.
        assert_eq!(name_info_flags(home.as_ptr(), 16, true, 0), Some(0));
        assert_eq!(
            name_info_flags(away.as_ptr(), 16, true, 0),
            Some(NI_NUMERICHOST)
        );
        assert_eq!(
            name_info_flags(away.as_ptr(), 16, false, NI_NAMEREQD),
            Some(NI_NAMEREQD)
        );
        assert_eq!(name_info_flags(away.as_ptr(), 16, true, NI_NAMEREQD), None);
        assert_eq!(last_error(), WSAHOST_NOT_FOUND);
    }

    #[test]
    fn a_bind_to_any_interface_is_narrowed_to_loopback() {
        up();
        let socket = open(AF_INET, 2, 17);
        assert_eq!(
            bind(socket, v4([0, 0, 0, 0], 0).as_ptr(), 16),
            0,
            "{}",
            last_error()
        );
        let (addr, len) = bound_address(socket);
        assert_eq!(len, 16);
        assert_eq!(&addr[4..8], &HOME);
        // A port was picked, so the caller can bind and listen.
        assert_ne!(addr[2], 0);
        close(socket);

        // A loopback bind is passed through untouched.
        let socket = open(AF_INET, 2, 17);
        assert_eq!(
            bind(socket, v4(HOME, 0).as_ptr(), 16),
            0,
            "{}",
            last_error()
        );
        let (addr, _) = bound_address(socket);
        assert_eq!(&addr[4..8], &HOME);
        close(socket);

        // Anything else is not ours to bind.
        let socket = open(AF_INET, 2, 17);
        assert_eq!(bind(socket, v4(AWAY, 0).as_ptr(), 16), SOCKET_ERROR);
        assert_eq!(last_error(), WSAEADDRNOTAVAIL);
        assert_eq!(bind(socket, std::ptr::null(), 16), SOCKET_ERROR);
        assert_eq!(last_error(), WSAEADDRNOTAVAIL);
        close(socket);

        // `[::]` becomes `[::1]` the same way.
        let Ok(listener) = std::net::UdpSocket::bind("[::1]:0") else {
            return;
        };
        drop(listener);
        let socket = open(AF_INET6, 2, 17);
        assert_eq!(
            bind(socket, v6([0; 16], 0).as_ptr(), 28),
            0,
            "{}",
            last_error()
        );
        let (addr, len) = bound_address(socket);
        assert_eq!(len, 28);
        assert_eq!(&addr[8..24], &HOME6);
        close(socket);
    }

    #[test]
    fn connects_are_vetted_by_address() {
        up();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        let socket = open(AF_INET, 1, 6);
        assert_eq!(connect(socket, v4(AWAY, 80).as_ptr(), 16), SOCKET_ERROR);
        assert_eq!(last_error(), WSAENETUNREACH);
        // No address at all is still no route.
        assert_eq!(connect(socket, std::ptr::null(), 16), SOCKET_ERROR);
        assert_eq!(last_error(), WSAENETUNREACH);
        assert_eq!(
            connect(socket, v4(HOME, port).as_ptr(), 16),
            0,
            "{}",
            last_error()
        );
        listener.accept().unwrap();
        close(socket);

        let socket = open(AF_INET, 1, 6);
        assert_eq!(
            WSAConnect(
                socket,
                v4(AWAY, 80).as_ptr(),
                16,
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null()
            ),
            SOCKET_ERROR
        );
        assert_eq!(last_error(), WSAENETUNREACH);
        close(socket);

        // Joining a group is a connect for our purposes.
        let socket = open(AF_INET, 2, 17);
        assert_eq!(
            WSAJoinLeaf(
                socket,
                v4(AWAY, 80).as_ptr(),
                16,
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
                0
            ),
            INVALID_SOCKET
        );
        assert_eq!(last_error(), WSAENETUNREACH);
        close(socket);

        let Ok(listener) = std::net::TcpListener::bind("[::1]:0") else {
            return;
        };
        let port = listener.local_addr().unwrap().port();
        let socket = open(AF_INET6, 1, 6);
        assert_eq!(connect(socket, v6(AWAY6, 80).as_ptr(), 28), SOCKET_ERROR);
        assert_eq!(last_error(), WSAENETUNREACH);
        assert_eq!(
            connect(socket, v6(HOME6, port).as_ptr(), 28),
            0,
            "{}",
            last_error()
        );
        listener.accept().unwrap();
        close(socket);
    }

    #[test]
    fn connects_are_vetted_by_name() {
        up();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let service = std::ffi::CString::new(port.to_string()).unwrap();
        let wide_service = utf16(&port.to_string());
        let lengths: *mut u32 = std::ptr::null_mut();
        let null = std::ptr::null_mut::<c_void>();

        let socket = open(AF_INET, 1, 6);
        // A name would need a lookup, an address a route.
        assert_eq!(
            WSAConnectByNameA(
                socket,
                c"example.com".as_ptr(),
                c"80".as_ptr(),
                lengths,
                null,
                lengths,
                null,
                null,
                null
            ),
            0
        );
        assert_eq!(last_error(), WSAHOST_NOT_FOUND);
        assert_eq!(
            WSAConnectByNameA(
                socket,
                c"192.0.2.1".as_ptr(),
                c"80".as_ptr(),
                lengths,
                null,
                lengths,
                null,
                null,
                null
            ),
            0
        );
        assert_eq!(last_error(), WSAENETUNREACH);
        // `localhost` is the one name answered.
        assert_eq!(
            WSAConnectByNameA(
                socket,
                c"localhost".as_ptr(),
                service.as_ptr(),
                lengths,
                null,
                lengths,
                null,
                null,
                null
            ),
            1,
            "{}",
            last_error()
        );
        listener.accept().unwrap();
        close(socket);

        let socket = open(AF_INET, 1, 6);
        let name = utf16("example.com");
        assert_eq!(
            WSAConnectByNameW(
                socket,
                name.as_ptr(),
                wide_service.as_ptr(),
                lengths,
                null,
                lengths,
                null,
                null,
                null
            ),
            0
        );
        assert_eq!(last_error(), WSAHOST_NOT_FOUND);
        let name = utf16("192.0.2.1");
        assert_eq!(
            WSAConnectByNameW(
                socket,
                name.as_ptr(),
                wide_service.as_ptr(),
                lengths,
                null,
                lengths,
                null,
                null,
                null
            ),
            0
        );
        assert_eq!(last_error(), WSAENETUNREACH);
        let name = utf16("127.0.0.1");
        assert_eq!(
            WSAConnectByNameW(
                socket,
                name.as_ptr(),
                wide_service.as_ptr(),
                lengths,
                null,
                lengths,
                null,
                null,
                null
            ),
            1,
            "{}",
            last_error()
        );
        listener.accept().unwrap();
        close(socket);

        // Every candidate of a list must be loopback.
        let home = [v4(HOME, port)];
        let away = [v4(AWAY, 80)];
        let socket = open(AF_INET, 1, 6);
        let list = address_list(&home);
        assert_eq!(
            WSAConnectByList(
                socket,
                list.as_ptr().cast(),
                lengths,
                null,
                lengths,
                null,
                null,
                null
            ),
            1,
            "{}",
            last_error()
        );
        listener.accept().unwrap();
        close(socket);

        let socket = open(AF_INET, 1, 6);
        let list = address_list(&away);
        assert_eq!(
            WSAConnectByList(
                socket,
                list.as_ptr().cast(),
                lengths,
                null,
                lengths,
                null,
                null,
                null
            ),
            0
        );
        assert_eq!(last_error(), WSAENETUNREACH);
        close(socket);

        // One remote candidate spoils the list, before anything connects.
        let mixed = [v4(HOME, port), v4(AWAY, 80)];
        let socket = open(AF_INET, 1, 6);
        let list = address_list(&mixed);
        assert_eq!(
            WSAConnectByList(
                socket,
                list.as_ptr().cast(),
                lengths,
                null,
                lengths,
                null,
                null,
                null
            ),
            0
        );
        assert_eq!(last_error(), WSAENETUNREACH);
        listener.set_nonblocking(true).unwrap();
        assert!(listener.accept().is_err(), "a blocked list connected");
        close(socket);

        // A count the host cannot honour is refused, not vetted into a pass.
        let mut negative = address_list(&away);
        negative[0] = u64::from(u32::MAX);
        let socket = open(AF_INET, 1, 6);
        assert_eq!(
            WSAConnectByList(
                socket,
                negative.as_ptr().cast(),
                lengths,
                null,
                lengths,
                null,
                null,
                null
            ),
            0
        );
        close(socket);
    }

    #[test]
    fn connect_ex_and_send_msg_come_from_the_hle() {
        up();
        let socket = open(AF_INET, 1, 6);
        let null = std::ptr::null_mut::<c_void>();
        let mut returned = 0u32;

        let connect_guid = WSAID_CONNECTEX;
        let mut pointer = 0usize;
        assert_eq!(
            WSAIoctl(
                socket,
                SIO_GET_EXTENSION_FUNCTION_POINTER,
                (&raw const connect_guid).cast(),
                16,
                (&raw mut pointer).cast(),
                8,
                &mut returned,
                null,
                null
            ),
            0,
            "{}",
            last_error()
        );
        assert_eq!(pointer, connect_ex as *const () as usize);

        let send_msg_guid = WSAID_WSASENDMSG;
        assert_eq!(
            WSAIoctl(
                socket,
                SIO_GET_EXTENSION_FUNCTION_POINTER,
                (&raw const send_msg_guid).cast(),
                16,
                (&raw mut pointer).cast(),
                8,
                &mut returned,
                null,
                null
            ),
            0,
            "{}",
            last_error()
        );
        assert_eq!(pointer, WSASendMsg as *const () as usize);

        // The multi-pointer query (Registered I/O) is refused as by an old
        // system, before any argument is looked at.
        assert_eq!(
            WSAIoctl(
                socket,
                SIO_GET_MULTIPLE_EXTENSION_FUNCTION_POINTER,
                (&raw const connect_guid).cast(),
                16,
                (&raw mut pointer).cast(),
                8,
                &mut returned,
                null,
                null
            ),
            SOCKET_ERROR
        );
        assert_eq!(last_error(), WSAEINVAL);
        assert_eq!(
            WSAIoctl(
                INVALID_SOCKET,
                SIO_GET_MULTIPLE_EXTENSION_FUNCTION_POINTER,
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
                null,
                null
            ),
            SOCKET_ERROR
        );
        assert_eq!(last_error(), WSAEINVAL);

        // A GUID we do not know is the host's pointer, or its refusal.
        let foreign = Guid(0, 0, 0, [0; 8]);
        let mut untouched = 0usize;
        assert_ne!(
            WSAIoctl(
                socket,
                SIO_GET_EXTENSION_FUNCTION_POINTER,
                (&raw const foreign).cast(),
                16,
                (&raw mut untouched).cast(),
                8,
                &mut returned,
                null,
                null
            ),
            0
        );
        assert_eq!(untouched, 0);

        // A short output buffer or a short input is never written to.
        let mut small = 0u32;
        assert_ne!(
            WSAIoctl(
                socket,
                SIO_GET_EXTENSION_FUNCTION_POINTER,
                (&raw const send_msg_guid).cast(),
                16,
                (&raw mut small).cast(),
                4,
                &mut returned,
                null,
                null
            ),
            0
        );
        assert_eq!(small, 0);
        let mut pointer = usize::MAX;
        assert_ne!(
            WSAIoctl(
                socket,
                SIO_GET_EXTENSION_FUNCTION_POINTER,
                (&raw const send_msg_guid).cast(),
                15,
                (&raw mut pointer).cast(),
                8,
                &mut returned,
                null,
                null
            ),
            0
        );
        assert_eq!(pointer, usize::MAX);

        // The `ConnectEx` handed out is the vetted one.
        assert_eq!(
            connect_ex(
                socket,
                v4(AWAY, 80).as_ptr(),
                16,
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
                null
            ),
            0
        );
        assert_eq!(last_error(), WSAENETUNREACH);
        close(socket);
    }

    #[test]
    fn datagrams_to_the_network_are_dropped() {
        up();
        let listener = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        listener
            .set_read_timeout(Some(std::time::Duration::from_millis(500)))
            .unwrap();
        let port = listener.local_addr().unwrap().port();
        let home = v4(HOME, port);
        let away = v4(AWAY, 9);
        let mut got = [0u8; 8];

        let socket = open(AF_INET, 2, 17);
        assert_eq!(
            sendto(socket, b"x".as_ptr(), 1, 0, away.as_ptr(), 16),
            SOCKET_ERROR
        );
        assert_eq!(last_error(), WSAENETUNREACH);
        assert!(
            listener.recv_from(&mut got).is_err(),
            "a blocked datagram arrived"
        );
        assert_eq!(
            sendto(socket, b"x".as_ptr(), 1, 0, home.as_ptr(), 16),
            1,
            "{}",
            last_error()
        );
        assert_eq!(listener.recv_from(&mut got).unwrap().0, 1);
        close(socket);

        let socket = open(AF_INET, 2, 17);
        let mut payload = *b"y";
        let buffers = [WsaBuf {
            len: 1,
            buf: payload.as_mut_ptr(),
        }];
        let mut sent = 0u32;
        assert_eq!(
            WSASendTo(
                socket,
                buffers.as_ptr().cast(),
                1,
                &mut sent,
                0,
                away.as_ptr(),
                16,
                std::ptr::null_mut(),
                std::ptr::null()
            ),
            SOCKET_ERROR
        );
        assert_eq!(last_error(), WSAENETUNREACH);
        assert!(
            listener.recv_from(&mut got).is_err(),
            "a blocked datagram arrived"
        );
        assert_eq!(
            WSASendTo(
                socket,
                buffers.as_ptr().cast(),
                1,
                &mut sent,
                0,
                home.as_ptr(),
                16,
                std::ptr::null_mut(),
                std::ptr::null()
            ),
            0,
            "{}",
            last_error()
        );
        assert_eq!(sent, 1);
        assert_eq!(listener.recv_from(&mut got).unwrap().0, 1);
        close(socket);

        let socket = open(AF_INET, 2, 17);
        let mut payload = *b"z";
        let buffers = [WsaBuf {
            len: 1,
            buf: payload.as_mut_ptr(),
        }];
        let mut msg = WsaMsg {
            name: away.as_ptr(),
            namelen: 16,
            buffers: buffers.as_ptr(),
            count: 1,
            control: WsaBuf {
                len: 0,
                buf: std::ptr::null_mut(),
            },
            flags: 0,
        };
        let mut sent = 0u32;
        assert_eq!(
            WSASendMsg(
                socket,
                (&raw const msg).cast(),
                0,
                &mut sent,
                std::ptr::null_mut(),
                std::ptr::null()
            ),
            SOCKET_ERROR
        );
        assert_eq!(last_error(), WSAENETUNREACH);
        assert!(
            listener.recv_from(&mut got).is_err(),
            "a blocked datagram arrived"
        );
        msg.name = home.as_ptr();
        assert_eq!(
            WSASendMsg(
                socket,
                (&raw const msg).cast(),
                0,
                &mut sent,
                std::ptr::null_mut(),
                std::ptr::null()
            ),
            0,
            "{}",
            last_error()
        );
        assert_eq!(sent, 1);
        assert_eq!(listener.recv_from(&mut got).unwrap().0, 1);
        close(socket);

        // A stream socket ignores `to`: the bytes reach the vetted peer.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let socket = open(AF_INET, 1, 6);
        assert_eq!(
            connect(socket, v4(HOME, port).as_ptr(), 16),
            0,
            "{}",
            last_error()
        );
        let (mut peer, _) = listener.accept().unwrap();
        peer.set_read_timeout(Some(std::time::Duration::from_millis(500)))
            .unwrap();
        assert_eq!(sendto(socket, b"s".as_ptr(), 1, 0, away.as_ptr(), 16), 1);
        let mut byte = [0u8; 1];
        assert_eq!(std::io::Read::read(&mut peer, &mut byte).unwrap(), 1);
        assert_eq!(byte, *b"s");
        let mut payload = *b"b";
        let buffers = [WsaBuf {
            len: 1,
            buf: payload.as_mut_ptr(),
        }];
        let mut sent = 0u32;
        assert_eq!(
            WSASendTo(
                socket,
                buffers.as_ptr().cast(),
                1,
                &mut sent,
                0,
                away.as_ptr(),
                16,
                std::ptr::null_mut(),
                std::ptr::null()
            ),
            0,
            "{}",
            last_error()
        );
        assert_eq!(sent, 1);
        assert_eq!(std::io::Read::read(&mut peer, &mut byte).unwrap(), 1);
        assert_eq!(byte, *b"b");
        close(socket);
    }

    #[test]
    fn forward_lookups_only_answer_local_names() {
        up();
        let mut result = std::ptr::null_mut();
        assert_eq!(
            getaddrinfo(
                c"example.com".as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                &mut result
            ),
            WSAHOST_NOT_FOUND
        );
        assert!(result.is_null());
        let name = utf16("example.com");
        assert_eq!(
            GetAddrInfoW(
                name.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                &mut result
            ),
            WSAHOST_NOT_FOUND
        );
        assert_eq!(
            GetAddrInfoExA(
                c"example.com".as_ptr(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                &mut result,
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null_mut()
            ),
            WSAHOST_NOT_FOUND
        );
        assert_eq!(
            GetAddrInfoExW(
                name.as_ptr(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                &mut result,
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null_mut()
            ),
            WSAHOST_NOT_FOUND
        );

        // A null name only asks about the machine itself.
        assert_eq!(
            getaddrinfo(
                std::ptr::null(),
                c"80".as_ptr(),
                std::ptr::null(),
                &mut result
            ),
            0,
            "{}",
            last_error()
        );
        assert!(!result.is_null());
        free_addrinfo(result, false);

        // `localhost` resolves, and to nothing but loopback.
        let localhost = utf16("localhost");
        assert_eq!(
            GetAddrInfoW(
                localhost.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                &mut result
            ),
            0,
            "{}",
            last_error()
        );
        let mut count = 0;
        let mut node = result.cast::<AddrInfo>();
        while !node.is_null() {
            // SAFETY: an entry of the list the host just returned.
            let info = unsafe { &*node };
            let addr = socket_addr(info.addr, info.addr_len as i32).expect("an address");
            assert!(loopback(addr.ip()), "{addr} is not loopback");
            count += 1;
            node = info.next;
        }
        assert!(count > 0, "localhost resolved to nothing");
        free_addrinfo(result, true);

        assert!(gethostbyname(c"example.com".as_ptr()).is_null());
        assert_eq!(last_error(), WSAHOST_NOT_FOUND);
        assert!(!gethostbyname(c"localhost".as_ptr()).is_null());
        let mut buf = [0 as c_char; 64];
        assert!(
            WSAAsyncGetHostByName(
                std::ptr::null_mut(),
                1,
                c"example.com".as_ptr(),
                buf.as_mut_ptr(),
                64
            )
            .is_null()
        );
        assert_eq!(last_error(), WSAHOST_NOT_FOUND);
    }

    #[test]
    fn reverse_lookups_stay_numeric_or_fail() {
        up();
        let away = v4(AWAY, 80);
        let mut host_text = [0 as c_char; 64];
        assert_eq!(
            getnameinfo(
                away.as_ptr(),
                16,
                host_text.as_mut_ptr(),
                64,
                std::ptr::null_mut(),
                0,
                0
            ),
            0,
            "{}",
            last_error()
        );
        assert_eq!(cstr(&host_text), "192.0.2.1");

        // A name is required: nobody answers for it.
        assert_eq!(
            getnameinfo(
                away.as_ptr(),
                16,
                host_text.as_mut_ptr(),
                64,
                std::ptr::null_mut(),
                0,
                NI_NAMEREQD
            ),
            WSAHOST_NOT_FOUND
        );
        // Unless the numeric form is asked for by name or by default.
        assert_eq!(
            getnameinfo(
                away.as_ptr(),
                16,
                host_text.as_mut_ptr(),
                64,
                std::ptr::null_mut(),
                0,
                NI_NAMEREQD | NI_NUMERICHOST
            ),
            0,
            "{}",
            last_error()
        );
        assert_eq!(cstr(&host_text), "192.0.2.1");

        // The service half comes from a local table, so it answers alone.
        let mut service = [0 as c_char; 16];
        assert_eq!(
            getnameinfo(
                away.as_ptr(),
                16,
                std::ptr::null_mut(),
                0,
                service.as_mut_ptr(),
                16,
                0
            ),
            0,
            "{}",
            last_error()
        );
        assert!(!cstr(&service).is_empty());

        let away6 = v6(AWAY6, 80);
        let mut text6 = [0 as c_char; 64];
        assert_eq!(
            getnameinfo(
                away6.as_ptr(),
                28,
                text6.as_mut_ptr(),
                64,
                std::ptr::null_mut(),
                0,
                0
            ),
            0,
            "{}",
            last_error()
        );
        assert!(cstr(&text6).contains("2001:db8"));

        // The wide variant reads the same address the same way.
        let mut wide_text = [0u16; 64];
        assert_eq!(
            GetNameInfoW(
                away.as_ptr(),
                16,
                wide_text.as_mut_ptr(),
                64,
                std::ptr::null_mut(),
                0,
                0
            ),
            0,
            "{}",
            last_error()
        );
        let end = wide_text.iter().position(|&c| c == 0).unwrap();
        assert_eq!(String::from_utf16(&wide_text[..end]).unwrap(), "192.0.2.1");

        assert!(gethostbyaddr(AWAY.as_ptr().cast(), 4, i32::from(AF_INET)).is_null());
        assert_eq!(last_error(), WSAHOST_NOT_FOUND);
        // Not an address of any family: still nothing to name.
        assert!(gethostbyaddr(AWAY.as_ptr().cast(), 4, 99).is_null());
        assert_eq!(last_error(), WSAHOST_NOT_FOUND);
        let mut buf = [0 as c_char; 64];
        assert!(
            WSAAsyncGetHostByAddr(
                std::ptr::null_mut(),
                1,
                AWAY.as_ptr().cast(),
                4,
                i32::from(AF_INET),
                buf.as_mut_ptr(),
                64
            )
            .is_null()
        );
        assert_eq!(last_error(), WSAHOST_NOT_FOUND);
    }

    #[test]
    fn name_space_queries_are_refused() {
        up();
        let mut handle: Handle = std::ptr::null_mut();
        assert_eq!(
            WSALookupServiceBeginA(std::ptr::null(), 0, &mut handle),
            SOCKET_ERROR
        );
        assert_eq!(last_error(), WSASERVICE_NOT_FOUND);
        assert!(handle.is_null());
        assert_eq!(
            WSALookupServiceBeginW(std::ptr::null(), 0, &mut handle),
            SOCKET_ERROR
        );
        assert_eq!(last_error(), WSASERVICE_NOT_FOUND);
        assert!(handle.is_null());
    }

    #[test]
    fn winhttp_only_reaches_the_machine() {
        up();
        let close_handle =
            host!("winhttp.dll", "WinHttpCloseHandle": extern "system" fn(Handle) -> i32);
        let query = host!("winhttp.dll", "WinHttpQueryOption": extern "system" fn(Handle, u32, *mut c_void, *mut u32) -> i32);

        // A session opened with a named proxy still has none.
        let proxy = utf16("192.0.2.1:8080");
        let session = WinHttpOpen(std::ptr::null(), 3, proxy.as_ptr(), std::ptr::null(), 0);
        assert!(!session.is_null(), "{}", last_error_win32());
        let mut info = [0u8; 24];
        let mut len = info.len() as u32;
        assert_eq!(
            query(
                session,
                WINHTTP_OPTION_PROXY,
                info.as_mut_ptr().cast(),
                &mut len
            ),
            1
        );
        assert_eq!(
            u32::from_le_bytes([info[0], info[1], info[2], info[3]]),
            WINHTTP_ACCESS_TYPE_NO_PROXY
        );

        // A proxy set later is accepted and not used.
        #[repr(C)]
        struct ProxyInfo {
            access: u32,
            proxy: *const u16,
            bypass: *const u16,
        }
        let named = ProxyInfo {
            access: 3,
            proxy: proxy.as_ptr(),
            bypass: std::ptr::null(),
        };
        assert_eq!(
            WinHttpSetOption(
                session,
                WINHTTP_OPTION_PROXY,
                (&raw const named).cast(),
                size_of::<ProxyInfo>() as u32
            ),
            1
        );
        let mut len = info.len() as u32;
        assert_eq!(
            query(
                session,
                WINHTTP_OPTION_PROXY,
                info.as_mut_ptr().cast(),
                &mut len
            ),
            1
        );
        assert_eq!(
            u32::from_le_bytes([info[0], info[1], info[2], info[3]]),
            WINHTTP_ACCESS_TYPE_NO_PROXY
        );

        // A handle nobody opened is the host's to refuse.
        assert_eq!(
            WinHttpSetOption(
                std::ptr::null_mut(),
                WINHTTP_OPTION_PROXY,
                std::ptr::null(),
                0
            ),
            0
        );

        // Loopback, by name or literal, gets a handle; nothing else does.
        for server in ["localhost", "127.0.0.1"] {
            let name = utf16(server);
            let connection = WinHttpConnect(session, name.as_ptr(), 80, 0);
            assert!(!connection.is_null(), "{server}: {}", last_error_win32());
            let _ = close_handle(connection);
        }
        for server in ["example.com", "192.0.2.1", "[::2]"] {
            let name = utf16(server);
            assert!(
                WinHttpConnect(session, name.as_ptr(), 80, 0).is_null(),
                "{server}"
            );
            assert_eq!(last_error_win32(), ERROR_WINHTTP_CANNOT_CONNECT, "{server}");
        }
        let _ = close_handle(session);
    }

    #[test]
    fn proxy_auto_detection_fails_like_an_unplugged_machine() {
        up();
        let mut url: *mut u16 = std::ptr::null_mut();
        assert_eq!(WinHttpDetectAutoProxyConfigUrl(1, &mut url), 0);
        assert_eq!(last_error_win32(), ERROR_WINHTTP_AUTODETECTION_FAILED);
        assert!(url.is_null());

        let page = utf16("http://example.com/");
        let mut info = [0u8; 24];
        assert_eq!(
            WinHttpGetProxyForUrl(
                std::ptr::null_mut(),
                page.as_ptr(),
                std::ptr::null(),
                info.as_mut_ptr().cast()
            ),
            0
        );
        assert_eq!(last_error_win32(), ERROR_WINHTTP_AUTODETECTION_FAILED);
        assert_eq!(
            WinHttpGetProxyForUrlEx(std::ptr::null_mut(), page.as_ptr(), std::ptr::null(), 0),
            ERROR_WINHTTP_AUTODETECTION_FAILED
        );
        assert_eq!(
            WinHttpGetProxyForUrlEx2(
                std::ptr::null_mut(),
                page.as_ptr(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                0
            ),
            ERROR_WINHTTP_AUTODETECTION_FAILED
        );
        assert_eq!(
            WinHttpGetProxySettingsEx(std::ptr::null_mut(), 0, std::ptr::null(), 0),
            ERROR_WINHTTP_AUTODETECTION_FAILED
        );
    }

    #[test]
    fn every_override_names_a_real_export() {
        // The ordinals we override name the functions they stand for.
        let net = module();
        for entry in net.exports.iter().filter(|e| e.name.starts_with('#')) {
            let named = net
                .exports
                .iter()
                .find(|n| n.address == entry.address && !n.name.starts_with('#'))
                .unwrap();
            let by_ordinal = ImportSymbol::Ordinal(entry.name[1..].parse().unwrap());
            assert_eq!(
                crate::HostPolicy::default().resolve("ws2_32.dll", &by_ordinal),
                crate::host::function("ws2_32.dll", named.name),
                "{} is not {}",
                entry.name,
                named.name
            );
        }
        // Every override names a real export of its DLL. Some exports only
        // exist on newer Windows (WinHttpGetProxySettingsEx is absent from
        // Windows 10's winhttp.dll); the HLE supplies those itself, so skip
        // them only when the host really lacks them.
        const NEWER_WINDOWS_ONLY: &[&str] = &["WinHttpGetProxySettingsEx"];
        for entry in net.exports.iter().filter(|e| !e.name.starts_with('#')) {
            let dll = if entry.name.starts_with("WinHttp") {
                "winhttp.dll"
            } else {
                "ws2_32.dll"
            };
            let present = crate::host::function(dll, entry.name).is_some();
            if !present && NEWER_WINDOWS_ONLY.contains(&entry.name) {
                continue;
            }
            assert!(present, "{dll} lacks {}", entry.name);
        }
    }
}
