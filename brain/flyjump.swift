// FlyJump: press and release the space key in Geometry Dash, and nothing else.
//
// Built into its own app (~/Applications/FlyJump.app) by brain/flyapps.py and launched through macOS's `open`, so the
// Accessibility permission (needed to send key events) belongs to this app alone, not to the terminal or IDE.
// What it can do is fixed here: one key (space), sent only to one process (CGEventPostToPid), and only if that
// process owns a window of the Wine program running the game.
//
// stdin (a named pipe): one byte per command, "d" = space down, "u" = space up. At the end of input, space is
// released and FlyJump exits.
// stderr (a log file): one status line, "ready pid <game pid> self <own pid>", or the error.
//
//   FlyJump --pid <game pid>          (the pid comes from FlyEye's status line)

import ApplicationServices
import CoreGraphics
import Foundation

func log(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

let SPACE: CGKeyCode = 49
var pid: pid_t = -1
var args = CommandLine.arguments.dropFirst().makeIterator()
while let a = args.next() {
    if a == "--pid", let v = args.next(), let p = pid_t(v) { pid = p } else { log("usage: FlyJump --pid <game pid>"); exit(2) }
}
guard pid > 0 else { log("usage: FlyJump --pid <game pid>"); exit(2) }

// Ask for Accessibility if we don't have it (macOS shows its dialog and adds FlyJump to the list).
let prompt = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
guard AXIsProcessTrustedWithOptions(prompt) else {
    log("no Accessibility permission (System Settings > Privacy & Security > Accessibility: allow FlyJump, then retry)")
    exit(1)
}

// The target must own a window of the Wine program (owner names are visible without any permission).
let windows = (CGWindowListCopyWindowInfo([.optionAll], kCGNullWindowID) as? [[String: Any]]) ?? []
guard windows.contains(where: { ($0[kCGWindowOwnerPID as String] as? pid_t) == pid
                                 && ($0[kCGWindowOwnerName as String] as? String) == "wine" }) else {
    log("pid \(pid) owns no window of \"wine\": not the game, refusing to send keys"); exit(1)
}

var down = false
func press(_ d: Bool) {
    guard d != down, let ev = CGEvent(keyboardEventSource: nil, virtualKey: SPACE, keyDown: d) else { return }
    ev.postToPid(pid)
    down = d
}

log("ready pid \(pid) self \(getpid())")
let input = FileHandle.standardInput
while true {
    let data = input.availableData          // blocks until there is input; empty = the writer closed the pipe
    if data.isEmpty { break }
    for b in data {
        if b == UInt8(ascii: "d") { press(true) } else if b == UInt8(ascii: "u") { press(false) }
    }
}
press(false)
exit(0)
