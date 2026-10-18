// FlyEye: stream the Geometry Dash window, and nothing else, with ScreenCaptureKit.
//
// Built into its own app (~/Applications/FlyEye.app) by brain/flyapps.py and launched through macOS's `open`, so the
// Screen Recording permission belongs to this app alone, not to the terminal or IDE that runs the Python side.
// The window is picked by its title; the filter is "this one window" (desktopIndependentWindow), so other windows
// never reach us, even where they overlap it. If the window isn't found we exit instead of falling back to the screen.
//
// stderr (a log file): one status line, "streaming window <id> pid <game pid> self <own pid> ...", or the error.
// stdout (a named pipe), per frame: a 32-byte header, then height * width * 4 bytes of BGRA pixels (no row padding).
//   header: "GDFR" | u32 frame number | f64 capture time (s) | f64 send time (s) | u32 width | u32 height
// Both times are on the host clock (mach_absolute_time), the clock Python's time.monotonic() uses on macOS.
// When the reader closes the pipe, the next write fails and FlyEye exits.
//
//   FlyEye [--title "Geometry Dash"] [--scale 0.5] [--fps 60]

import AppKit
import CoreMedia
import CoreVideo
import Foundation
import ScreenCaptureKit

var title = "Geometry Dash"
var scale = 0.5
var fps = 60.0
var args = CommandLine.arguments.dropFirst().makeIterator()
while let a = args.next() {
    switch a {
    case "--title": title = args.next() ?? title
    case "--scale": scale = Double(args.next() ?? "") ?? scale
    case "--fps": fps = Double(args.next() ?? "") ?? fps
    default: FileHandle.standardError.write("unknown argument \(a)\n".data(using: .utf8)!); exit(2)
    }
}

func log(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

let hostClock = CMClockGetHostTimeClock()
func now() -> Double { CMClockGetTime(hostClock).seconds }

final class Sink: NSObject, SCStreamOutput, SCStreamDelegate {
    var count: UInt32 = 0
    let out = FileHandle.standardOutput

    func stream(_ stream: SCStream, didOutputSampleBuffer sb: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen, sb.isValid, let px = sb.imageBuffer else { return }
        // skip "nothing changed" and other incomplete frames
        if let info = CMSampleBufferGetSampleAttachmentsArray(sb, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]],
           let raw = info.first?[.status] as? Int, SCFrameStatus(rawValue: raw) != .complete { return }
        CVPixelBufferLockBaseAddress(px, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(px, .readOnly) }
        let w = CVPixelBufferGetWidth(px), h = CVPixelBufferGetHeight(px), bpr = CVPixelBufferGetBytesPerRow(px)
        guard let base = CVPixelBufferGetBaseAddress(px) else { return }
        var frame = Data(capacity: 32 + w * h * 4)
        frame.append("GDFR".data(using: .ascii)!)
        withUnsafeBytes(of: count.littleEndian) { frame.append(contentsOf: $0) }
        withUnsafeBytes(of: sb.presentationTimeStamp.seconds) { frame.append(contentsOf: $0) }
        withUnsafeBytes(of: now()) { frame.append(contentsOf: $0) }
        withUnsafeBytes(of: UInt32(w).littleEndian) { frame.append(contentsOf: $0) }
        withUnsafeBytes(of: UInt32(h).littleEndian) { frame.append(contentsOf: $0) }
        for y in 0..<h { frame.append(base.advanced(by: y * bpr).assumingMemoryBound(to: UInt8.self), count: w * 4) }
        count += 1
        do { try out.write(contentsOf: frame) } catch { exit(0) }   // reader went away
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        log("capture stopped: \(error.localizedDescription)")
        exit(1)
    }
}

_ = NSApplication.shared   // connects this tool to the window server, which ScreenCaptureKit needs

let sink = Sink()
var stream: SCStream?

Task {
    do {
        // every window, on every desktop (Space): metadata only, no pixels
        let content = try await SCShareableContent.excludingDesktopWindows(true, onScreenWindowsOnly: false)
        let matches = content.windows.filter { $0.title == title && $0.frame.width > 100 }
        guard let win = matches.max(by: { $0.frame.width * $0.frame.height < $1.frame.width * $1.frame.height }) else {
            log("no window titled \"\(title)\": is the game running? (open geometrydash.app)"); exit(1)
        }
        let filter = SCContentFilter(desktopIndependentWindow: win)
        let cfg = SCStreamConfiguration()
        cfg.width = Int((win.frame.width * scale).rounded())
        cfg.height = Int((win.frame.height * scale).rounded())
        cfg.minimumFrameInterval = CMTime(seconds: 1.0 / fps, preferredTimescale: 600)
        cfg.pixelFormat = kCVPixelFormatType_32BGRA
        cfg.queueDepth = 3
        cfg.showsCursor = false
        let s = SCStream(filter: filter, configuration: cfg, delegate: sink)
        try s.addStreamOutput(sink, type: .screen, sampleHandlerQueue: DispatchQueue(label: "flyeye"))
        try await s.startCapture()
        stream = s
        let pid = win.owningApplication?.processID ?? -1
        log("streaming window \(win.windowID) pid \(pid) self \(getpid()) \"\(title)\" of "
            + "\(win.owningApplication?.applicationName ?? "?"): \(Int(win.frame.width))x\(Int(win.frame.height)) points "
            + "-> \(cfg.width)x\(cfg.height) px, up to \(fps) fps")
    } catch {
        log("could not start capture: \(error.localizedDescription) "
            + "(System Settings > Privacy & Security > Screen & System Audio Recording: allow FlyEye, then retry)")
        exit(1)
    }
}

signal(SIGPIPE, SIG_IGN)
dispatchMain()
