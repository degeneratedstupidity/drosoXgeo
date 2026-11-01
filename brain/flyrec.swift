// FlyRec: record the Geometry Dash window, and nothing else, to a video file at full (Retina) resolution.
//
// For the post: FlyEye streams the window at ~440 px wide (all the eye needs, and fast); that's too blurry to show.
// FlyRec is a separate app with its own Screen Recording permission, run alongside a live play: ScreenCaptureKit
// with the same "this one window" filter as FlyEye (desktopIndependentWindow: other windows never reach it, even
// where they overlap; if the window isn't found it exits instead of falling back to the screen), written by the
// hardware encoder straight to an MP4 (SCRecordingOutput). It never touches FlyEye's stream or the model's timing.
//
// stderr (a log file): "recording window <id> ... start <host time>" when it starts (host clock = Python's
// time.monotonic() on macOS, so the video can be lined up with brain.live's log), "finished <path>" at the end.
//
// Runs a real AppKit main loop (not dispatchMain): ScreenCaptureKit puts its "recording" indicator in the menu bar
// while a recording output is attached, and that has to be created on the main thread.
//
//   FlyRec --out <path.mp4> [--seconds 100] [--title "Geometry Dash"] [--scale 1] [--fps 60]

import AppKit
import AVFoundation
import CoreMedia
import Foundation
import ScreenCaptureKit

var title = "Geometry Dash"
var scale = 1.0                                  // window points -> pixels (2 = full Retina: ~4x the file size)
var fps = 60.0
var seconds = 100.0
var outPath = ""
var args = CommandLine.arguments.dropFirst().makeIterator()
while let a = args.next() {
    switch a {
    case "--title": title = args.next() ?? title
    case "--scale": scale = Double(args.next() ?? "") ?? scale
    case "--fps": fps = Double(args.next() ?? "") ?? fps
    case "--seconds": seconds = Double(args.next() ?? "") ?? seconds
    case "--out": outPath = args.next() ?? ""
    default: FileHandle.standardError.write("unknown argument \(a)\n".data(using: .utf8)!); exit(2)
    }
}

func log(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

let hostClock = CMClockGetHostTimeClock()
func now() -> Double { CMClockGetTime(hostClock).seconds }

if outPath.isEmpty { log("--out <path.mp4> is required"); exit(2) }

final class Rec: NSObject, SCStreamDelegate, SCRecordingOutputDelegate {
    func stream(_ stream: SCStream, didStopWithError error: Error) {
        log("capture stopped: \(error.localizedDescription)")
        exit(1)
    }
    func recordingOutputDidStartRecording(_ recordingOutput: SCRecordingOutput) {
        log("recording started \(now())")
    }
    func recordingOutput(_ recordingOutput: SCRecordingOutput, didFailWithError error: Error) {
        log("recording failed: \(error.localizedDescription)")
        exit(1)
    }
    func recordingOutputDidFinishRecording(_ recordingOutput: SCRecordingOutput) {
        log("finished \(outPath)")
        exit(0)
    }
}

let app = NSApplication.shared   // connects this tool to the window server, which ScreenCaptureKit needs
app.setActivationPolicy(.accessory)

let rec = Rec()
var stream: SCStream?

Task {
    do {
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
        cfg.showsCursor = false
        cfg.capturesAudio = false
        let s = SCStream(filter: filter, configuration: cfg, delegate: rec)
        let rc = SCRecordingOutputConfiguration()
        rc.outputURL = URL(fileURLWithPath: outPath)
        rc.outputFileType = .mp4
        rc.videoCodecType = .h264
        try s.addRecordingOutput(SCRecordingOutput(configuration: rc, delegate: rec))
        try await s.startCapture()
        stream = s
        log("recording window \(win.windowID) \"\(title)\" of \(win.owningApplication?.applicationName ?? "?"): "
            + "\(Int(win.frame.width))x\(Int(win.frame.height)) points -> \(cfg.width)x\(cfg.height) px, "
            + "\(fps) fps, \(seconds) s, to \(outPath); start \(now())")
        try await Task.sleep(nanoseconds: UInt64(seconds * 1e9))
        try await s.stopCapture()                // the recording output then finishes the file and we exit
        try await Task.sleep(nanoseconds: 15_000_000_000)
        log("stopped (no finish callback)")
        exit(0)
    } catch {
        log("could not start recording: \(error.localizedDescription) "
            + "(System Settings > Privacy & Security > Screen & System Audio Recording: allow FlyRec, then retry)")
        exit(1)
    }
}

app.run()
