/* Bunnys Audio Separator — panel logic */
(function () {
  "use strict";

  var path = Host.nodeRequire("path");
  var fs = Host.nodeRequire("fs");
  var os = Host.nodeRequire("os");
  var childProcess = Host.nodeRequire("child_process");

  var $ = function (sel, root) { return (root || document).querySelector(sel); };
  var $$ = function (sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); };

  var STORE_KEY = "bunnys-audio-separator-settings";
  var state = { mode: "music", device: "auto", quality: "balanced", running: null };

  var STAGE_TEXT = {
    download: "Downloading AI model (first time only)…",
    decode: "Reading audio…",
    separate: "Separating",
    write: "Saving stems…"
  };

  // ---------------------------------------------------------------- settings
  function loadSettings() {
    try {
      var s = JSON.parse(localStorage.getItem(STORE_KEY) || "{}");
      if (s.mode) state.mode = s.mode;
      if (s.device) state.device = s.device;
      if (s.quality) state.quality = s.quality;
      if (s.stems) {
        Object.keys(s.stems).forEach(function (mode) {
          $$('[data-stems="' + mode + '"] input').forEach(function (cb) {
            cb.checked = s.stems[mode].indexOf(cb.value) >= 0;
          });
        });
      }
      $("#mute-original").checked = !!s.muteOriginal;
    } catch (e) { /* ignore broken storage */ }
  }

  function saveSettings() {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify({
        mode: state.mode, device: state.device, quality: state.quality,
        muteOriginal: $("#mute-original").checked,
        stems: { music: selectedStems("music"), cinematic: selectedStems("cinematic") }
      }));
    } catch (e) { /* ignore */ }
  }

  function selectedStems(mode) {
    return $$('[data-stems="' + mode + '"] input:checked').map(function (cb) { return cb.value; });
  }

  // ---------------------------------------------------------------- UI wiring
  function applyMode() {
    $$(".mode").forEach(function (b) {
      var on = b.getAttribute("data-mode") === state.mode;
      b.classList.toggle("active", on);
      b.setAttribute("aria-selected", on ? "true" : "false");
    });
    $("#stems-music").classList.toggle("hidden", state.mode !== "music");
    $("#stems-cinematic").classList.toggle("hidden", state.mode !== "cinematic");
  }

  function applySeg() {
    $$(".seg").forEach(function (seg) {
      var key = seg.getAttribute("data-setting");
      $$("button", seg).forEach(function (b) {
        b.classList.toggle("active", b.getAttribute("data-value") === state[key]);
      });
    });
  }

  function wire() {
    $$(".mode").forEach(function (b) {
      b.addEventListener("click", function () {
        state.mode = b.getAttribute("data-mode");
        applyMode(); saveSettings();
      });
    });
    $$(".seg button").forEach(function (b) {
      b.addEventListener("click", function () {
        state[b.parentNode.getAttribute("data-setting")] = b.getAttribute("data-value");
        applySeg(); saveSettings();
      });
    });
    $$("[data-preset-cinematic]").forEach(function (b) {
      b.addEventListener("click", function () {
        var want = b.getAttribute("data-preset-cinematic").split(",");
        $$('[data-stems="cinematic"] input').forEach(function (cb) { cb.checked = want.indexOf(cb.value) >= 0; });
        saveSettings();
      });
    });
    $$("[data-preset]").forEach(function (b) {
      b.addEventListener("click", function () {
        var want = b.getAttribute("data-preset").split(",");
        $$('[data-stems="music"] input').forEach(function (cb) { cb.checked = want.indexOf(cb.value) >= 0; });
        saveSettings();
      });
    });
    $$(".stem input, #mute-original").forEach(function (cb) { cb.addEventListener("change", saveSettings); });
    $$("[data-url]").forEach(function (a) {
      a.addEventListener("click", function (e) { e.preventDefault(); Host.openURL(a.getAttribute("data-url")); });
    });
    $("#go").addEventListener("click", run);
    $("#cancel").addEventListener("click", cancel);
  }

  // ---------------------------------------------------------------- engine
  function enginePath() {
    var exe = Host.isWindows ? "bunnys-engine.exe" : "bunnys-engine";
    return path.join(Host.extensionPath(), "bin", "bunnys-engine", exe);
  }

  /** Spawn the engine; onEvent gets every JSON line. Resolves with the final event. */
  function runEngine(args, onEvent, background) {
    return new Promise(function (resolve, reject) {
      var exe = enginePath();
      if (!fs.existsSync(exe)) {
        reject(new Error("Engine not found. Please reinstall Bunnys Audio Separator.\n(" + exe + ")"));
        return;
      }
      var proc = childProcess.spawn(exe, args, { windowsHide: true });
      var buf = "";
      var stderr = "";
      var last = null;
      if (!background) state.running = proc;
      proc.stdout.setEncoding("utf8");
      proc.stdout.on("data", function (chunk) {
        buf += chunk;
        var lines = buf.split("\n");
        buf = lines.pop();
        lines.forEach(function (line) {
          line = line.trim();
          if (!line) return;
          try {
            var ev = JSON.parse(line);
            last = ev;
            onEvent && onEvent(ev);
          } catch (e) { stderr += line + "\n"; }
        });
      });
      proc.stderr.setEncoding("utf8");
      proc.stderr.on("data", function (d) { stderr = (stderr + d).slice(-4000); });
      proc.on("error", function (err) { if (!background) state.running = null; reject(err); });
      proc.on("close", function (code) {
        var killed = proc.killedByUser;
        if (!background) state.running = null;
        if (killed) return reject(new Error("Cancelled"));
        if (last && last.type === "done") return resolve(last);
        if (last && last.type === "info") return resolve(last);
        if (last && last.type === "error") return reject(new Error(last.message));
        reject(new Error("Engine stopped unexpectedly (code " + code + ")\n" + stderr.slice(-600)));
      });
    });
  }

  function detectDevice() {
    var line = $("#device-line");
    if (!Host.available) { line.textContent = "Open this panel inside Premiere Pro."; return; }
    runEngine(["info"], null, true).then(function (info) {
      state.gpu = !!info.gpu;
      line.innerHTML = info.gpu
        ? 'Best device: <span class="ok">' + escapeHtml(info.best) + "</span> — GPU acceleration ready"
        : '<span class="warn">No supported GPU found</span> — CPU mode (slower)';
    }).catch(function (err) {
      line.innerHTML = '<span class="warn">' + escapeHtml(err.message) + "</span>";
    });
  }

  // ---------------------------------------------------------------- run
  function setBusy(busy) {
    $("#go").disabled = busy;
    $("#go-label").textContent = busy ? "Working…" : "Separate selected clip";
    $("#progress").classList.toggle("hidden", !busy);
  }

  function setProgress(value, text) {
    $("#bar-fill").style.width = Math.round(value * 100) + "%";
    if (text) $("#progress-text").textContent = text;
  }

  function showResult(kind, html) {
    var el = $("#result");
    el.className = "result " + kind;
    el.innerHTML = html;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function stamp() {
    var d = new Date();
    var p = function (n) { return (n < 10 ? "0" : "") + n; };
    return d.getFullYear() + p(d.getMonth() + 1) + p(d.getDate()) + "-" + p(d.getHours()) + p(d.getMinutes()) + p(d.getSeconds());
  }

  function mkdirp(dir) {
    if (fs.existsSync(dir)) return;
    mkdirp(path.dirname(dir));
    fs.mkdirSync(dir);
  }

  /** Find a WAV (audio-only) export preset shipped with Premiere Pro. */
  function findWavPreset(appPath) {
    if (!appPath) return null;
    var roots = [appPath, path.join(appPath, "Contents"), path.dirname(appPath)];
    var budget = 30000;
    var found = [];
    function walk(dir, depth) {
      if (depth > 8 || budget <= 0) return;
      var entries;
      try { entries = fs.readdirSync(dir); } catch (e) { return; }
      for (var i = 0; i < entries.length && budget > 0; i++) {
        budget--;
        var full = path.join(dir, entries[i]);
        var st;
        try { st = fs.statSync(full); } catch (e2) { continue; }
        if (st.isDirectory()) walk(full, depth + 1);
        else if (/\.epr$/i.test(entries[i]) && /(wav|waveform)/i.test(entries[i])) found.push(full);
      }
    }
    for (var r = 0; r < roots.length && !found.length; r++) {
      var mediaIO = path.join(roots[r], "MediaIO", "systempresets");
      if (fs.existsSync(mediaIO)) walk(mediaIO, 0);
    }
    for (r = 0; r < roots.length && !found.length; r++) walk(roots[r], 0);
    found.sort(function (a, b) {
      var score = function (p) { return (/48/.test(p) ? 0 : 1) + (/16/.test(p) ? 0 : 1); };
      return score(a) - score(b);
    });
    return found[0] || null;
  }

  function outputDir(sel) {
    var base;
    if (sel.projectPath) base = path.join(path.dirname(sel.projectPath), "Bunnys Stems");
    else base = path.join(os.homedir(), "Documents", "Bunnys Stems");
    var clip = (sel.name || "clip").replace(/[\\/:*?"<>|]+/g, "_").slice(0, 60);
    return path.join(base, clip + " " + stamp());
  }

  function run() {
    if (state.running) return;
    var stems = selectedStems(state.mode);
    if (!stems.length) {
      showResult("err", "Please choose at least one stem.");
      return;
    }
    $("#result").className = "result hidden";
    setBusy(true);
    setProgress(0, "Reading your selection…");

    var t0 = Date.now();
    var sel;
    Host.call("getSelection").then(function (res) {
      if (!res.ok) throw new Error(res.error);
      sel = res;
      sel.outDir = outputDir(sel);
      if (!sel.isSequence) return sel.mediaPath;
      // nested sequence: let Premiere render its mixed audio first
      setProgress(0.02, "Rendering nested sequence audio…");
      var preset = findWavPreset(sel.appPath);
      if (!preset) {
        throw new Error("Could not find Premiere's WAV export preset needed for nested sequences. " +
          "Use Render and Replace on the nested clip, then try again.");
      }
      mkdirp(sel.outDir);
      var outPath = path.join(sel.outDir, "nested-sequence-audio.wav");
      return Host.call("renderNested", { itemNodeId: sel.itemNodeId, outPath: outPath, presetPath: preset })
        .then(function (r) {
          if (!r.ok) throw new Error(r.error);
          return r.path;
        });
    }).then(function (inputPath) {
      var timelineDur = Math.max(0.001, sel.end - sel.start);
      var args = [
        "separate",
        "--input", inputPath,
        "--start", String(sel.inPoint),
        "--end", String(sel.outPoint),
        "--mode", state.mode,
        "--stems", stems.join(","),
        "--device", state.device,
        "--quality", state.quality,
        "--out-dir", sel.outDir,
        "--name", sel.name || "clip",
        "--speed", String(sel.speed > 0 ? sel.speed : 1),
        "--timeline-duration", String(timelineDur)
      ];
      if (sel.reversed) args.push("--reverse");
      return runEngine(args, function (ev) {
        if (ev.type === "progress") {
          var label = STAGE_TEXT[ev.stage] || "Working…";
          if (ev.stage === "separate") label += " " + Math.round(ev.value * 100) + "%";
          setProgress(ev.value, label);
        } else if (ev.type === "device") {
          var note = (state.mode === "cinematic" && ev.provider === "CPUExecutionProvider")
            ? " — CPU mode, this can take a few minutes" : "";
          $("#progress-text").textContent = "Running on " + ev.label + note;
        }
      });
    }).then(function (done) {
      setProgress(0.99, "Adding stems to your timeline…");
      return Host.call("placeStems", {
        files: done.files,
        start: sel.start,
        duration: done.audio_seconds,
        trackIndex: sel.trackIndex,
        nodeId: sel.nodeId,
        mediaPath: sel.mediaPath,
        muteOriginal: $("#mute-original").checked
      }).then(function (placed) { return { done: done, placed: placed }; });
    }).then(function (r) {
      setProgress(1, "Done");
      var secs = ((Date.now() - t0) / 1000).toFixed(1);
      var items = r.done.files.map(function (f) { return "<li>" + escapeHtml(f.label) + "</li>"; }).join("");
      var msg = "<strong>Done in " + secs + "s</strong> on " + escapeHtml(r.done.device) + ".<ul>" + items + "</ul>";
      if (r.placed.ok) {
        msg += "Placed on new audio tracks, in sync with your clip.";
        if (r.placed.warning) msg += "<br>" + escapeHtml(r.placed.warning);
        showResult("ok", msg);
      } else {
        msg += "The stems could not be placed on the timeline automatically: " + escapeHtml(r.placed.error)
          + "<br>They are saved in:<br><code>" + escapeHtml(sel.outDir) + "</code><br>"
          + "Import them and place them at the clip's start time.";
        showResult("err", msg);
      }
    }).catch(function (err) {
      var m = err && err.message ? err.message : String(err);
      showResult(m === "Cancelled" ? "" : "err", m === "Cancelled" ? "Cancelled." : escapeHtml(m).replace(/\n/g, "<br>"));
    }).then(function () {
      setBusy(false);
    });
  }

  function cancel() {
    if (state.running) {
      state.running.killedByUser = true;
      state.running.kill();
    }
  }

  // ---------------------------------------------------------------- init
  loadSettings();
  applyMode();
  applySeg();
  wire();
  if (path && fs && childProcess) detectDevice();
  else $("#device-line").textContent = "Open this panel inside Premiere Pro.";
})();
