/* Bunnys Audio Separator — Premiere Pro (ExtendScript) side.
 * ES3: no JSON object guaranteed, so we serialise by hand. */

var BAS = BAS || {};

BAS._str = function (s) {
    s = String(s);
    var out = '"';
    for (var i = 0; i < s.length; i++) {
        var c = s.charAt(i), code = s.charCodeAt(i);
        if (c === '"') out += '\\"';
        else if (c === '\\') out += '\\\\';
        else if (c === '\n') out += '\\n';
        else if (c === '\r') out += '\\r';
        else if (c === '\t') out += '\\t';
        else if (code < 32 || code > 126) out += '\\u' + ('0000' + code.toString(16)).slice(-4);
        else out += c;
    }
    return out + '"';
};

BAS._json = function (v) {
    if (v === null || v === undefined) return 'null';
    var t = typeof v;
    if (t === 'number') return isFinite(v) ? String(v) : 'null';
    if (t === 'boolean') return v ? 'true' : 'false';
    if (t === 'string') return BAS._str(v);
    var parts = [], k;
    if (v instanceof Array) {
        for (k = 0; k < v.length; k++) parts.push(BAS._json(v[k]));
        return '[' + parts.join(',') + ']';
    }
    for (k in v) {
        if (v.hasOwnProperty(k)) parts.push(BAS._str(k) + ':' + BAS._json(v[k]));
    }
    return '{' + parts.join(',') + '}';
};

BAS._ok = function (o) { o = o || {}; o.ok = true; return BAS._json(o); };
BAS._err = function (msg) { return BAS._json({ ok: false, error: String(msg) }); };

BAS._parse = function (s) {
    // input is produced by our own panel via JSON.stringify
    return eval('(' + s + ')');
};

/** Selected clip in the active sequence (audio preferred). */
BAS.getSelection = function () {
    try {
        if (!app.project) return BAS._err('No project is open.');
        var seq = app.project.activeSequence;
        if (!seq) return BAS._err('Open a sequence and select an audio clip first.');
        var sel = seq.getSelection();
        var audio = null, any = null;
        for (var i = 0; i < sel.length; i++) {
            var it = sel[i];
            if (!it || !it.projectItem) continue;
            if (!any) any = it;
            if (!audio && it.mediaType === 'Audio') audio = it;
        }
        var clip = audio || any;
        if (!clip) return BAS._err('Select an audio (or video with audio) clip in the timeline first.');
        var media = clip.projectItem.getMediaPath();
        if (!media) return BAS._err('This clip has no media file (nested sequences are not supported yet).');

        var trackIndex = -1;
        if (clip.mediaType === 'Audio') {
            for (var t = 0; t < seq.audioTracks.numTracks && trackIndex < 0; t++) {
                var tr = seq.audioTracks[t];
                for (var c = 0; c < tr.clips.numItems; c++) {
                    if (BAS._sameClip(tr.clips[c], clip.nodeId, clip.start.seconds, media)) { trackIndex = t; break; }
                }
            }
        }
        // speed = source duration / timeline duration (robust across versions)
        var tlDur = clip.end.seconds - clip.start.seconds;
        var speed = tlDur > 0 ? (clip.outPoint.seconds - clip.inPoint.seconds) / tlDur : 1;

        return BAS._ok({
            mediaPath: media,
            name: clip.name,
            inPoint: clip.inPoint.seconds,
            outPoint: clip.outPoint.seconds,
            start: clip.start.seconds,
            end: clip.end.seconds,
            speed: speed,
            trackIndex: trackIndex,
            nodeId: clip.nodeId || '',
            projectPath: app.project.path || '',
            sequence: seq.name
        });
    } catch (e) {
        return BAS._err('Could not read selection: ' + e.toString());
    }
};

/** Same timeline clip? Uses nodeId when available, else position + media path. */
BAS._sameClip = function (a, id, start, media) {
    if (id) return a.nodeId === id;
    try {
        return Math.abs(a.start.seconds - start) < 0.0005 &&
            BAS._norm(a.projectItem.getMediaPath()) === BAS._norm(media);
    } catch (e) { return false; }
};

BAS._findBin = function (name) {
    var root = app.project.rootItem;
    for (var i = 0; i < root.children.numItems; i++) {
        var ch = root.children[i];
        if (ch.type === ProjectItemType.BIN && ch.name === name) return ch;
    }
    return root.createBin(name);
};

BAS._norm = function (p) {
    return String(p).replace(/\\/g, '/').toLowerCase();
};

BAS._trackFree = function (track, start, end) {
    try { if (track.isLocked && track.isLocked()) return false; } catch (e) {}
    for (var i = 0; i < track.clips.numItems; i++) {
        var c = track.clips[i];
        if (c.start.seconds < end - 0.0005 && c.end.seconds > start + 0.0005) return false;
    }
    return true;
};

BAS._freeTracks = function (seq, from, start, end, need) {
    var free = [];
    for (var t = Math.max(0, from); t < seq.audioTracks.numTracks && free.length < need; t++) {
        if (BAS._trackFree(seq.audioTracks[t], start, end)) free.push(t);
    }
    return free;
};

BAS._addAudioTracks = function (count) {
    var seq = app.project.activeSequence;
    app.enableQE();
    var qs = qe.project.getActiveSequence();
    var before = seq.audioTracks.numTracks;
    try {
        // (videoCount, videoIndex, audioCount, audioType[1 = stereo], audioIndex, submixCount, submixType)
        qs.addTracks(0, 0, count, 1, before, 0, 0);
    } catch (e) {
        qs.addTracks(0, 0, count);
    }
    return app.project.activeSequence.audioTracks.numTracks - before;
};

/** Import stems into a bin and lay them on free audio tracks at the clip's start time. */
BAS.placeStems = function (argJson) {
    var a, warning = '';
    try { a = BAS._parse(argJson); } catch (e0) { return BAS._err('Bad arguments'); }
    try {
        var seq = app.project.activeSequence;
        if (!seq) return BAS._err('No active sequence.');
        var bin = BAS._findBin('Bunnys Stems');
        var paths = [];
        for (var i = 0; i < a.files.length; i++) paths.push(a.files[i].path);
        if (!app.project.importFiles(paths, true, bin, false)) {
            // some versions return false/undefined even on success; we verify below
        }

        var items = [];
        for (i = 0; i < a.files.length; i++) {
            var want = BAS._norm(a.files[i].path), found = null;
            for (var k = bin.children.numItems - 1; k >= 0 && !found; k--) {
                var ch = bin.children[k];
                try { if (BAS._norm(ch.getMediaPath()) === want) found = ch; } catch (e2) {}
            }
            if (!found) return BAS._err('Premiere could not import ' + a.files[i].path);
            items.push(found);
        }

        var start = a.start, end = a.start + a.duration;
        var need = items.length;
        var from = a.trackIndex >= 0 ? a.trackIndex + 1 : 0;
        var free = BAS._freeTracks(seq, from, start, end, need);
        if (free.length < need) {
            try {
                BAS._addAudioTracks(need - free.length);
                seq = app.project.activeSequence;
                free = BAS._freeTracks(seq, from, start, end, need);
            } catch (e3) {}
        }
        if (free.length < need) {
            free = BAS._freeTracks(seq, 0, start, end, need);
        }
        if (free.length < need) {
            return BAS._err('Not enough empty audio tracks. Add ' + (need - free.length) +
                ' empty audio track(s) and try again.');
        }

        var placed = [];
        for (i = 0; i < items.length; i++) {
            var track = seq.audioTracks[free[i]];
            try {
                track.overwriteClip(items[i], start);
            } catch (e5) {
                var tm = new Time();
                tm.seconds = start;
                track.overwriteClip(items[i], tm);
            }
            placed.push(free[i] + 1);
        }

        if (a.muteOriginal) {
            var muted = false;
            for (var t = 0; t < seq.audioTracks.numTracks && !muted; t++) {
                var tr = seq.audioTracks[t];
                for (var c = 0; c < tr.clips.numItems; c++) {
                    if (BAS._sameClip(tr.clips[c], a.nodeId, a.start, a.mediaPath)) {
                        try { tr.clips[c].disabled = true; muted = true; } catch (e4) {}
                        break;
                    }
                }
            }
            if (!muted) warning = 'Could not mute the original clip automatically.';
        }
        return BAS._ok({ tracks: placed, warning: warning });
    } catch (e) {
        return BAS._err(e.toString());
    }
};
