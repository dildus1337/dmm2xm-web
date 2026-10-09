"""
dmm2xm_convert.py

High level conversion driver functions built on top of dmm2xm_core, mirroring
the algorithm in dmm2xm.dpr's main program block (RecordPatternsFromDMMPattern,
PatternToDMMNotes, LoadInstrumentFromDMI, etc.)
"""

import copy
import math
from operator import mul as _mul
import os
import struct
import types

try:
    import numpy as _np
except ImportError:  # pure-Python fallback below stays fully functional
    _np = None

from dmm2xm_core import (
    DMM_8_CHANNELS, MAX_SAMPLES, NOTE_STOP,
    EFFECT_VOLUME, EFFECT_PATTERN_BREAK, EFFECT_SPEED_TEMPO, EFFECT_EXTRA_FINE_PORTA,
    EFFECT_EXTENDED_EFFECTS, EXT_EFFECT_NOTE_DELAY, EXT_EFFECT_NOTE_CUT, EXT_EFFECT_PATTERN_DELAY,
    DMM_REF_TEMPO, ITVolumeEnvelope,
    PAN_SETTING, VERSION_TEXT,
    WadFile, list_dmm_in_wad, parse_dmm_bytes, parse_dmi_bytes, apply_sample_fixes,
    is_finish_note, DMMNote,
    PatternNote, make_note, Pattern, Sample, Instrument, XMModule,
    sample_rate_to_xm, xm_to_sample_rate,
    xm_module_save, xm_module_load, mod_module_load, load_module,
    trunc_div, trunc_mod, get_extension, cut_extension, ensure_dir_for_file,
    clamp,
)


class ConversionError(Exception):
    pass


def _default_log(msg):
    print(msg, end="")


# ---------------------------------------------------------------------------
# Forward conversion: DMM (standalone or from WAD) -> XM
# ---------------------------------------------------------------------------

def _find_dmi_path(input_dir, home_dir, name):
    candidates = [
        os.path.join(input_dir, name),
        os.path.join(home_dir, name),
        os.path.join(input_dir, "dmi", name),
        os.path.join(home_dir, "dmi", name),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _load_instrument_from_dmi(ctx, name, instrument_index, instruments_total, wad):
    """Returns (Sample|None, ok:bool). ok=False means 'not found' (warning)."""
    if name == "":
        return None, True

    sample = Sample()
    sample.bits = 8
    sample.channels = 1
    sample.name = name
    if not ctx.no_stereo and instruments_total <= MAX_SAMPLES:
        sample.panning = PAN_SETTING[instruments_total - 1][instrument_index]

    raw = None
    clamp_size = None
    if wad is not None:
        idx = wad.get_file_index(name)
        if idx != -1:
            offset, size, _n = wad.records[idx]
            raw = wad.read_at(offset, size)
            clamp_size = size

    if raw is None:
        path = _find_dmi_path(ctx.input_dir, ctx.home_dir, name)
        if path is None:
            return None, False
        with open(path, "rb") as f:
            raw = f.read()
        clamp_size = len(raw)

    length, samplerate, loopstart_raw, looplength_raw, samples = parse_dmi_bytes(raw, clamp_size)
    sample.sample_length = length
    sample.relative_note, sample.finetune = sample_rate_to_xm(samplerate, ctx.emulate_bug)
    if looplength_raw > 0:
        sample.loop = True
        sample.loop_start = loopstart_raw
        sample.loop_length = looplength_raw
    apply_sample_fixes(name, loopstart_raw, looplength_raw, sample)
    sample.data = [v * 256 for v in samples]
    return sample, True


def _record_patterns_from_dmm_pattern(ctx, xm, dmm_pattern, pattern_number, start_pattern_number):
    max_note_row = 0
    for c in range(DMM_8_CHANNELS):
        xm.patterns[start_pattern_number].set_note(0, c, make_note(NOTE_STOP))

    notes_recorded = 0
    delay = 0
    row = 0
    channel = 0
    current_chan_delay = 0
    max_chan_delay = 0
    current_chan_notes = 0
    max_chan_notes = 0
    n_total = len(dmm_pattern)

    while True:
        dmm_note = dmm_pattern[notes_recorded]

        if delay < ctx.quantization:
            notes_recorded += 1

            if dmm_note.volume == 128:
                channel += 1
                max_chan_delay = max(max_chan_delay, current_chan_delay)
                max_chan_notes = max(max_chan_notes, current_chan_notes)
                delay = 0
                current_chan_delay = 0
                current_chan_notes = 0
                row = 0
                if notes_recorded == n_total:
                    break
                continue

            if dmm_note.note == 254:
                inc = (dmm_note.instrument * 256) + dmm_note.delay
                delay += inc
                current_chan_delay += inc
                if notes_recorded == n_total:
                    break
                continue

            # volume 1 would land on 0x10 = "volume 0", which the backward pass reads as a stop; a DMM
            # volume of 1 is a very quiet but *sounding* note (СОЛО fades in from 1 with 0xFF events)
            result_volume = 0 if dmm_note.volume > 126 else max(0x11, dmm_note.volume // 2 + 16)

            if dmm_note.volume > 0 and dmm_note.note == 255:
                # 0xFF is not a note: the engine only updates the channel volume and keeps the
                # sound playing (SOUND.ASM playnext). It used to be converted into a bogus note
                # (255 + 25 wrapped to XM note 24). XM has the same thing: a volume-column-only cell.
                if dmm_note.volume < 128:          # 0x81..0xFF = "keep volume": nothing to write
                    vcol = 0x50 if dmm_note.volume >= 127 else max(0x11, dmm_note.volume // 2 + 16)
                    r = row - 1 if delay < 0 else row
                    r = max(r, 0)
                    tgt_pattern = xm.patterns[start_pattern_number + r // ctx.max_rows]
                    old_note = tgt_pattern.get_note(r % ctx.max_rows, channel)
                    if old_note.note != 0 and old_note.note != NOTE_STOP:
                        old_note.volume = vcol
                        tgt_pattern.set_note(r % ctx.max_rows, channel, old_note)
                    else:
                        tgt_pattern.set_note(r % ctx.max_rows, channel, make_note(0, 0, vcol))
                    current_chan_notes += 1
            elif dmm_note.volume > 0:
                push_up = 0
                my_note = None
                if delay > 0:
                    put_delay = delay
                    if put_delay > 15:
                        put_delay = 15
                        ctx.misaligned_notes.append(
                            (delay, put_delay, pattern_number, row, channel + 1))
                    my_note = make_note(dmm_note.note + 25, dmm_note.instrument, result_volume,
                                        EFFECT_EXTENDED_EFFECTS, put_delay | (EXT_EFFECT_NOTE_DELAY << 4))
                elif delay < 0:
                    put_delay = ctx.quantization + delay
                    if put_delay <= 0:
                        ctx.log("\nWarning: Wrong delay, choose smaller quantization!\n")
                    else:
                        if put_delay > 15:
                            put_delay = 15
                            ctx.misaligned_notes.append(
                                (delay, put_delay, pattern_number, row, channel + 1))
                        my_note = make_note(dmm_note.note + 25, dmm_note.instrument, result_volume,
                                            EFFECT_EXTENDED_EFFECTS, put_delay | (EXT_EFFECT_NOTE_DELAY << 4))
                        push_up = 1
                else:
                    my_note = make_note(dmm_note.note + 25, dmm_note.instrument, result_volume)

                if my_note is not None:
                    r = row - push_up
                    tgt_pattern = xm.patterns[start_pattern_number + r // ctx.max_rows]
                    old_note = tgt_pattern.get_note(r % ctx.max_rows, channel)
                    if old_note.note != 0 and old_note.note != NOTE_STOP:
                        ctx.log(
                            f"\nWarning: Note {old_note.note} overwritten by {my_note.note} at "
                            f"pattern {start_pattern_number + r // ctx.max_rows} column "
                            f"{channel + 1} row {r}, lost data! Set smaller quantization.\n")
                    tgt_pattern.set_note(r % ctx.max_rows, channel, my_note)
                    current_chan_notes += 1
            else:
                xm.patterns[start_pattern_number + row // ctx.max_rows].set_note(
                    row % ctx.max_rows, channel, make_note(NOTE_STOP))
                current_chan_notes += 1

            if ctx.used_channels < channel + 1:
                ctx.used_channels = channel + 1

            if dmm_note.delay == 0:
                delay += 256
                current_chan_delay += 256
            else:
                delay += dmm_note.delay
                current_chan_delay += dmm_note.delay
            if ctx.min_note_delay > delay:
                ctx.min_note_delay = delay

        row += 1
        max_note_row = max(max_note_row, row)
        delay -= ctx.quantization
        if notes_recorded == n_total:
            break

    row += trunc_div(delay, ctx.quantization)
    put_delay = trunc_mod(max_chan_delay, ctx.quantization)
    if put_delay != 0:
        ctx.log(f"\nWarning: {put_delay} ticks lost at the pattern end, trying to fix - "
                f"not all players support this!\n")
        fixed = False
        for c in range(ctx.used_channels + 1):
            tgt_pattern = xm.patterns[start_pattern_number + max_note_row // ctx.max_rows]
            old_note = tgt_pattern.get_note((max_note_row - 1) % ctx.max_rows, c)
            if old_note.effect == 0:
                old_note.effect = EFFECT_EXTRA_FINE_PORTA
                old_note.effect_parameter = 0x60 | put_delay
                tgt_pattern.set_note((max_note_row - 1) % ctx.max_rows, c, old_note)
                put_delay = 0
                fixed = True
                break
        if not fixed:
            ctx.log("\nWarning: Effects are all occupied, couldn't fix!\n")

    max_note_row = max(max_note_row, row)
    if max_note_row % ctx.max_rows != 0:
        xm.patterns[start_pattern_number + max_note_row // ctx.max_rows].resize(
            max_note_row % ctx.max_rows, DMM_8_CHANNELS)
    result = trunc_div(max_note_row - 1, ctx.max_rows) + 1
    ctx.log(f"...ok, ticks: {max_chan_delay}, notes: {max_chan_notes}, rows: {max_note_row}")
    return result


class ForwardOptions:
    def __init__(self):
        self.input_file = ""
        self.dmm_in_wad_name = ""
        self.output_file = ""
        self.quantization = 4
        self.max_rows = 256
        self.no_stereo = False
        self.emulate_bug = False
        self.preloaded_wad = None  # optional WadFile, to avoid re-reading/re-parsing
                                    # the same .wad from disk for every song in a batch


def convert_forward(opts: ForwardOptions, log=_default_log):
    """DMM/WAD -> XM. Returns output path on success, raises ConversionError on failure."""
    if not (1 <= opts.quantization <= 31):
        raise ConversionError("Invalid quantization value, should be 1 <= Q <= 31.")
    if not (1 <= opts.max_rows <= 1024):
        raise ConversionError("Invalid max rows value, should be 1 <= R <= 1024.")
    if opts.max_rows > 256:
        log("Warning: Rows value > 256, non-standard, some players WILL glitch!\n")

    if not os.path.isfile(opts.input_file):
        raise ConversionError(f'File "{opts.input_file}" not found!')

    input_dir = os.path.dirname(os.path.abspath(opts.input_file)) or "."
    ctx = types.SimpleNamespace(
        quantization=opts.quantization,
        max_rows=opts.max_rows,
        used_channels=2,
        min_note_delay=256,
        misaligned_notes=[],
        no_stereo=opts.no_stereo,
        emulate_bug=opts.emulate_bug,
        input_dir=input_dir,
        home_dir=os.path.dirname(os.path.abspath(__file__)),
        log=log,
    )

    ext = get_extension(opts.input_file)
    loading_from_wad = ext == "wad"
    wad = None

    if loading_from_wad:
        if not opts.dmm_in_wad_name:
            names = list_dmm_in_wad(opts.input_file)
            if names is None:
                raise ConversionError(f'Invalid WAD file: "{opts.input_file}".')
            raise ConversionError(
                "No song specified. Valid DMM names found in WAD:\n" + "\n".join(names))
        output_file = opts.output_file or (opts.dmm_in_wad_name + ".xm")
        wad = opts.preloaded_wad if opts.preloaded_wad is not None else WadFile(opts.input_file)
        if not wad.valid:
            raise ConversionError(f'Invalid WAD file: "{opts.input_file}".')
        idx = wad.get_file_index(opts.dmm_in_wad_name)
        if idx == -1:
            raise ConversionError(f'File "{opts.dmm_in_wad_name}" not found in WAD!')
        offset, size, _n = wad.records[idx]
        raw = wad.read_at(offset, size)
        log(f':: Reading DMM from WAD: "{opts.dmm_in_wad_name}"...\n')
        dmm = parse_dmm_bytes(raw)
        if dmm is None:
            raise ConversionError("Unknown DMM format!")
    else:
        output_file = opts.output_file or (cut_extension(opts.input_file) + ".xm")
        with open(opts.input_file, "rb") as f:
            raw = f.read()
        log(f':: Reading DMM: "{opts.input_file}"...\n')
        dmm = parse_dmm_bytes(raw)
        if dmm is None:
            raise ConversionError("Unknown DMM format!")

    xm = XMModule()
    xm.name = opts.dmm_in_wad_name if loading_from_wad else \
        cut_extension(os.path.basename(opts.input_file))
    xm.linear_slides = True
    xm.start_tempo = DMM_REF_TEMPO
    xm.start_speed = ctx.quantization
    xm.comment = f"Converted with {VERSION_TEXT}"

    log(":: Reading instruments...\n")
    for counter in range(dmm["instrument_number"]):
        name = dmm["instruments"][counter]
        sample, ok = _load_instrument_from_dmi(ctx, name, counter, dmm["instrument_number"], wad)
        inst = Instrument()
        if sample is not None:
            inst.sample[0] = sample
        xm.instruments[counter + 1] = inst
        if not ok:
            log(f'Warning: Couldn\'t find instrument "{name}".\n')
    if not ctx.no_stereo and dmm["instrument_number"] > MAX_SAMPLES:
        log("Warning: Found more than 11 instruments, stereo is not available.\n")

    log(f":: Recording notes with quantization = {ctx.quantization}...\n")
    dmm_notes = dmm["notes"]
    dmm_patterns = []
    notes_read = 0
    for _p in range(dmm["pattern_number"]):
        channels_finished = 0
        current = []
        while True:
            if notes_read == len(dmm_notes):
                log("Warning: Unexpected end of notes, possibly broken data.\n")
                break
            current.append(dmm_notes[notes_read])
            if is_finish_note(dmm_notes[notes_read]):
                channels_finished += 1
            notes_read += 1
            if channels_finished == DMM_8_CHANNELS:
                break
        dmm_patterns.append(current)
        if notes_read == len(dmm_notes):
            break

    for i in range(256):
        xm.patterns[i].resize(ctx.max_rows, DMM_8_CHANNELS)

    start_pattern_number = 0
    dmm2xm_pattern = [[0, 0] for _ in range(256)]
    for counter in range(len(dmm_patterns)):
        log(f"DMM pattern {counter}...")
        counter2 = _record_patterns_from_dmm_pattern(
            ctx, xm, dmm_patterns[counter], counter, start_pattern_number)
        if counter2 == 1:
            log(f" -> XM pattern {start_pattern_number}\n")
        else:
            log(f" -> XM patterns {start_pattern_number}..{start_pattern_number + counter2 - 1}\n")
        dmm2xm_pattern[counter] = [start_pattern_number, counter2]
        start_pattern_number += counter2

    for (d, nd, pat, r, ch) in ctx.misaligned_notes:
        log(f"Misaligned note at pattern {pat}, channel {ch}, row {r} (Delay = {d} -> {nd})\n")
    if ctx.misaligned_notes:
        log(f"Warning: Quantization may be wrong or too large, misaligned "
            f"{len(ctx.misaligned_notes)} notes.\n")
    log(f"Minimum note delay was {ctx.min_note_delay}.\n")

    xm.track_length = 0
    counter3 = 0
    order_start = []      # XM order index at which each DMM order entry begins
    dmm_pattern_order = dmm["pattern_order"]
    dmm_pattern_order_size = dmm["pattern_order_size"]
    if dmm_pattern_order_size == 0:
        log("Warning: Pattern order missing, rebuilding as sequential.\n")
        dmm_pattern_order = list(range(dmm["pattern_number"]))
        dmm_pattern_order_size = dmm["pattern_number"]

    for counter in range(dmm_pattern_order_size):
        if dmm_pattern_order[counter] == 255:
            if counter == dmm_pattern_order_size - 1:
                log("Warning: Unexpected end of pattern order, possibly broken data.\n")
                break
            # The value after the 255 marker is an index into the order list (the engine does
            # curseq = seq[i+1]), not a pattern number; map it to the XM order index where that
            # DMM order entry starts (a DMM pattern may have been split into several XM patterns).
            restart_order = dmm_pattern_order[counter + 1]
            xm.restart_position = order_start[restart_order] if restart_order < len(order_start) else 0
            log(f"Restart position: {restart_order} -> {xm.restart_position}\n")
            break
        else:
            order_start.append(counter3)
            pat_shift, pat_length = dmm2xm_pattern[dmm_pattern_order[counter]]
            xm.track_length += pat_length
            for counter2 in range(pat_length):
                xm.pattern_order[counter3] = pat_shift + counter2
                counter3 += 1
    log(f"Track length: {xm.track_length}\n")

    log(":: Optimizing...\n")
    for counter in range(start_pattern_number, 256):
        xm.patterns[counter].resize(0, 0)
    if ctx.used_channels < DMM_8_CHANNELS:
        for counter in range(start_pattern_number):
            if xm.patterns[counter].rows > 0:
                xm.patterns[counter].resize(xm.patterns[counter].rows, ctx.used_channels)
        log(f"Only {ctx.used_channels} of 8 channels used. Removed unused channels.\n")
    xm.number_of_channels = ctx.used_channels

    log(f':: Saving to "{output_file}"...')
    xm_module_save(xm, output_file)
    log(" OK.\n\n")
    return output_file


# ---------------------------------------------------------------------------
# Backward conversion: XM -> DMM (+ DMI instrument files)
# ---------------------------------------------------------------------------

def _is_stop_note_for_dmm(note):
    return (note.effect == EFFECT_VOLUME and note.effect_parameter == 0) or note.volume == 16 \
        or note.note == NOTE_STOP


def _envelope_value_at(points, tick):
    """Linear-interpolated envelope value at `tick`, given `points` (a list
    of (tick, value) pairs sorted ascending by tick, points[0][0] == 0).
    Holds flat at the first/last point's value outside the defined range,
    matching how IT itself plays an envelope back."""
    if tick <= points[0][0]:
        return points[0][1]
    if tick >= points[-1][0]:
        return points[-1][1]
    for i in range(len(points) - 1):
        t0, v0 = points[i]
        t1, v1 = points[i + 1]
        if t0 <= tick <= t1:
            if t1 == t0:
                return v1
            return v0 + (v1 - v0) * (tick - t0) / (t1 - t0)
    return points[-1][1]  # unreachable given the guards above; kept defensive


def _envelope_segment_area(points, t_start, t_end, power=False):
    """Integral of the piecewise-linear envelope over [t_start, t_end).

    With `power=False` (the default): plain area-under-curve in
    value*ticks units, i.e. what a linear time-weighted average needs.

    With `power=True`: integral of value(t)^2 dt instead - what an RMS
    (power-weighted) average needs. For a straight segment from (t0,v0) to
    (t1,v1), the closed form is (t1-t0)*(v0^2 + v0*v1 + v1^2)/3 (the
    standard result for integrating a linear ramp squared), so this still
    only needs the segment endpoints, no extra sampling.

    RMS matters here specifically because a plain linear average badly
    understates how loud a fast attack-then-decay envelope actually sounds
    while it's playing: perceived loudness tracks *power*, not raw
    amplitude, and a short loud attack contributes much less to a linear
    mean than it does to how loud a listener actually judges the note to
    be. A percussive envelope that spends most of a note's held duration
    near-silent but attacks sharply at the very start should end up
    sounding closer to "quiet but with punch" than "uniformly quiet" -
    RMS gets meaningfully closer to that than a linear mean does, without
    swinging to the opposite extreme of just sampling the envelope's value
    at the single instant the note happens to end (which, for anything
    that decays, is close to the *quietest* point almost by construction -
    that was tried and sounds clearly worse in practice, see the
    accompanying discussion)."""
    if t_end <= t_start:
        return 0.0
    boundaries = sorted(t for t, _ in points if t_start < t < t_end)
    cursor = t_start
    area = 0.0
    for t in boundaries + [t_end]:
        v0 = _envelope_value_at(points, cursor)
        v1 = _envelope_value_at(points, t)
        if power:
            area += (t - cursor) * (v0 * v0 + v0 * v1 + v1 * v1) / 3.0
        else:
            area += (v0 + v1) / 2.0 * (t - cursor)
        cursor = t
    return area


def _envelope_average_over_hold(env, hold_ticks, age=0):
    """RMS (power-weighted) average volume-envelope value (0..64) over the
    first `hold_ticks` ticks that a note on `env` (an ITVolumeEnvelope) is
    actually held, ASSUMING it is never explicitly released within that
    span - i.e. it's either cut short by the next trigger or just keeps
    playing, but never receives a real Note Off. That matches how the
    large majority of notes actually behave in practice (the default New
    Note Action is "Cut", which does not run any release phase at all, and
    explicit Note Off events are comparatively rare), and it's also the
    only assumption that makes sense here regardless: DMM's own note model
    has no release phase to represent in the first place, so there would
    be nothing useful to do with a "how loud is it while releasing"
    calculation even if we computed one.

    Priority matches the real IT engine: a sustain loop, if enabled, gates
    envelope progress while a note is considered "on" - under the
    no-release assumption above it never advances past it, so it is
    checked first. A plain loop (unconditional on note-on/off) is checked
    next. With neither, the envelope plays through once and then holds
    flat at its last point's value for as long as the note keeps sounding.

    `age` is how many ticks old the note already is when this hold starts -
    i.e. a pattern-boundary carry (see `holdover` in _pattern_to_dmm_notes):
    the DMM retrigger must sound like the original's *tail* ([age,
    age+hold]), not replay its attack ([0, hold]) at full volume. Fresh
    notes always pass 0, which keeps the historical bit-identical path.
    """
    if not env.active or not env.points or hold_ticks <= 0:
        return 64.0
    if age > 0:
        # Tail of an already-decayed note: RMS of the gated envelope over
        # [age, age+hold]. Sampled through _envelope_instant_value (which
        # already implements the same sustain/loop gating and loop phase as
        # the events/bake paths), so all three volume modes agree with each
        # other. Midpoint rule with 4 sub-steps per tick keeps the error
        # against the analytic integral inaudible.
        steps = max(1, int(hold_ticks) * 4)
        total = 0.0
        for i in range(steps):
            v = _envelope_instant_value(env, age + (i + 0.5) * hold_ticks / steps)
            total += v * v
        return (total / steps) ** 0.5
    points = env.points

    if env.sustain:
        lo, hi = env.sustain_start, env.sustain_end
    elif env.loop:
        lo, hi = env.loop_start, env.loop_end
    else:
        lo, hi = None, None

    if lo is None or not (0 <= lo < len(points)) or not (0 <= hi < len(points)):
        last_tick = points[-1][0]
        if hold_ticks <= last_tick:
            power_area = _envelope_segment_area(points, 0, hold_ticks, power=True)
        else:
            power_area = _envelope_segment_area(points, 0, last_tick, power=True)
            power_area += (points[-1][1] ** 2) * (hold_ticks - last_tick)
        return (power_area / hold_ticks) ** 0.5

    loop_tick_lo = points[lo][0]
    loop_tick_hi = points[hi][0]
    if hold_ticks <= loop_tick_lo:
        # Cut short before ever reaching the loop/sustain segment - only
        # the attack portion leading up to it actually played.
        power_area = _envelope_segment_area(points, 0, hold_ticks, power=True)
        return (power_area / hold_ticks) ** 0.5

    power_area = _envelope_segment_area(points, 0, loop_tick_lo, power=True)
    remaining = hold_ticks - loop_tick_lo
    loop_len = loop_tick_hi - loop_tick_lo
    if loop_len <= 0:
        # Degenerate (single-point) loop/sustain - flat value from here on,
        # e.g. a sustain loop pinned to the attack's own peak node.
        power_area += (points[lo][1] ** 2) * remaining
    else:
        loop_power_area = _envelope_segment_area(points, loop_tick_lo, loop_tick_hi, power=True)
        whole, rem = divmod(remaining, loop_len)
        power_area += loop_power_area * whole
        power_area += _envelope_segment_area(points, loop_tick_lo, loop_tick_lo + rem, power=True)
    return (power_area / hold_ticks) ** 0.5


def _note_volume_with_envelope(note, hold_ticks, it_volume_envelopes, it_instrument, age=0):
    """Like _get_note_volume_for_dmm(note), but first scales the explicit
    volume-column level down by how much a real IT volume envelope (on
    `it_instrument`, if it has one and it_volume_envelopes knows about it)
    would actually have decayed it over the note's real `hold_ticks`
    duration - instead of using the value the note was triggered at for
    its entire, possibly much longer, audible duration. `age` offsets the
    envelope clock for pattern-boundary carries (see
    _envelope_average_over_hold)."""
    env = it_volume_envelopes.get(it_instrument) if it_instrument else None
    if env is None or not env.active or not (0x10 <= note.volume <= 0x50):
        return _get_note_volume_for_dmm(note)
    level = note.volume - 0x10
    avg = _envelope_average_over_hold(env, hold_ticks, age)
    scaled_level = clamp(round(level * avg / 64.0), 0, 64)
    if scaled_level == level:
        return _get_note_volume_for_dmm(note)
    scaled = note.copy()
    scaled.volume = 0x10 + scaled_level
    return _get_note_volume_for_dmm(scaled)


# Exact copy of the reference note-frequency table from d2d_dmm_player.py
# (NOTE_FREQ / NOTE_C4), needed so _render_envelope_shaped_sample can work
# out the real playback rate DMM will actually use for a given note - see
# that function's docstring for why this matters.
NOTE_FREQ = [
    16.35, 17.32, 18.35, 19.45, 20.60, 21.83, 23.12, 24.50, 25.96, 27.50, 29.14, 30.87,
    32.70, 34.65, 36.71, 38.89, 41.20, 43.65, 46.25, 49.00, 51.91, 55.00, 58.27, 61.74,
    65.41, 69.30, 73.42, 77.78, 82.41, 87.31, 92.50, 98.00, 103.83, 110.00, 116.54, 123.47,
    130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94,
    261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88,
    523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880.00, 932.33, 987.77,
    1046.50, 1108.73, 1174.66, 1244.51, 1318.51, 1396.91, 1479.98, 1567.98, 1661.22, 1760.00, 1864.66, 1975.53,
    2093.00, 2217.46, 2349.32, 2489.02, 2637.02, 2793.83, 2959.96, 3135.96, 3322.44, 3520.00, 3729.31, 3951.07,
]
NOTE_C4 = NOTE_FREQ[48]


def _envelope_instant_value(env, tick):
    """The envelope's *instantaneous* value (0..64) at an arbitrary tick,
    extended indefinitely past its own last defined point according to the
    same sustain-loop / loop / hold-flat gating (and the same "never
    released" assumption) as _envelope_average_over_hold - but returning
    the value *at* that tick rather than an aggregate over a span. This is
    what's needed to actually shape a PCM waveform frame-by-frame, as
    opposed to picking one representative flat volume for a whole note."""
    points = env.points
    if env.sustain:
        lo, hi = env.sustain_start, env.sustain_end
    elif env.loop:
        lo, hi = env.loop_start, env.loop_end
    else:
        lo, hi = None, None
    if lo is None or not (0 <= lo < len(points)) or not (0 <= hi < len(points)):
        return _envelope_value_at(points, tick)
    loop_tick_lo, loop_tick_hi = points[lo][0], points[hi][0]
    if tick <= loop_tick_lo:
        return _envelope_value_at(points, tick)
    loop_len = loop_tick_hi - loop_tick_lo
    if loop_len <= 0:
        return points[lo][1]
    phase = (tick - loop_tick_lo) % loop_len
    return _envelope_value_at(points, loop_tick_lo + phase)


def _sample_frame_at(data, src_len, loop, pingpong, loop_start, loop_len, i):
    """The waveform value at *virtual* output frame `i`, simulating the
    sample's own loop behaviour (forward, ping-pong/bidi, or none) so it
    can be read for arbitrarily many frames - exactly like the original
    engine would when a note is held far longer than one straight
    play-through of the raw data. Returns None past the natural end of a
    non-looping sample (i.e. silence from there on).

    Bidi (ping-pong) loops in particular are simulated properly here even
    though the .dmi format itself can only store a plain forward loop
    (see _build_dmi_bytes) - it doesn't matter, because the whole point of
    this function is to produce a one-off, already-shaped, non-looping
    buffer: baking a true ping-pong read in *now* is strictly more
    faithful to the original than what a normal (forward-loop-only) DMM
    instrument could ever manage for this sample regardless."""
    if i < src_len and (not loop or i < loop_start + loop_len):
        return data[i] if i < src_len else None
    if not loop or loop_len <= 0:
        return None
    period = loop_len * (2 if pingpong else 1)
    phase = (i - loop_start) % period
    if pingpong and phase >= loop_len:
        phase = period - 1 - phase
    return data[loop_start + phase]


def _shaped_frames_fast(src, src_len, loop, pingpong, loop_start, loop_length,
                        points, gate, tick_per_frame, num_frames, age=0):
    """Vectorized twin of the per-frame loop in _render_envelope_shaped_sample
    (only called when numpy is available, otherwise that loop runs as-is).

    Computes every output frame with the exact same operations in the exact
    same order as the scalar code - one independent element-wise pass, no
    reductions - so the resulting PCM bytes are identical down to the last
    bit (verified by byte-comparing full conversions); it is only faster
    (one C-speed pass over N frames instead of N Python iterations with
    several function calls each).

    `gate` is (lo_tick, hi_tick, lo_value) or None, mirroring
    _envelope_instant_value's sustain/loop gating (None/disabled/degenerate
    means "plain envelope"). `age` offsets the envelope clock for a
    pattern-boundary carry (the tail, not a replayed attack); 0 keeps the
    historical bit-identical path. Returns (frames, silent_from) with the
    same truncation semantics as the scalar loop.
    """
    np = _np
    idx = np.arange(num_frames, dtype=np.float64)

    arr = np.asarray(src, dtype=np.float64)
    if not loop:
        num = num_frames if num_frames < src_len else src_len
        vals = arr[:num]
        idx = idx[:num]
        silent_from = num if num < num_frames else None
    else:
        silent_from = None
        le = loop_start + loop_length
        direct = (idx < src_len) & (idx < le)
        period = loop_length * (2 if pingpong else 1)
        phase = (idx - loop_start) % period
        if pingpong:
            over = phase >= loop_length
            phase = np.where(over, period - 1 - phase, phase)
        widx = (loop_start + phase).astype(np.int64)
        safe = np.where(direct, idx, 0.0).astype(np.int64)
        vals = np.where(direct, arr[safe], arr[widx])

    t = idx * tick_per_frame + age
    # Gating exactly like _envelope_instant_value: ticks at/before the loop
    # start always read the plain envelope there (even a degenerate,
    # zero-length loop still plays its attack); past it a degenerate loop
    # holds the single node's value, a real one wraps around it.
    # (`t` already carries the carry-age offset, so gating/phase stays exact.)
    if gate is None:
        tm, pre, use_const = t, None, False
    else:
        lo_tick, hi_tick, lo_value = gate
        llen = hi_tick - lo_tick
        pre = t <= lo_tick
        if llen <= 0:
            tm, use_const, const = t, True, float(lo_value)
        else:
            tm = np.where(pre, t, lo_tick + (t - lo_tick) % llen)
            use_const, const = False, 0.0

    pt = np.array([p[0] for p in points], dtype=np.float64)
    pv = np.array([p[1] for p in points], dtype=np.float64)
    if len(pt) == 1:
        # A single node is flat everywhere (the scalar guards catch every
        # tick before any segment scan runs); the degenerate-gate constant
        # below still overrides past the loop start, exactly like scalar.
        iv = np.full_like(t, pv[0])
    else:
        # Same segment choice as the scalar scan (first segment with
        # t0 <= t <= t1, hence side='left'), same formula, flat ends -
        # all evaluated at the gated tick tm, like _envelope_value_at's
        # caller does.
        j = np.searchsorted(pt, tm, side="left") - 1
        j = np.clip(j, 0, len(pt) - 2)
        t0, v0 = pt[j], pv[j]
        t1, v1 = pt[j + 1], pv[j + 1]
        # Guarded denominator: np.where still evaluates the division on
        # degenerate (t1 == t0) lanes, which would warn (0/0) without this;
        # those lanes are masked out by the where itself either way.
        span = np.where(t1 == t0, 1.0, t1 - t0)
        mid = np.where(t1 == t0, v1, v0 + (v1 - v0) * (tm - t0) / span)
        iv = np.where(tm <= pt[0], pv[0], np.where(tm >= pt[-1], pv[-1], mid))
    if gate is None or not use_const:
        value = iv
    else:
        value = np.where(pre, iv, const)

    prod = vals * (value / 64.0)
    # tolist() on an int64 array yields plain Python ints, exactly what the
    # scalar loop appended - without a per-element Python-level int() call.
    return np.rint(prod).astype(np.int64).tolist(), silent_from


def _render_envelope_shaped_sample(sample, env, hold_ticks, tempo, played_note, age=0):
    """Returns a new Sample: `sample`'s own waveform, amplitude-multiplied
    by `env`'s real decay/loop shape over exactly `hold_ticks` tracker
    ticks of real time (converted via `tempo` using the standard
    tick-duration-in-ms = 2500/tempo formula), and then stopped - not
    looped, since the whole point is a one-off rendering of "this
    instrument, held for exactly this long", which DMM's own flat,
    unshaped note volume can never reproduce on its own (see the
    discussion this accompanies).

    `age` offsets the envelope clock for a pattern-boundary carry: the
    shape renders the tail ([age, age+hold]), not a replayed attack
    ([0, hold]). Fresh notes always pass 0.

    `played_note` is the *absolute* note this specific occurrence is
    played at (PatternNote.note's own numbering, i.e. what `_pattern_to_
    dmm_notes` subtracts 25 from to get the DMM note byte). This matters
    because DMM/d2d_dmm_player.py doesn't resample a sample's pitch by
    reading its data at a different step for different notes - it just
    plays the *same* stored PCM at a note-dependent output rate (see
    render_dmm_f32's `step = (slot["rate"]/out_rate) * (NOTE_FREQ[fidx]/
    NOTE_C4)`). So a buffer sized as if it always played at the sample's
    own base rate will actually run `NOTE_FREQ[fidx]/NOTE_C4` times faster
    or slower than intended once DMM applies that same note-dependent
    rate to it - for a note pitched up from the sample's reference pitch,
    the buffer finishes (and the channel falls silent) *before* the next
    scheduled note, opening up a gap of silence that has no business being
    there. Deriving the buffer length (and the tick/frame conversion used
    to shape it) from the *actual* played-note rate instead keeps its
    real-world playing time exactly `hold_ticks` long regardless of pitch.

    Returns None if there's nothing sensible to render (no envelope data,
    no source PCM, or a non-positive duration).
    """
    if not env.active or not env.points or hold_ticks <= 0:
        return None
    if sample is None or sample.sample_length <= 0 or not sample.data:
        return None
    base_rate = xm_to_sample_rate(sample.relative_note, sample.finetune)
    if base_rate <= 0:
        return None
    fidx = clamp(played_note - 25 + 24, 0, len(NOTE_FREQ) - 1)
    note_rate = base_rate * (NOTE_FREQ[fidx] / NOTE_C4)
    if note_rate <= 0:
        return None

    duration_s = hold_ticks * 2.5 / tempo
    num_frames = max(1, round(duration_s * note_rate))

    loop = sample.loop and sample.loop_length > 0
    pingpong = loop and sample.pingpong_loop
    src, src_len = sample.data, sample.sample_length
    tick_per_frame = tempo / (2.5 * note_rate)  # ticks advanced per output frame

    if _np is not None:
        # Same gate decision as _envelope_instant_value's preamble.
        points = env.points
        if env.sustain:
            glo, ghi = env.sustain_start, env.sustain_end
        elif env.loop:
            glo, ghi = env.loop_start, env.loop_end
        else:
            glo = ghi = None
        if glo is None or not (0 <= glo < len(points)) or not (0 <= ghi < len(points)):
            gate = None
        else:
            gate = (float(points[glo][0]), float(points[ghi][0]), points[glo][1])
        out, silent_from = _shaped_frames_fast(
            src, src_len, loop, pingpong, sample.loop_start, sample.loop_length,
            points, gate, tick_per_frame, num_frames, age)
    else:
        out = [0] * num_frames
        silent_from = None
        for i in range(num_frames):
            v = _sample_frame_at(src, src_len, loop, pingpong, sample.loop_start, sample.loop_length, i)
            if v is None:
                silent_from = i
                break
            scale = _envelope_instant_value(env, age + i * tick_per_frame) / 64.0
            out[i] = int(round(v * scale))
    if silent_from is not None:
        out = out[:silent_from]
        num_frames = silent_from
    if num_frames <= 0:
        return None

    shaped = Sample()
    shaped.data = out
    shaped.sample_length = num_frames
    shaped.bits = sample.bits
    shaped.channels = 1
    shaped.name = sample.name
    shaped.loop = False
    shaped.pingpong_loop = False
    shaped.loop_start = 0
    shaped.loop_length = 0
    shaped.volume = 64
    shaped.panning = sample.panning
    shaped.relative_note = sample.relative_note
    shaped.finetune = sample.finetune
    return shaped


# Rounding granularity (tracker ticks) used to key the envelope-shaped-
# sample cache below: two notes whose real hold durations are within this
# many ticks of each other reuse the same rendered sample rather than each
# getting their own. This keeps the number of extra instrument slots a
# busy, heavily-enveloped song needs bounded, at the cost of a small,
# inaudible amount of imprecision in exactly where the shape cuts off.
ENVELOPE_BAKE_TICK_GRANULARITY = 2

# Hard ceiling on the *total* instrument count (real + baked) baking is
# allowed to push a conversion to. DMM's own instrument count is a single
# byte (see convert_backward), so 255 is the absolute limit; capping well
# under that leaves headroom for the module's own real instruments and
# stops a pathological input (many distinct notes x durations needing
# their own render) from silently overflowing the format.
DMM_MAX_INSTRUMENTS_FOR_BAKING = 250


def _get_or_render_envelope_instrument(envelope_ctx, mapped_sample, it_instrument, hold_ticks, tempo,
                                        played_note, age=0):
    """Returns a DMM instrument number carrying a one-off, envelope-shaped
    rendering of `mapped_sample`'s waveform for a note of `it_instrument`
    held `hold_ticks` ticks and played at `played_note`, creating and
    caching it on first use via envelope_ctx (see convert_backward), or
    None if baking isn't applicable (no active envelope) or has hit its
    instrument-slot budget - in which case the caller should fall back to
    the flat-volume approximation instead. `age` offsets the envelope
    clock for a pattern-boundary carry (the tail, not a replayed attack)."""
    if envelope_ctx is None or envelope_ctx.get("dry_run") \
            or not it_instrument or not mapped_sample:
        return None
    env = envelope_ctx["envelopes"].get(it_instrument)
    if env is None or not env.active:
        return None
    xm = envelope_ctx["xm"]
    src_inst = xm.instruments.get(mapped_sample)
    src_sample = src_inst.sample.get(0) if src_inst is not None else None
    if src_sample is None or not src_sample.loop:
        # Baking is only actually needed for a *looping* sample: it has no
        # decay of its own, so a flat DMM note volume - however carefully
        # picked - can never turn it into anything but a constant drone for
        # as long as it's held, which is exactly the "holds a key instead
        # of pressing it" problem this exists to fix.
        #
        # A one-shot, non-looping sample is a different situation: it
        # already has its *own* natural attack/decay baked into the raw
        # waveform regardless of what the volume envelope does, so a flat
        # (RMS-weighted) note volume on top of it already reads as a
        # percussive hit. Baking the envelope's shape into it as well would
        # multiply the two decays together, which is more technically
        # "correct" in isolation but in practice tends to snuff the note
        # out far more abruptly than the real engine's separate volume and
        # sample-amplitude paths ever would together - so this is left on
        # the existing flat-volume path deliberately.
        return None
    g = ENVELOPE_BAKE_TICK_GRANULARITY
    rounded_ticks = max(g, ((hold_ticks + g // 2) // g) * g)
    rounded_age = max(0, ((int(age) + g // 2) // g) * g) if age > 0 else 0
    key = (mapped_sample, it_instrument, played_note, rounded_ticks, rounded_age)
    cache = envelope_ctx["cache"]
    if key in cache:
        return cache[key]
    if xm.number_of_instruments() >= DMM_MAX_INSTRUMENTS_FOR_BAKING:
        return None
    shaped = _render_envelope_shaped_sample(src_sample, env, rounded_ticks, tempo, played_note,
                                            rounded_age)
    if shaped is None:
        cache[key] = None
        return None
    new_num = xm.number_of_instruments() + 1
    new_inst = Instrument()
    new_inst.name = (src_inst.name if src_inst is not None else "") or ""
    new_inst.sample[0] = shaped
    xm.instruments[new_num] = new_inst
    envelope_ctx["extra_count"] += 1
    cache[key] = new_num
    return new_num


# Volume automation through "0xFF events". In DMM a note byte of 0xFF means "no new note": the
# channel's volume (and instrument pointer) is updated and the sound keeps playing (SOUND.ASM
# playnext: note==0FFh -> straight to the volume/delay code; the game's own СОЛО song uses it).
# That lets a held note fade, swell or be re-volumed at any tick WITHOUT extra instruments,
# for the price of one 4-byte note per change.
EVENT_MIN_STEP = 4      # skip slide-derived changes smaller than this many DMM volume units


def _event_volume(level):
    """0..64 tracker level -> DMM volume for a mid-note event. Never 0: volume 0 stops the voice
    for good in DMM (irate=0), whereas a tracker note at volume 0 can come back with a later
    volume command."""
    return max(1, min(127, int(round(level)) * 2 + 1))


def _volume_events(traj, total):
    """traj: [(tracker_tick, level, dmm_time, explicit[, "R"])] of a note held for `total` DMM ticks
    ("R" marks a retrigger: the note is struck again at that moment).
    Returns (initial_volume_override_or_None, [(dmm_tick_offset, dmm_volume, is_retrigger), ...])."""
    if not traj or len(traj) < 2 or total <= 1:
        return None, []
    init_v = None
    raw = []
    for p in traj[1:]:
        _tk, lvl, dt, explicit = p[0], p[1], p[2], p[3]
        retrig = len(p) > 4 and p[4] == "R"
        t = int(round(dt))
        v = _event_volume(lvl)
        if t >= total:
            break
        if t <= 0:
            if not retrig:
                init_v = v
            continue
        if raw and raw[-1][0] == t:
            raw[-1] = (t, v, explicit or raw[-1][2], retrig or raw[-1][3])
        else:
            raw.append((t, v, explicit, retrig))
    last_v = init_v if init_v is not None else _event_volume(traj[0][1])
    events = []
    for i, (t, v, explicit, retrig) in enumerate(raw):
        if retrig:                       # a retrigger is always kept, even at an unchanged volume
            events.append((t, v, True))
            last_v = v
        elif v != last_v and (explicit or abs(v - last_v) >= EVENT_MIN_STEP or i == len(raw) - 1):
            events.append((t, v, False))
            last_v = v
    return init_v, events


def _emit_note_events(out, note, inst, vol, total, traj, use_events, first_from_traj=False):
    """Like _emit_long, but a note whose volume changed while it was held is split into the note
    plus 0xFF volume events (see above). `first_from_traj`: the trajectory already contains the
    complete level (e.g. with an IT envelope multiplied in), so it also gives the note's own volume."""
    init_v, events = _volume_events(traj, total) if (use_events and vol > 0) else (None, [])
    if first_from_traj and traj and vol > 0:
        vol = _event_volume(traj[0][1])
    if init_v is not None:
        vol = init_v
    if not events:
        _emit_long(out, note, inst, vol, total)
        return
    t_prev = 0
    seg_note, seg_inst, seg_vol = note, inst, vol
    for t, v, retrig in events:
        _emit_long(out, seg_note, seg_inst, seg_vol, t - t_prev)
        # a retrigger is the same note struck again (note+instrument), a plain change is a 0xFF event
        seg_note, seg_inst, seg_vol = (note, inst, v) if retrig else (255, 0, v)
        t_prev = t
    _emit_long(out, seg_note, seg_inst, seg_vol, total - t_prev)


# Sample offsets (S3M/IT Oxx, MOD/XM 9xx). DMM cannot start a sample in the middle, so a note with an
# offset plays a copy of its instrument whose PCM starts at that offset. One copy per distinct
# (instrument, offset); at most OFFSET_MAX_VARIANTS per song (WAD lump budget) - beyond that the
# offset is ignored and the note plays from the start.
OFFSET_MAX_VARIANTS = 32
_OFFSET_SILENT = -1


def _apply_sample_offset(envelope_ctx, mapped_sample, offset):
    """Instrument number to use for a note of `mapped_sample` that starts `offset` frames in.
    Returns `mapped_sample` when there is nothing to do, _OFFSET_SILENT when the offset lies past
    the end of a non-looping sample (the note makes no sound)."""
    if envelope_ctx is None or envelope_ctx.get("dry_run") \
            or not mapped_sample or offset <= 0:
        return mapped_sample
    xm = envelope_ctx["xm"]
    src_inst = xm.instruments.get(mapped_sample)
    smp = src_inst.sample.get(0) if src_inst is not None else None
    if smp is None:
        return mapped_sample
    n = min(smp.sample_length, len(smp.data))
    if n <= 0:
        return mapped_sample
    looped = bool(smp.loop) and smp.loop_length > 1
    ls = min(smp.loop_start, n - 1)
    le = min(n, smp.loop_start + smp.loop_length) if looped else n
    if looped and le - ls < 2:
        looped = False
        le = n
    if offset >= n:
        if not looped:
            return _OFFSET_SILENT
        offset = ls + (offset - ls) % (le - ls)     # the offset lands somewhere in the repeating part
    key = ("offset", mapped_sample, offset)
    cache = envelope_ctx["cache"]
    if key in cache:
        return cache[key]
    if envelope_ctx.get("offset_count", 0) >= OFFSET_MAX_VARIANTS or \
            xm.number_of_instruments() >= DMM_MAX_INSTRUMENTS_FOR_BAKING:
        envelope_ctx["offset_skipped"] = envelope_ctx.get("offset_skipped", 0) + 1
        cache[key] = mapped_sample
        return mapped_sample
    old = smp.data
    new_smp = copy.copy(smp)
    if looped and ls <= offset < le:
        # starts inside the loop: play offset..loop end once, then the loop proper. The DMI plays its
        # whole data once and only then repeats [loop_start, loop_start+loop_length).
        new_smp.data = list(old[offset:le]) + list(old[ls:le])
        new_smp.loop_start = le - offset
        new_smp.loop_length = le - ls
        new_smp.loop = True
    elif looped:                                      # starts before the loop
        new_smp.data = list(old[offset:])
        new_smp.loop_start = ls - offset
        new_smp.loop_length = le - ls
        new_smp.loop = True
    else:
        new_smp.data = list(old[offset:n])
        new_smp.loop = False
        new_smp.loop_start = new_smp.loop_length = 0
    new_smp.sample_length = len(new_smp.data)
    new_num = xm.number_of_instruments() + 1
    new_inst = Instrument()
    new_inst.name = (src_inst.name if src_inst is not None else "") or ""
    new_inst.sample[0] = new_smp
    xm.instruments[new_num] = new_inst
    envelope_ctx["offset_count"] = envelope_ctx.get("offset_count", 0) + 1
    cache[key] = new_num
    return new_num


def _sample_is_looped(envelope_ctx, mapped_sample):
    """Whether the PCM sample an instrument number maps to has a loop. A one-shot (non-looped)
    sample naturally falls silent once its data runs out, pattern boundary or not - carrying it
    over as a synthetic retrigger (see `holdover`) would make it audibly replay a decayed/finished
    sound that the source never asked to hear again. Only genuinely still-sounding (looped)
    samples need the retrigger."""
    if envelope_ctx is None or not mapped_sample:
        return False
    inst = envelope_ctx["xm"].instruments.get(mapped_sample)
    smp = inst.sample.get(0) if inst is not None else None
    return bool(smp is not None and smp.loop and smp.loop_length > 1)


def _holdover_audible(it_volume_envelopes, it_instrument, age):
    """Whether a note carried over a pattern boundary (`holdover`) is still
    actually making sound `age` tracker ticks after it was triggered.

    A looping sample alone is not enough: a volume envelope with no loop
    or sustain point holds its last value forever, so a note whose envelope
    has decayed to 0 is silent in the tracker even though the voice is
    "still playing" - re-triggering it in DMM would audibly replay its
    whole attack (BGM12.XM pattern 5 channels 4-8: instrument 3's envelope
    ends at tick 180 while the note is 300+ ticks old, so the correct DMM
    is silence, not a fresh attack-decay swell). Sustained/looping
    envelopes never die on their own and always carry over; envelopeless
    notes ring as long as their (looping) sample does.
    """
    env = it_volume_envelopes.get(it_instrument) if it_instrument else None
    if env is None or not env.active or not env.points:
        return True
    pts = env.points
    if env.sustain:
        lo = max(0, min(env.sustain_start, len(pts) - 1))
        hi = max(0, min(env.sustain_end, len(pts) - 1))
        return max(p[1] for p in pts[min(lo, hi):max(lo, hi) + 1]) > 0
    if env.loop:
        lo = max(0, min(env.loop_start, len(pts) - 1))
        hi = max(0, min(env.loop_end, len(pts) - 1))
        return max(p[1] for p in pts[min(lo, hi):max(lo, hi) + 1]) > 0
    return _envelope_value_at(pts, age) > 0


def _flush_note(out, note_obj, hold_ticks, total, eff_it_instr, eff_mapped_sample, tempo,
                it_volume_envelopes, envelope_ctx, traj, volume_slides, sample_offset=0, age=0):
    """Close out a note held `hold_ticks` tracker ticks / `total` DMM ticks and append it to `out`.

    volume_slides == "events": IT volume envelopes and volume slides both become 0xFF volume
    events (no baked instruments at all). "bake": the shape is rendered into an instrument.
    "off": envelopes are baked/averaged as they always were, slides are ignored. `age` is how
    many ticks old the note already is when this hold starts (a pattern-boundary carry): the
    envelope clock starts there, so the carry sounds like the original's tail, not a replayed
    attack. Fresh notes always pass 0."""
    if sample_offset:
        eff_mapped_sample = _apply_sample_offset(envelope_ctx, eff_mapped_sample, sample_offset)
        if eff_mapped_sample == _OFFSET_SILENT:          # offset past the end of a one-shot sample
            _emit_long(out, 24, 0, 0, total)
            return
    env = it_volume_envelopes.get(eff_it_instr) if (eff_it_instr and it_volume_envelopes) else None
    env_events = volume_slides == "events" and env is not None and env.active and \
        bool(eff_mapped_sample) and hold_ticks > 0 and total > 1
    if env_events:
        # flat volume without the envelope (it is applied through the events instead)
        inst_val, vol = _resolve_note_for_dmm(note_obj, hold_ticks, {}, eff_it_instr, eff_mapped_sample,
                                               tempo, None, None)
        base = [(p[0], p[1]) for p in traj] if traj else [(0, _note_start_level(note_obj))]
        ratio = total / float(hold_ticks)
        # An "E" point in traj (see PatternNote.env_restart) restarts the envelope's own clock -
        # an XM cell that names the instrument again without a new note (FT2 quirk: resets volume
        # and the envelope, keeps the sample playing) - common as a rhythmic "gate" on chip-style
        # arpeggios. Without it every such restart is silently skipped and a held note that should
        # pulse several times just plays its very first pulse and then coasts at the envelope's
        # tail value for the rest of the hold.
        restarts = sorted(p[0] for p in traj if len(p) > 4 and p[4] == "E") if traj else []
        bounds = [0] + [t for t in restarts if 0 < t < hold_ticks] + [int(hold_ticks)]
        trj = []
        for segi, (seg_start, seg_end) in enumerate(zip(bounds, bounds[1:])):
            for t in range(seg_start, seg_end):
                # The first segment continues an already-`age`-ticks-old envelope (a carry); later
                # segments were restarted mid-pattern ("E") and start over at 0. Slides (`base`)
                # always start at 0 in the new pattern - only the instrument envelope is old.
                eclock = (t - seg_start) + (age if segi == 0 else 0)
                lvl = _envelope_value_at(base, t) * _envelope_instant_value(env, eclock) / 64.0
                trj.append((t, lvl, t * ratio, False))
        if traj:
            trj += [(p[0], _envelope_value_at(base, p[0]) * _envelope_instant_value(env, age + p[0]) / 64.0,
                     p[2], True, "R") for p in traj if len(p) > 4 and p[4] == "R"]
            trj.sort(key=lambda p: p[2])
        _emit_note_events(out, _dmm_note_number(note_obj.note), inst_val, vol, total, trj, True, True)
        return
    inst_val, vol = _resolve_note_for_dmm(note_obj, hold_ticks, it_volume_envelopes, eff_it_instr,
                                           eff_mapped_sample, tempo, envelope_ctx,
                                           traj if volume_slides == "bake" else None, age)
    _emit_note_events(out, _dmm_note_number(note_obj.note), inst_val, vol, total, traj,
                      volume_slides != "bake")


def _retrig_level(code, level):
    """Volume (0..64) after one retrigger of S3M/IT Qxy / XM Rxy with volume-change code x."""
    if code in (0, 8):
        return level
    if code <= 5:
        return max(0.0, level - (1, 2, 4, 8, 16)[code - 1])
    if code == 6:
        return level * 2.0 / 3.0
    if code == 7:
        return level / 2.0
    if code <= 0xD:
        return min(64.0, level + (1, 2, 4, 8, 16)[code - 9])
    if code == 0xE:
        return min(64.0, level * 1.5)
    return min(64.0, level * 2.0)


def _note_start_level(n):
    """Volume level (0..64) a note is triggered at; 64 when the cell says nothing."""
    if n.effect == EFFECT_VOLUME:
        return float(min(n.effect_parameter, 64))
    if 0x10 <= n.volume <= 0x50:
        return float(n.volume - 0x10)
    return 64.0


# Limits for volume-shape baking. A baked note is a whole extra DMI, and a long one (a pad that
# fades over ten seconds) would exceed the DMI's 65535-sample limit and get decimated into mush
# (and made conversions 8x bigger and slower), so only notes whose *audible* part fits into
# SHAPED_MAX_FRAMES are baked individually, and all baked shapes together are limited to
# SHAPED_TOTAL_FRAMES; anything beyond that falls back to a single RMS volume for the note.
SHAPED_MAX_FRAMES = 24000
SHAPED_TOTAL_FRAMES = 1200000
# A volume change smaller than this fraction of the note's peak level is not worth a custom
# instrument (the flat RMS level is close enough).
SHAPED_MIN_DEPTH = 0.25
SHAPED_MIN_LEVELS = 6.0


def _get_or_render_shaped_instrument(envelope_ctx, mapped_sample, it_instrument, hold_ticks, tempo,
                                      played_note, shape_points, age=0):
    """Like _get_or_render_envelope_instrument, but for an arbitrary volume shape (`shape_points`:
    [(tick, 0..64)], e.g. produced by volume-slide commands) instead of an IT envelope; an active
    IT envelope on a looping sample is multiplied in. DMM notes have one flat volume, so this is
    the only way to reproduce a note that fades (or swells) while it is held. `age` offsets the
    multiplied envelope clock for a pattern-boundary carry (slides themselves always start at 0
    in the new pattern - only the instrument envelope is already `age` ticks old)."""
    if envelope_ctx is None or envelope_ctx.get("dry_run") or not mapped_sample:
        return None
    xm = envelope_ctx["xm"]
    src_inst = xm.instruments.get(mapped_sample)
    src_sample = src_inst.sample.get(0) if src_inst is not None else None
    if src_sample is None or src_sample.sample_length <= 0 or not src_sample.data:
        return None
    g = ENVELOPE_BAKE_TICK_GRANULARITY
    rounded = max(g, ((hold_ticks + g // 2) // g) * g)
    rounded_age = max(0, ((int(age) + g // 2) // g) * g) if age > 0 else 0
    env = envelope_ctx["envelopes"].get(it_instrument) if it_instrument else None
    use_env = env is not None and env.active and src_sample.loop
    vals = []
    for t in range(rounded + 1):
        v = _envelope_value_at(shape_points, t)
        if use_env:
            v = v * _envelope_instant_value(env, rounded_age + t) / 64.0
        vals.append(int(round(v / 4.0)) * 4)          # 1/16 gain steps keep the cache small
    # Everything after the shape has died away is silence: end the sample there instead of
    # rendering (and storing) a long tail of zeros. This also makes equal decays share one DMI.
    last = max((t for t, v in enumerate(vals) if v > 0), default=-1)
    if last < 0:
        return None
    active = min(rounded, last + 1)
    vals = vals[:active + 1]
    key = ("shape", mapped_sample, played_note, active, tuple(vals), it_instrument if use_env else 0,
           rounded_age if use_env else 0)
    cache = envelope_ctx["cache"]
    if key in cache:
        return cache[key]
    if xm.number_of_instruments() >= DMM_MAX_INSTRUMENTS_FOR_BAKING:
        return None
    base_rate = xm_to_sample_rate(src_sample.relative_note, src_sample.finetune)
    fidx = clamp(played_note - 25 + 24, 0, len(NOTE_FREQ) - 1)
    est_frames = active * 2.5 / tempo * base_rate * NOTE_FREQ[fidx] / NOTE_C4
    if est_frames > SHAPED_MAX_FRAMES or \
            envelope_ctx.get("shaped_frames", 0) + est_frames > SHAPED_TOTAL_FRAMES:
        cache[key] = None
        return None
    e = ITVolumeEnvelope()
    e.active = True
    e.points = [(t, float(v)) for t, v in enumerate(vals)]
    shaped = _render_envelope_shaped_sample(src_sample, e, active, tempo, played_note)
    if shaped is not None:
        envelope_ctx["shaped_frames"] = envelope_ctx.get("shaped_frames", 0) + shaped.sample_length
    if shaped is None:
        cache[key] = None
        return None
    new_num = xm.number_of_instruments() + 1
    new_inst = Instrument()
    new_inst.name = (src_inst.name if src_inst is not None else "") or ""
    new_inst.sample[0] = shaped
    xm.instruments[new_num] = new_inst
    envelope_ctx["extra_count"] += 1
    cache[key] = new_num
    return new_num


def _resolve_note_for_dmm(note_obj, hold_ticks, it_volume_envelopes, eff_it_instr, eff_mapped_sample,
                           tempo, envelope_ctx, traj=None, age=0):
    """Returns (instrument_number, dmm_volume) for a note that's about to
    be closed out after being held `hold_ticks` ticks. `eff_it_instr` and
    `eff_mapped_sample` are the *effective* IT-instrument / mapped-sample
    numbers to use - i.e. already resolved through "remembered" channel
    memory by the caller if note_obj itself doesn't carry its own.

    `age` is how many ticks old the note already is when this hold starts
    (a pattern-boundary carry): the envelope clock starts there, not at 0,
    so the carry sounds like the original's tail instead of a replayed
    attack. Fresh notes always pass 0.

    `traj` is the note's volume trajectory while it was held, [(tick, level 0..64), ...], built
    from volume slides / mid-note volume changes. If the volume really moved, the note is
    rendered into a one-off instrument that carries that shape (a hi-hat with A0F must die in
    a couple of ticks, not hiss at full level for the whole row); if no instrument slot is
    left it falls back to the RMS level over the hold.

    Otherwise prefers a pre-rendered, envelope-shaped one-off sample (see
    _get_or_render_envelope_instrument) when available, and falls back to the flat-volume RMS
    approximation (_note_volume_with_envelope)."""
    if traj is not None and len(traj) > 1 and hold_ticks > 0 and eff_mapped_sample:
        traj = [(p[0], p[1]) for p in traj]
        pts = [(t, l) for t, l in traj if t < hold_ticks]
        pts.append((hold_ticks, _envelope_value_at(traj, hold_ticks)))
        peak = max(l for _t, l in pts)
        low = min(l for _t, l in pts)
        if peak - low >= max(SHAPED_MIN_LEVELS, SHAPED_MIN_DEPTH * peak):
            if peak < 0.5:
                return eff_mapped_sample, 0
            played = _dmm_note_number(note_obj.note) + 25
            shape = [(t, 64.0 * l / peak) for t, l in pts]
            env = it_volume_envelopes.get(eff_it_instr) if eff_it_instr else None
            peak_level = peak
            if env is not None and env.active:
                src = envelope_ctx["xm"].instruments.get(eff_mapped_sample) if envelope_ctx else None
                smp = src.sample.get(0) if src is not None else None
                if not (smp is not None and smp.loop):
                    peak_level = peak * _envelope_average_over_hold(env, hold_ticks, age) / 64.0
            dmm_vol = min(127, int(round(peak_level)) * 2 + 1)
            baked = _get_or_render_shaped_instrument(envelope_ctx, eff_mapped_sample, eff_it_instr,
                                                      hold_ticks, tempo, played, shape, age)
            if baked is not None:
                return baked, dmm_vol
            rms = (_envelope_segment_area(pts, 0, hold_ticks, power=True) / hold_ticks) ** 0.5
            return eff_mapped_sample, min(127, int(round(rms)) * 2 + 1)
    if envelope_ctx is not None:
        baked = _get_or_render_envelope_instrument(envelope_ctx, eff_mapped_sample, eff_it_instr,
                                                     hold_ticks, tempo, _dmm_note_number(note_obj.note) + 25,
                                                     age)
        if baked is not None:
            return baked, _get_note_volume_for_dmm(note_obj)
    vol = _note_volume_with_envelope(note_obj, hold_ticks, it_volume_envelopes, eff_it_instr, age)
    if eff_mapped_sample == 0:
        vol = 0
    return eff_mapped_sample, vol


# FILES.H: the game keeps ONE directory of MAX_WAD = 2000 lumps for all loaded WADs together
# (a lump whose name already exists only overrides that entry, a new name takes a slot). The stock
# doom2d.wad already uses 1468 of them and DMI0001..DMI0091.
WAD_MAX_LUMPS = 2000
ORIGINAL_WAD_LUMPS = 1468
ORIGINAL_WAD_LAST_DMI = 91

# NOTETAB.BAS: note_tab[i] = INT(1024 * 0.25 * 2^(i/12)), i = 0..95; the engine plays a note at
# instrument_rate * note_tab[note] >> 10 samples/s (SOUND.ASM playnext), clamped to 2000..65535.
_NOTE_TAB = [int(0.25 * 2 ** (i / 12) * 1024) for i in range(96)]
ENGINE_MIN_RATE = 2000
ENGINE_MAX_RATE = 65535


def _report_engine_limits(all_dmm_notes, inst_rate, log):
    """Post-compensation audit using the same per-channel memory as SOUND.ASM."""
    current, channel = [0] * DMM_8_CHANNELS, 0
    total = high = low = 0
    per_inst = {}
    for n in all_dmm_notes:
        if n.volume == 128:                         # channel terminator
            channel = (channel + 1) % DMM_8_CHANNELS
            continue
        if n.instrument:                            # also true for a 0xFE rest in SOUND.ASM
            current[channel] = n.instrument
        inst = current[channel]
        if n.note < 96 and n.volume not in (0, 128) and inst and inst_rate.get(inst):
            total += 1
            r = (inst_rate[inst] * _NOTE_TAB[n.note]) >> 10
            if r > ENGINE_MAX_RATE:
                high += 1
                per_inst[inst] = per_inst.get(inst, 0) + 1
            elif r < ENGINE_MIN_RATE:
                low += 1
                per_inst[inst] = per_inst.get(inst, 0) + 1
    if high or low:
        worst = ", ".join(f"#{i} ({c})" for i, c in sorted(per_inst.items(), key=lambda kv: -kv[1])[:5])
        log(f"Warning: post-compensation audit found {high + low} of {total} notes outside "
            f"{ENGINE_MIN_RATE}..{ENGINE_MAX_RATE} Hz ({high} too high, {low} too low). "
            f"The engine will clamp them; instruments: {worst}.\n")


def _at_frames(idxs, old_arr, loop_len, loop_start, loop_end, last):
    """Vectorized twin of the at() closure in _resample_dmi_time_compressed
    (integer index arithmetic only, hence exact). `idxs` is anything
    int-convertible; returns the gathered values."""
    np = _np
    idxs = np.asarray(idxs, dtype=np.int64)
    if loop_len > 1:
        over = idxs >= loop_end
        idxs = np.where(over, loop_start + (idxs - loop_start) % loop_len, idxs)
        under = idxs < 0
        if loop_start == 0:
            idxs = np.where(under, idxs % loop_len, idxs)
        else:
            idxs = np.where(under, 0, idxs)
    else:
        idxs = np.clip(idxs, 0, last)
    return old_arr[idxs]


def _resample_frames_fast(old, old_length, new_length, factor, loop_len, loop_start,
                          loop_end, last):
    """Vectorized twin of the two resampling loops in
    _resample_dmi_time_compressed (numpy only; the caller keeps the plain
    loops as fallback).

    Every output frame uses the same operations in the same order as the
    scalar code: per-element float arithmetic (identical rounding lane by
    lane) and left-to-right accumulation via cumsum (numpy's plain sum uses
    pairwise summation, which can round the last ulp differently and flip
    an int(round()) - cumsum is sequential like Python's sum). Output bytes
    are therefore identical; only the speed differs."""
    np = _np
    oarr = np.asarray(old, dtype=np.float64)
    if factor < _SINC_MIN_FACTOR:
        pos = np.arange(new_length, dtype=np.float64) * factor
        left = pos.astype(np.int64)
        frac = pos - left
        aa = _at_frames(left, oarr, loop_len, loop_start, loop_end, last)
        bb = _at_frames(left + 1, oarr, loop_len, loop_start, loop_end, last)
        return [int(v) for v in np.rint(aa + (bb - aa) * frac).astype(np.int64)]
    half = max(2, int(math.ceil(_RESAMPLE_TAPS_PER_SIDE * factor)))
    fc = 1.0 / factor
    # Same math.* kernel construction as the scalar path (kept in Python on
    # purpose: numpy's sin/cos can differ by 1 ulp from libm's, and kernels
    # are tiny - a few ms - while the dots below are the real cost).
    kernels = []
    for p in range(_SINC_PHASES):
        frac = p / float(_SINC_PHASES)
        ker = []
        for k in range(-half + 1, half + 1):
            t = k - frac
            x = fc * t
            ker.append(fc * (math.sin(math.pi * x) / (math.pi * x) if x else 1.0)
                       * (0.5 + 0.5 * math.cos(math.pi * t / half)))
        total = sum(ker)
        kernels.append([v / total for v in ker] if total else ker)
    karr = np.asarray(kernels)
    ext = _at_frames(np.arange(-half, old_length + half + 1), oarr,
                     loop_len, loop_start, loop_end, last)
    centres = np.arange(new_length, dtype=np.float64) * factor
    base = np.floor(centres).astype(np.int64)
    ph = np.rint((centres - base) * _SINC_PHASES).astype(np.int64)
    fix = ph >= _SINC_PHASES
    ph = np.where(fix, 0, ph)
    base = np.where(fix, base + 1, base)
    starts = base + 1
    taps = 2 * half
    cols = np.arange(taps)
    out = np.empty(new_length, dtype=np.int64)
    chunk = 4096
    for s in range(0, new_length, chunk):
        e = min(s + chunk, new_length)
        dots = np.cumsum(ext[starts[s:e, None] + cols] * karr[ph[s:e]], axis=1)[:, -1]
        out[s:e] = np.rint(dots).astype(np.int64)
    return out.tolist()


def _resample_dmi_time_compressed(sample, factor):
    """Compress PCM by ``factor`` while reducing its DMI rate by the same factor.

    Small factors use linear interpolation (aliasing is negligible: > 40 dB SNR against ideal
    decimation at 1.3x). From _SINC_MIN_FACTOR upwards each output frame is a Hann-windowed-sinc
    low-pass of the input around the position it replaces (cut-off = the new Nyquist, DC gain 1):
    single interpolated points would fold everything above the new Nyquist back into the audible
    band. Measured against ideal band-limited decimation on the real samples the windowed sinc
    won everywhere it matters (+4.6 dB at 5.8x, +2.8 dB at 3.3x; a box average lost more to
    passband droop than it gained), but it is ~10x slower, hence the threshold."""
    old_length = min(sample.sample_length, len(sample.data))
    if old_length <= 0 or factor <= 1.0:
        return False
    old = sample.data[:old_length]
    new_length = max(1, int(round(old_length / factor)))
    last = old_length - 1
    # Beyond the ends the filter sees what the engine would play there: the loop wraps around
    # (a looped single-cycle waveform is periodic!), otherwise the edge value is held.
    if sample.loop and sample.loop_length > 0:
        loop_start = min(sample.loop_start, last)
        loop_end = min(old_length, sample.loop_start + sample.loop_length)
        loop_len = loop_end - loop_start
    else:
        loop_start = loop_end = loop_len = 0

    def at(idx):
        if loop_len > 1:
            if idx >= loop_end:
                idx = loop_start + (idx - loop_start) % loop_len
            elif idx < 0:
                idx = idx % loop_len if loop_start == 0 else 0
            return old[idx]
        return old[0 if idx < 0 else (last if idx > last else idx)]

    new_data = []
    if _np is not None:
        new_data = _resample_frames_fast(old, old_length, new_length, factor,
                                         loop_len, loop_start, loop_end, last)
    else:
        if factor < _SINC_MIN_FACTOR:
            for j in range(new_length):
                pos = j * factor
                left = int(pos)
                a, b = at(left), at(left + 1)
                new_data.append(int(round(a + (b - a) * (pos - left))))
        else:
            half = max(2, int(math.ceil(_RESAMPLE_TAPS_PER_SIDE * factor)))
            fc = 1.0 / factor
            # kernels for _SINC_PHASES sub-sample offsets (normalised to DC gain 1), and the input with
            # `half` extra frames on both sides taken through at(), so the loop below is a plain dot product
            kernels = []
            for p in range(_SINC_PHASES):
                frac = p / float(_SINC_PHASES)
                ker = []
                for k in range(-half + 1, half + 1):
                    t = k - frac
                    x = fc * t
                    ker.append(fc * (math.sin(math.pi * x) / (math.pi * x) if x else 1.0)
                               * (0.5 + 0.5 * math.cos(math.pi * t / half)))
                total = sum(ker)
                kernels.append([v / total for v in ker] if total else ker)
            ext = [at(i) for i in range(-half, old_length + half + 1)]
            for j in range(new_length):
                centre = j * factor
                base = int(math.floor(centre))
                phase = int(round((centre - base) * _SINC_PHASES))
                if phase >= _SINC_PHASES:
                    phase = 0
                    base += 1
                start = base - half + 1 + half              # index into ext of tap k = -half + 1
                new_data.append(int(round(sum(map(_mul, ext[start:start + 2 * half], kernels[phase])))))
    if sample.loop and sample.loop_length > 0:
        old_end = min(old_length, sample.loop_start + sample.loop_length)
        new_start = min(max(0, int(round(sample.loop_start / factor))), new_length - 1)
        new_end = min(max(new_start + 1, int(round(old_end / factor))), new_length)
        sample.loop_start, sample.loop_length = new_start, new_end - new_start
        sample.loop = sample.loop_length > 0
    else:
        sample.loop_start = sample.loop_length = 0
    sample.data = new_data
    sample.sample_length = new_length
    # A one-frame loop is valid DMI, but it is effectively DC and can sound
    # like a hum. Keep it rather than silently changing duration, and report it.
    return sample.loop and sample.loop_length == 1


_SINC_PHASES = 128            # sub-sample positions the windowed-sinc kernel is precomputed for
_SINC_MIN_FACTOR = 2.5         # below this plain linear interpolation is used (fast, alias-free enough)
_RESAMPLE_TAPS_PER_SIDE = 8    # sinc lobes on each side of the centre (in output-frame units)


def _fit_dmm_instruments_to_engine(xm, all_dmm_notes, log):
    """Keep DMI playback inside SOUND.ASM's 2000..65535 Hz voice range.

    Lower an overflowing DMI base rate and time-compress its PCM by the
    reciprocal ratio. The engine then traverses the original waveform at the
    intended speed, without collapsing different high notes to 65535 Hz.
    """
    ranges, current, channel = {}, [0] * DMM_8_CHANNELS, 0
    for event in all_dmm_notes:
        if event.volume == 128:
            channel = (channel + 1) % DMM_8_CHANNELS
            continue
        # SOUND.ASM selects an instrument before handling a 0xFE rest.
        if event.instrument:
            current[channel] = event.instrument
        if event.note < 96 and 0 < event.volume < 128 and current[channel]:
            lo_hi = ranges.setdefault(current[channel], [event.note, event.note])
            lo_hi[0] = min(lo_hi[0], event.note)
            lo_hi[1] = max(lo_hi[1], event.note)

    changed, unsplittable_range, unsplittable_size = [], [], []
    for inst_num, (low_note, high_note) in sorted(ranges.items()):
        inst = xm.instruments.get(inst_num)
        sample = inst.sample.get(0) if inst is not None else None
        if sample is None or sample.sample_length <= 0 or not sample.data:
            continue
        old_rate = xm_to_sample_rate(sample.relative_note, sample.finetune)
        if old_rate <= 0:
            continue
        old_length = min(sample.sample_length, len(sample.data))
        low_limit = (ENGINE_MIN_RATE * 1024 + _NOTE_TAB[low_note] - 1) // _NOTE_TAB[low_note]
        engine_high_limit = (ENGINE_MAX_RATE * 1024) // _NOTE_TAB[high_note]
        # The same time-compression factor lowers both DMI rate and length.
        # Thus a rate no higher than this also makes the stored PCM fit in
        # DMI's u16 length field, without a second resampling pass in writer.
        size_high_limit = (old_rate * DMI_MAX_LENGTH) // old_length
        if engine_high_limit < low_limit:
            # The requested note span itself exceeds the engine's usable
            # range at any one DMI base rate.
            unsplittable_range.append((inst_num, low_note, high_note))
            continue
        dmi_high_limit = min(DMI_MAX_LENGTH, size_high_limit)
        if dmi_high_limit < low_limit:
            # Pitch fits the engine, but DMI's u16 data/rate fields do not.
            unsplittable_size.append((inst_num, low_note, high_note))
            continue
        upper_limit = min(engine_high_limit, dmi_high_limit)
        # A safe rate exists exactly in this closed interval. Do not favour
        # the upper edge so strongly that a valid low-note bound is lost.
        target = max(low_limit, min(old_rate, upper_limit))
        if target <= 1:
            unsplittable_size.append((inst_num, low_note, high_note))
            continue
        actual_rate = None
        while target > 1:
            rel, fine = sample_rate_to_xm(target, False)
            actual_rate = xm_to_sample_rate(rel, fine)
            new_length = int(round(old_length * actual_rate / old_rate))
            if actual_rate <= upper_limit and new_length <= DMI_MAX_LENGTH:
                break
            target -= 1
        if actual_rate is None or actual_rate < low_limit or actual_rate >= old_rate or \
                actual_rate > upper_limit or new_length > DMI_MAX_LENGTH:
            if actual_rate is None or actual_rate < low_limit:
                unsplittable_size.append((inst_num, low_note, high_note))
            continue
        factor = old_rate / float(actual_rate)
        one_frame_loop = _resample_dmi_time_compressed(sample, factor)
        sample.relative_note, sample.finetune = rel, fine
        changed.append((inst_num, old_rate, actual_rate, factor, low_note, high_note, one_frame_loop))

    if changed:
        details = ", ".join(f"#{n}: {old}->{new} Hz ({factor:.2f}x, notes {lo}..{hi})"
                            for n, old, new, factor, lo, hi, _one_frame in changed[:8])
        suffix = "" if len(changed) <= 8 else f"; and {len(changed) - 8} more"
        log(f"Engine-rate compensation: time-compressed {len(changed)} DMI sample(s): {details}{suffix}.\n")
        one_frame = [n for n, _old, _new, _factor, _lo, _hi, collapsed in changed if collapsed]
        if one_frame:
            log(f"Warning: compensation reduced loop(s) to one PCM frame for instruments {one_frame}; "
                "they remain valid but may sound like a steady tone.\n")
    if unsplittable_range:
        details = ", ".join(f"#{n} (notes {lo}..{hi})" for n, lo, hi in unsplittable_range[:8])
        suffix = "" if len(unsplittable_range) <= 8 else f"; and {len(unsplittable_range) - 8} more"
        log(f"Warning: {len(unsplittable_range)} instrument(s) span too wide a pitch range for one "
            f"DMI rate; use separate instrument copies for different note ranges: {details}{suffix}.\n")
    if unsplittable_size:
        details = ", ".join(f"#{n} (notes {lo}..{hi})" for n, lo, hi in unsplittable_size[:8])
        suffix = "" if len(unsplittable_size) <= 8 else f"; and {len(unsplittable_size) - 8} more"
        log(f"Warning: {len(unsplittable_size)} instrument(s) cannot fit both their required rate "
            f"and PCM data in one DMI; split the sample into copies/chunks or reduce its fidelity: {details}{suffix}.\n")

def _get_note_volume_for_dmm(note):
    result = 127
    # Valid XM "set volume" commands in the volume column occupy the raw byte
    # range 0x10-0x50 inclusive (level 0-64, i.e. note.volume - 16). The old
    # check `note.volume < 64` silently dropped bytes 0x40-0x50 (levels
    # 48-64) back to the "no volume set" default of 127, and the unclamped
    # formula could itself produce 129 for the very top level (0x50), one
    # more than a DMM volume byte can hold. Both are fixed here.
    if 0x10 <= note.volume <= 0x50:
        result = min(127, (note.volume - 16) * 2 + 1)
    if note.effect == EFFECT_VOLUME:
        result = note.effect_parameter * 2 + 1 if note.effect_parameter < 64 else 127
    return result


def _select_busiest_channels(xm, want=DMM_8_CHANNELS, by="loudness"):
    """Rank the source module's channels by how much they actually
    contribute, and return the `want` most prominent ones as a sorted list
    of 0-based indices.

    `by` selects the ranking metric:
      "count"    - the number of cells carrying a note or an instrument
                   (the original heuristic). Simple, but can rank a channel
                   full of quiet filler/arpeggio notes above a channel that
                   only plays a handful of loud, prominent lead/bass hits
                   per bar - on a real song that dropped exactly the
                   instrument a listener would call "the loudest thing in
                   the mix".
      "loudness" - each note-trigger weighted by its own effective DMM
                   volume (_get_note_volume_for_dmm, the same volume the
                   final conversion itself uses), so a channel's score is
                   "loudness x how often it's heard" rather than a flat
                   count. Matches what a listener actually notices more
                   closely than a raw count does.

    Either way, ranking happens over the song's real playback order (a
    pattern that repeats several times is weighted several times, matching
    how much of the song each channel really fills), not just unique
    pattern data.
    """
    total = xm.number_of_channels
    if total <= want:
        return list(range(total))
    if by not in ("count", "loudness"):
        raise ValueError(f'Unknown channel ranking "{by}" (expected "count" or "loudness").')
    scores = [0] * total
    order = list(xm.pattern_order[:xm.track_length]) or list(range(xm.number_of_patterns()))
    for pat_idx in order:
        p = xm.patterns[pat_idx]
        for row in range(p.rows):
            for ch in range(total):
                n = p.get_note(row, ch)
                if by == "count":
                    if n.note or n.instrument:
                        scores[ch] += 1
                elif n.note and n.note != NOTE_STOP and n.instrument:
                    scores[ch] += _get_note_volume_for_dmm(n)
    # Highest score first; ties keep the lower (more "natural") channel
    # number, so behaviour degrades gracefully towards the old first-8 rule
    # whenever usage/loudness is roughly even across channels.
    ranked = sorted(range(total), key=lambda ch: (-scores[ch], ch))
    return sorted(ranked[:want])


def _emit_long(out, note, inst, vol, delay):
    """Append a note held for `delay` ticks the way the original DMM data does: the delay byte is
    0..255 with 0 meaning 256; anything longer is that note (delay 0) followed by ONE rest for
    the remainder. (This used to split at 255 and chop rests every 256 ticks, so a converted
    file never matched the original byte for byte.)"""
    if delay <= 255:
        out.append(DMMNote(note, inst, vol, delay))
        return
    out.append(DMMNote(note, inst, vol, 0))
    rest = delay - 256
    if rest > 0:
        out.append(DMMNote(254, rest // 256, 0, rest % 256))


def _apply_channel_volume(n, vols, ch):
    """Channel volume memory. A note with no volume column/Cxx and no instrument number of its
    own keeps the channel's previous volume (in every tracker); only an explicit instrument
    resets it to the sample default. Without this such notes were played at full volume."""
    if n.note == 0 or n.note == NOTE_STOP:
        return n
    if n.effect == EFFECT_VOLUME:
        vols[ch] = min(n.effect_parameter, 64)
        return n
    if 0x10 <= n.volume <= 0x50:
        vols[ch] = n.volume - 0x10
        return n
    if (n.instrument == 0 or n.keep_volume) and vols[ch] is not None:
        n = n.copy()
        n.volume = 0x10 + int(round(vols[ch]))
        return n
    vols[ch] = None      # explicit instrument, no volume: sample default again
    return n


def _copy_state(state):
    """Copy of the playback state carried from pattern to pattern: speed/tempo plus the per-channel
    memory that survives a pattern boundary in every tracker (last instrument, last volume)."""
    out = dict(state)
    for k in ("inst", "itinst", "vol", "carry", "slidemem", "retrigmem", "offmem", "holdover"):
        if k in out:
            out[k] = list(out[k])
    return out


def _dmm_note_number(xm_note):
    """XM/IT note number -> DMM note byte. SOUND.ASM indexes NOTETAB (96 words) with the raw,
    *unsigned* note byte: DMM notes are 0..95 (= XM notes 25..120), 0xFE is the rest marker and
    0xFF means "no new note, only volume/instrument" (NOT a note). Anything outside 0..95 would
    read past the table, so fold it by whole octaves so the pitch class survives."""
    v = xm_note - 25
    while v < 0:
        v += 12
    while v > 95:
        v -= 12
    return v


# XM/MOD position jump (Bxx: jump to order-table position xx). The S3M/IT
# loaders never produce this effect value (their own Bxx is resolved into
# restart information at load time), so any 0x0B cell seen here comes from an
# XM/MOD source file.
EFFECT_POSITION_JUMP = 0x0B


def _find_pattern_jump(xm, pat_idx, channel_map):
    """First Bxx/Cxx in one pattern in row-major order, or None.

    Returns (row, kind, param) with kind "jump" (Bxx) or "break" (Cxx).
    Only the first one matters for playback state: a real tracker acts on
    it and never executes any later rows/effects of that pattern pass.
    """
    channels = xm.number_of_channels
    if channel_map is None:
        channel_map = list(range(channels))
    src_chans = sorted({channel_map[c] for c in range(channels)})
    pattern = xm.patterns[pat_idx]
    for row in range(pattern.rows):
        for ch in src_chans:
            t = pattern.get_note(row, ch)
            if t.effect == EFFECT_POSITION_JUMP:
                return (row, "jump", t.effect_parameter)
            if t.effect == EFFECT_PATTERN_BREAK:
                return (row, "break", t.effect_parameter)
    return None


def _truncated_pattern_view(xm, pat_idx, jump_row):
    """Pattern limited to rows 0..jump_row (inclusive).

    A Bxx/Cxx in row `jump_row` still executes that row's own effects (e.g.
    an Fxx speed change on the very same row takes effect) and then leaves
    the pattern, so everything after it never plays on that pass. Returns
    the original pattern when there is no jump or it sits on the last row.
    The view shares the (never mutated by conversion) PatternNote cells.
    """
    src = xm.patterns[pat_idx]
    if jump_row is None or jump_row >= src.rows - 1:
        return src
    view = Pattern()
    view.resize(jump_row + 1, src.channels)
    for r in range(jump_row + 1):
        for ch in range(src.channels):
            view.set_note(r, ch, src.get_note(r, ch))
    return view


def _detect_subsongs(xm, channel_map=None):
    """Split the order table into real playback songs, the way OpenMPT /
    libopenmpt itself reports subsongs for the same file (verified order by
    order and to 0.1 s on BGM12.XM: 6:04 / 0:21 / 4:09 / 0:08).

    Returns a list of {"start", "orders", "loop_to", "restart"} sorted by
    start, where "orders" are order-table positions in actual play order
    (jumps already expanded) and "restart" is the index into "orders" the
    DMM loop should return to (0 for songs that just end - DMM loops those
    from their own start, like single-file conversion always did).

    Partition semantics: every order-table position belongs to exactly one
    song. Songs are found by playing from the lowest not-yet-covered order
    and following jumps (Bxx only - a Cxx break never leaves the pattern,
    the whole pattern converts anyway). The song stops:
      * before an order already covered by an earlier song (a jump into
        another song's territory - e.g. an intro that rejoins the main
        loop, or a jump back to the start);
      * on revisiting one of its own orders (a real loop closes -
        loop_to = that order position, played exactly once like OpenMPT
        counts a single pass for the duration);
      * at the end of the table or on a jump past it (loop_to = None).
    The old overlapping grouping (same loop replayed inside every song that
    can reach it, longest intro swallowing the plain main song) is gone:
    on BGM12.XM it produced 3 songs (4:24 / 10:14 / 0:16) with the 0..26
    main song missing entirely, while OpenMPT plays 4 (6:04 / 0:21 /
    4:09 / 0:08).
    """
    order = list(xm.pattern_order[:xm.track_length])
    n = len(order)
    if n == 0:
        return []
    if channel_map is None:
        channel_map = list(range(xm.number_of_channels))
    jump_cache = {}
    for pat_idx in set(order):
        found = _find_pattern_jump(xm, pat_idx, channel_map)
        if found is not None:
            jump_cache[pat_idx] = found

    songs = []
    covered = set()
    for start in range(n):
        if start in covered:
            continue
        seq, seen, pos, loop_to = [], set(), start, None
        while pos is not None and pos not in seen and pos not in covered:
            seen.add(pos)
            seq.append(pos)
            jumped = jump_cache.get(order[pos])
            if jumped is not None and jumped[1] == "jump":
                nxt = jumped[2]
                pos = nxt if 0 <= nxt < n else None
            else:
                pos = pos + 1 if pos + 1 < n else None
        if pos is not None and pos in seen:
            loop_to = pos
        # pos in `covered` or None: the song just ends (foreign territory
        # or table end) - DMM loops it from its own start.
        if not seq:
            continue  # unreachable (`start` itself is uncovered), kept defensive
        covered.update(seq)
        songs.append({"start": start, "orders": seq, "loop_to": loop_to,
                      "restart": seq.index(loop_to) if loop_to is not None else 0})
    return sorted(songs, key=lambda g: g["start"])


def _subsong_truncations(xm, channel_map=None):
    """Patterns cut short by a mid-pattern jump/break: pat_idx -> last
    played row. A jump on the very last row changes nothing (the pattern
    plays through, only the flow continues elsewhere)."""
    if channel_map is None:
        channel_map = list(range(xm.number_of_channels))
    out = {}
    for pat_idx in range(xm.number_of_patterns()):
        found = _find_pattern_jump(xm, pat_idx, channel_map)
        if found is not None and found[0] < xm.patterns[pat_idx].rows - 1:
            out[pat_idx] = found[0]
    return out


def _build_subsong_view(xm, song_orders, trunc_map):
    """Standalone module for one detected subsong.

    The song's order list is already real play order (jumps expanded), so
    the view carries no jumps at all: truncated patterns are row-copies and
    every remaining Bxx/Cxx cell is cleared (flow lives in the order list
    now). Speeds therefore flow naturally along the song with no cross-
    subsong leakage and no alignment hacks. Instruments/samples are
    deep-copied per view (baking and engine-rate fitting mutate them);
    volume envelopes are shared read-only.
    """
    view = XMModule()
    view.name = xm.name
    view.comment = xm.comment
    view.global_volume = xm.global_volume
    view.master_volume = xm.master_volume
    view.start_speed = xm.start_speed
    view.start_tempo = xm.start_tempo
    view.linear_slides = xm.linear_slides
    view.extended_filter_range = xm.extended_filter_range
    if hasattr(xm, "fast_volume_slides"):
        view.fast_volume_slides = xm.fast_volume_slides
    view.number_of_channels = xm.number_of_channels
    view.it_volume_envelopes = xm.it_volume_envelopes
    view.instruments = copy.deepcopy(xm.instruments)
    view.patterns = [Pattern() for _ in range(256)]
    slot_of = {}
    seq = []
    for orig in (xm.pattern_order[o] for o in song_orders):
        key = (orig, orig in trunc_map)
        if key not in slot_of:
            src = xm.patterns[orig]
            dst = Pattern()
            dst.resize(src.rows, src.channels)
            for r in range(src.rows):
                for ch in range(src.channels):
                    dst.set_note(r, ch, src.get_note(r, ch).copy())
            if key[1]:
                dst.resize(trunc_map[orig] + 1, src.channels)
            for r in range(dst.rows):
                for ch in range(dst.channels):
                    cell = dst.get_note(r, ch)
                    if cell.effect in (EFFECT_POSITION_JUMP, EFFECT_PATTERN_BREAK):
                        cell.effect = 0
                        cell.effect_parameter = 0
                        dst.set_note(r, ch, cell)
            slot_of[key] = len(slot_of)
            view.patterns[slot_of[key]] = dst
        seq.append(slot_of[key])
    view.pattern_order = seq + [0] * (256 - len(seq))
    view.track_length = len(seq)
    return view, seq


def _fresh_playback_state(xm):
    return {"speed": xm.start_speed, "tempo": xm.start_tempo,
            "fast_slides": bool(getattr(xm, "fast_volume_slides", False))}


def _holdover_set_level(holdover_entry, level):
    """Holdover tuple with its volume level replaced, preserving the note's
    age (see _pattern_to_dmm_notes: entries are (note, it, inst, level, age)).
    The old inline `[:4] + (setlvl,)` wrote the new level into the *age*
    slot instead - resetting the envelope clock on every mid-note volume
    change and replaying the attack at full volume on the next carry."""
    if not holdover_entry:
        return holdover_entry
    parts = list(holdover_entry)
    if len(parts) <= 4:
        return (parts[0], parts[1], parts[2], float(level))
    return (parts[0], parts[1], parts[2], float(level), parts[4])


def _holdover_decayed_level(it_volume_envelopes, it, level, age):
    """Channel level (0..64) a carried note actually has when the next
    pattern starts: its stored level scaled by the instrument envelope's
    instantaneous value at `age` ticks. Envelopeless notes (and sustained
    ones, which gate at their sustain point) come back unchanged."""
    env = it_volume_envelopes.get(it) if it else None
    if env is None or not env.active or not env.points or age <= 0:
        return float(level)
    return float(level) * _envelope_instant_value(env, age) / 64.0


def _entry_ring_key(state, it_volume_envelopes):
    """What the previous table entry left ringing, as hashable state.

    Two occurrences of one pattern can share a DMM slot exactly when these
    coincide: channel instrument/volume/slide/retrigger/offset memories plus
    the audible part of the holdover note, if any (a decayed-to-silence
    envelope carries nothing even though the voice is "still playing").
    A carry stores the note's *decayed* entry level (rounded to half a
    level, i.e. one DMM volume step): the same note picked up fresh vs as
    an old quiet tail needs a different retrigger volume, so those two
    entries must not share one slot. Speed/tempo/carry are deliberately
    excluded: speeds are resolved separately per pattern, carry only shifts
    sub-tick rounding.
    """
    hold = state.get("holdover", [])
    holdkey = []
    for h in hold:
        if not h:
            holdkey.append(None)
            continue
        parts = list(h)
        age = parts[4] if len(parts) > 4 else 0
        it = parts[1] if len(parts) > 1 else 0
        if _holdover_audible(it_volume_envelopes, it, age):
            level = parts[3] if len(parts) > 3 else 0
            decayed = _holdover_decayed_level(it_volume_envelopes, it, level, age)
            holdkey.append(tuple(parts[:3]) + (round(decayed * 2) / 2,))
        else:
            holdkey.append(None)
    def freeze(v):
        if isinstance(v, list):
            return tuple(freeze(x) for x in v)
        if isinstance(v, tuple):
            return tuple(freeze(x) for x in v)
        return v
    return (
        freeze(state.get("inst", [])),
        freeze(state.get("itinst", [])),
        freeze(state.get("vol", [])),
        freeze(state.get("slidemem", [])),
        freeze(state.get("retrigmem", [])),
        freeze(state.get("offmem", [])),
        tuple(holdkey),
    )


def _group_occurrences(order, order_states, it_volume_envelopes):
    """Group each pattern's order positions by entry ring state.

    Returns {pattern: [(key, [positions])]} with groups in first-occurrence
    order. Two occurrences share a DMM slot exactly when everything the
    previous table entry left behind coincides - the forum rule ("same
    endings share one copy") made exact.
    """
    grouped = {}
    for pos, pat in enumerate(order):
        key = _entry_ring_key(order_states[pos], it_volume_envelopes)
        bucket = grouped.setdefault(pat, [])
        for i, (k, poss) in enumerate(bucket):
            if k == key:
                poss.append(pos)
                break
        else:
            bucket.append((key, [pos]))
    return grouped


def _assign_slots(order, grouped, order_states, speed_override, dmm_pattern_number,
                  fallback_state):
    """Assign DMM slots: each pattern's first group keeps its own index
    (stable numbering), further groups append fresh slots after all existing
    ones. Each slot's bake state is its first occurrence's walk state, with
    speed/tempo from `speed_override` when present (jump-aware per-pattern
    speeds). Returns (slot_entries, slot_pattern, remapped_order,
    total_slots, dup_info) with dup_info = [(pattern, new_slot, [orders])].
    """
    slot_entries = {}
    slot_pattern = {}
    slot_of_pos = {}
    next_free = dmm_pattern_number
    dup_info = []
    for pat in sorted(grouped, key=lambda p: min(pos for _, poss in grouped[p] for pos in poss)):
        for gi, (key, poss) in enumerate(grouped[pat]):
            if gi == 0:
                slot = pat
            else:
                slot = next_free
                next_free += 1
                dup_info.append((pat, slot, list(poss)))
            st = _copy_state(order_states[poss[0]])
            if pat in speed_override:
                st["speed"], st["tempo"] = speed_override[pat]
            slot_entries[slot] = st
            slot_pattern[slot] = pat
            for pos in poss:
                slot_of_pos[pos] = slot
    # Orphan slots (patterns never reached in the table) keep indices aligned.
    # Out-of-range table entries (corrupt files) keep theirs too; make sure
    # the total covers every assigned slot and fill any gaps with silence.
    total = next_free
    if slot_pattern:
        total = max(total, max(slot_pattern.values()) + 1)
    for idx in range(total):
        if idx not in slot_entries:
            slot_entries[idx] = _copy_state(fallback_state)
            slot_pattern[idx] = idx
    remapped = [slot_of_pos[pos] for pos in range(len(order))]
    return slot_entries, slot_pattern, remapped, total, dup_info


def _compute_pattern_entry_states(xm, dmm_pattern_number, log, channel_map=None, envelope_ctx=None,
                                   volume_slides="off"):
    """Determine per-pattern bake states for backward conversion, walking the
    song's order table rather than assuming patterns play back in raw
    storage order 0..N-1.

    Returns (slot_entries, slot_pattern, remapped_order, total_slots):
    DMM slot -> bake state, DMM slot -> source XM pattern, the order table
    with repeated patterns mapped onto per-occurrence duplicate slots where
    needed, and the total slot count.

    This matters because Set Speed/Tempo effects accumulate into state that
    carries from one pattern into the next, and _pattern_to_dmm_notes()
    bakes whatever state is current *at conversion time* into every note's
    delay field - DMM has no live effect processing during playback, just
    fixed per-note delay ticks. Converting patterns in storage order (as
    this used to do) uses whatever speed happened to be left over from
    unrelated, earlier-in-storage patterns, which is very often not the
    speed actually in effect when that pattern is really reached in the
    song. On a real S3M this silently converted three patterns that are
    always played at speed 6 using speed 10 leaked over from earlier
    storage slots, adding up to roughly 30 seconds of extra length over
    the whole song.

    Position jumps (XM/MOD Bxx) and pattern breaks (Cxx) are followed as
    well, but only for speed/tempo: an Fxx speed/tempo change inside a
    pattern that real playback never reaches (because an earlier pattern
    jumps over it, or a subsong starts past it) must not leak into the
    patterns that play after it. Every distinct playback path is simulated
    starting from order 0, from every jump target, and from right after
    every jumping pattern (songs like the multi-subsong BGM16.XM start
    sections mid-table, with no jump pointing at them). Channel memories
    and holdover notes always come from the linear walk instead, and an
    order repeat whose entry state differs from an earlier occurrence gets
    its own duplicate DMM slot - one slot can no longer smear two
    different tails together. DMM itself stays a linear order of full
    patterns (a mid-pattern jump cannot cut a DMM pattern short).

    Known residual limitation: the loop-restart byte points at a single
    order position, so when a song loops back into a pattern whose ringing
    state then differs from its linear entry (tail of the last pattern vs
    the table neighbour), one DMM slot cannot serve both passes - the
    linear entry wins and the loop pass is approximate. It logs nothing
    special; everything else (mid-table repeats included) gets exact slots.
    """
    order = list(xm.pattern_order[:xm.track_length])
    if channel_map is None:
        channel_map = list(range(xm.number_of_channels))

    # First Bxx/Cxx per pattern (row-major). Patterns without jumps play
    # through, exactly like the old linear walk assumed everywhere.
    jump_of = {}
    for pat_idx in range(xm.number_of_patterns()):
        found = _find_pattern_jump(xm, pat_idx, channel_map)
        if found is not None:
            jump_of[pat_idx] = found

    # Linear walk over the order table (full patterns, table order): every
    # order position records its entry state. DMM itself always plays its
    # own order table linearly, so whatever the previous *table* entry left
    # ringing is what the next pattern has to start with.
    #
    # walk_ctx shares the caller's envelope context exactly when there are
    # no jumps (the historical behaviour, including populating the shared
    # baking cache in walk order). With jumps the walk runs dry: baking
    # only affects emitted note bytes, never the carried state, and the
    # real pass bakes what it needs through its own shared cache.
    walk_ctx = envelope_ctx
    if jump_of and order:
        walk_ctx = {"xm": xm, "envelopes": xm.it_volume_envelopes,
                    "cache": {}, "extra_count": 0, "offsets": False,
                    "dry_run": True}
    order_states = []
    state = _fresh_playback_state(xm)
    for pat_idx in order:
        order_states.append(_copy_state(state))
        _pattern_to_dmm_notes(xm.patterns[pat_idx], xm.number_of_channels, state, channel_map,
                              xm.it_volume_envelopes, walk_ctx, volume_slides)

    if not jump_of or not order:
        speed_override = {}
        fallback_state = state
        first_speed, conflicts, main_reachable = {}, {}, set(order)
        warn_reuse = True

    else:
        # --- Jump-aware speeds --------------------------------------------------
        # Views truncated at the jump row: effects after the jump never execute
        # on that pass and must not leak speed/tempo into what follows. Only
        # speed/tempo are taken from here (per pattern, earliest path wins);
        # memories and holdovers always come from the linear walk above.
        views = {p: _truncated_pattern_view(xm, p, jump_of[p][0]) for p in jump_of}

        targets, followers = set(), set()
        for pos, pat_idx in enumerate(order):
            if pat_idx in jump_of:
                _row, kind, param = jump_of[pat_idx]
                if kind == "jump" and 0 <= param < len(order):
                    targets.add(param)
                if pos + 1 < len(order):
                    followers.add(pos + 1)
        starts = sorted({0} | targets | followers)
        max_steps = max(64, 4 * max(1, len(order)))

        # Simulation passes must not pollute the real conversion: envelope/
        # shape baking and sample-offset copies allocate new xm.instruments
        # slots as a side effect, so snapshot the table and drop whatever the
        # throwaway passes added afterwards. (Channel memory, speed/tempo and
        # fractional carry - the actual entry state - are unaffected by using
        # a throwaway context.)
        saved_instruments = dict(xm.instruments)
        first_speed = {}      # pat_idx -> (speed, tempo) of the earliest path
        conflicts = {}        # pat_idx -> set((speed, tempo)) over all paths
        main_reachable = set()
        try:
            for start in starts:
                state = _fresh_playback_state(xm)
                # dry_run: only speed/tempo are read out of these passes, not
                # baked PCM - skip sample rendering/offset copies entirely.
                sim_ctx = {"xm": xm, "envelopes": xm.it_volume_envelopes,
                           "cache": {}, "extra_count": 0, "offsets": False,
                           "dry_run": True}
                seen_orders = set()
                pos = start
                steps = 0
                while 0 <= pos < len(order) and steps < max_steps:
                    if pos in seen_orders:
                        break  # song loop on this path - further passes add no new states
                    seen_orders.add(pos)
                    pat_idx = order[pos]
                    key = (state["speed"], state["tempo"])
                    if pat_idx not in first_speed:
                        first_speed[pat_idx] = key
                    conflicts.setdefault(pat_idx, set()).add(key)
                    if start == 0:
                        main_reachable.add(pat_idx)
                    _pattern_to_dmm_notes(views.get(pat_idx, xm.patterns[pat_idx]),
                                          xm.number_of_channels, state, channel_map,
                                          xm.it_volume_envelopes, sim_ctx, volume_slides)
                    jumped = jump_of.get(pat_idx)
                    if jumped is not None and jumped[1] == "jump":
                        param = jumped[2]
                        if not (0 <= param < len(order)):
                            break  # jumped past the end of the song
                        pos = param
                    else:
                        pos += 1  # plain advance, also for Cxx (break row
                                  # inside the pattern is out of scope: the
                                  # whole pattern converts anyway)
                    steps += 1
        finally:
            for key in [k for k in xm.instruments if k not in saved_instruments]:
                del xm.instruments[key]

    if jump_of and order:
        speed_override = first_speed
        fallback_state = _fresh_playback_state(xm)
        warn_reuse = False

    grouped = _group_occurrences(order, order_states, xm.it_volume_envelopes)
    slot_entries, slot_pattern, remapped, total, dup_info = _assign_slots(
        order, grouped, order_states, speed_override, dmm_pattern_number, fallback_state)
    if total > 255:
        raise ConversionError(
            f"{total} pattern slots needed (order repeats with different entry "
            f"states); DMM allows at most 255 patterns"
            + (" - try split_subsongs=True." if jump_of else "."))
    if warn_reuse:
        warned = set()
        for pat, glist in grouped.items():
            if len(glist) == 1:
                speeds = sorted({(order_states[p]["speed"], order_states[p]["tempo"])
                                 for p in glist[0][1]})
                if len(speeds) > 1 and pat not in warned:
                    warned.add(pat)
                    log(f"Warning: pattern {pat} is reused later in the song at a different "
                        f"speed/tempo ({speeds[0][0]}/{speeds[0][1]} vs "
                        f"{speeds[-1][0]}/{speeds[-1][1]}); DMM can only store one "
                        f"version of it, so the repeat will keep the first "
                        f"occurrence's speed.\n")

    multi = {p: sorted(v) for p, v in conflicts.items() if len(v) > 1}
    if multi:
        detail = ", ".join(
            f"{p} ({' vs '.join(f'{s}/{t}' for s, t in v)})"
            for p, v in sorted(multi.items())[:8])
        suffix = "" if len(multi) <= 8 else f"; and {len(multi) - 8} more"
        log(f"Warning: {len(multi)} pattern(s) are reached at different speed/tempo on "
            f"different playback paths (position jumps / subsongs); DMM keeps the earliest "
            f"path's speed: {detail}{suffix}.\n")
    in_order = set(order)
    skipped = sorted(p for p in in_order
                     if p < xm.number_of_patterns() and p not in main_reachable
                     and xm.patterns[p].rows > 0)
    if skipped or multi:
        bits = []
        if skipped:
            heads = ", ".join(str(p) for p in skipped[:12])
            suffix = "" if len(skipped) <= 12 else f", and {len(skipped) - 12} more"
            bits.append(f"patterns {heads}{suffix} are in the order table but never play "
                        f"from order 0 (loops/subsongs)")
        if multi:
            bits.append(f"{len(multi)} pattern(s) need different speeds on different paths")
        log(f"Note: the song uses position jumps (Bxx) or breaks (Cxx) - {'; '.join(bits)}. "
            f"Everything is still converted in table order with the earliest real path's "
            f"speed, and a jump in the middle of a pattern does not cut it short in DMM.\n")
    elif jump_of:
        log(f"Note: the song uses {len(jump_of)} pattern(s) with position jumps "
            f"(Bxx) or breaks (Cxx); speeds are taken from real playback paths.\n")
    if dup_info:
        heads = ", ".join(f"pat {p} also as slot {s}" for p, s, _ in dup_info[:6])
        suffix = "" if len(dup_info) <= 6 else f"; and {len(dup_info) - 6} more"
        log(f"Note: {len(dup_info)} extra pattern slot(s) for order repeats with different "
            f"entry states - a repeated pattern continues whatever its actual table "
            f"neighbour left ringing ({heads}{suffix}).\n")
    return slot_entries, slot_pattern, remapped, total


def _pattern_to_dmm_notes(pattern, channels, state, channel_map=None, it_volume_envelopes=None,
                           envelope_ctx=None, volume_slides="off"):
    # channel_map[dmm_ch] = which channel of the SOURCE module to read for
    # DMM channel slot dmm_ch. Defaults to the identity (0,1,2,...) - i.e.
    # the source module's first `channels` channels, in order - which is
    # the original, always-first-8 behaviour. A non-identity map is how
    # "use the 8 busiest channels" (see _select_busiest_channels) actually
    # takes effect: everything below still only ever touches DMM_8_CHANNELS
    # slots, it just reads a different source channel for each slot.
    if channel_map is None:
        channel_map = list(range(channels))
    if it_volume_envelopes is None:
        it_volume_envelopes = {}
    converted = [[] for _ in range(DMM_8_CHANNELS)]
    delay = [0] * DMM_8_CHANNELS
    # Fractional tick remainder carried row-to-row per channel (see the
    # accumulation below) - without this, rounding each row's tick count
    # independently silently drops whatever's left of the fraction every
    # single row, and that loss compounds over 64 rows and every repeat of
    # the pattern in the song into several real seconds of drift.
    # The fractional remainder is also carried from one pattern into the next (kept in `state`):
    # dropping it at every pattern end made a converted song ~0.1-0.2 % too short whenever
    # speed*165/tempo isn't an integer (e.g. 7.92 ticks/row -> 253.44 ticks per 32 rows).
    tick_carry = state.setdefault("carry", [0.0] * DMM_8_CHANNELS)
    # Channel memory (last instrument, last volume level) lives in `state` so it survives pattern
    # boundaries. It used to be reset at every pattern start, which silenced any note that relied
    # on the instrument selected in an earlier pattern.
    remembered_instrument = state.setdefault("inst", [0] * DMM_8_CHANNELS)
    remembered_it_instrument = state.setdefault("itinst", [0] * DMM_8_CHANNELS)
    chan_volume = state.setdefault("vol", [None] * DMM_8_CHANNELS)   # live channel level 0..64
    slide_mem = state.setdefault("slidemem", [None] * DMM_8_CHANNELS)  # last volume slide (D00/A00)
    retrig_mem = state.setdefault("retrigmem", [None] * DMM_8_CHANNELS)  # last retrigger (Q00)
    offset_mem = state.setdefault("offmem", [0] * DMM_8_CHANNELS)        # last sample offset (O00/900)
    # What's still genuinely sounding (triggered, not stopped) on each channel as this pattern
    # starts, carried over from wherever this pattern is actually reached in play order. Needed
    # because of a real DMM/engine limitation found in SOUND.ASM: playblk zeroes the *sounding
    # voice* (irate) for all 8 channels every time a new pattern starts, even when nothing in the
    # new pattern's row 0 touches that channel - unlike XM/IT/S3M/MOD, where a note simply keeps
    # ringing across a pattern boundary unless something explicitly stops it. Without this, any
    # note that's still audible when one pattern ends and isn't re-triggered on row 0 of the next
    # goes silent on the real engine even though nothing in the source ever told it to stop.
    # Each entry is (raw_note, effective_it_instrument, effective_instrument, level 0..64) or None.
    holdover = state.setdefault("holdover", [None] * DMM_8_CHANNELS)
    cur_offset = [0] * DMM_8_CHANNELS      # start offset (frames) of the note now sounding on each channel
    # Volume trajectory of the note currently sounding on each channel: [(tick, level)], built from
    # volume slides and mid-note volume changes (see _resolve_note_for_dmm).
    traj = [None] * DMM_8_CHANNELS
    # Elapsed *tracker* ticks (one row = state["speed"] ticks - a different,
    # unrelated unit from the `delay` above, which is in DMM's own
    # real-time-ish delay units) since note[ch] was actually triggered.
    # This is exactly what a volume envelope is defined in terms of, and is
    # used to work out how far into its decay a held note has really got
    # by the time it's closed out below, rather than just using the flat
    # value it was triggered at for its whole (possibly much longer)
    # audible duration.
    env_ticks = [0] * DMM_8_CHANNELS
    channel_started = [False] * DMM_8_CHANNELS
    ch_has_content = [False] * DMM_8_CHANNELS
    # Whether a real (sounding) note has already been emitted on each channel in this pattern.
    # A stop span before the first real note (typically the row-0 stop every pattern starts
    # with) is leading silence: the engine starts every pattern silent anyway, so it is
    # written as a 0xFE rest - exactly how the original DMM data encodes it. A stop span
    # after a real note sounded must stay a (24,0,0) stop span: only volume 0 actually cuts
    # the voice. Without this split every leading silence came out as (24,0,0), which sounds
    # identical but breaks byte- and event-level roundtrips (dmm -> xm -> dmm).
    ch_sounded = [False] * DMM_8_CHANNELS
    # A pattern with no note at all still has a length; some channel has to carry it as a rest.
    any_content = any(
        (lambda t: t.note != 0 or _is_stop_note_for_dmm(t))(pattern.get_note(r, channel_map[c]))
        for r in range(pattern.rows) for c in range(channels))
    note = [PatternNote() for _ in range(DMM_8_CHANNELS)]
    # Absolute age (tracker ticks since trigger) of the note sounding on each
    # channel as this pattern starts: 0 for a fresh pattern, the carried age
    # for a holdover seed. env_ticks below stays pattern-relative; the two
    # together give the true envelope clock (age_base + env_ticks).
    age_base = [0] * DMM_8_CHANNELS
    for ch in range(channels):
        if holdover[ch] is not None:
            parts = holdover[ch]
            raw_note, hold_it, hold_inst, level = parts[:4]
            age = parts[4] if len(parts) > 4 else 0
            if not _holdover_audible(it_volume_envelopes, hold_it, age):
                continue  # decayed to silence: nothing to carry over
            seed = PatternNote(note=raw_note, instrument=hold_inst, it_instrument=hold_it,
                                effect=EFFECT_VOLUME, effect_parameter=int(round(level)))
            note[ch] = seed
            channel_started[ch] = True
            traj[ch] = [(0, float(level), 0.0, True)]
            age_base[ch] = age
            if envelope_ctx is not None:
                envelope_ctx["holdover_count"] = envelope_ctx.get("holdover_count", 0) + 1
            # Row 0 of THIS pattern, if it has its own trigger for this channel, will flush this
            # seed immediately below with delay 0 (silently dropped, see the `if delay[ch] > 0`
            # guards ahead) and take over normally - the seed only actually gets used when row 0
            # doesn't touch the channel, i.e. exactly the case that would otherwise go silent.
    pattern_end = False
    pattern_end_on_next_row = False
    new_note = PatternNote()

    for row_counter in range(pattern.rows + 1):
        for ch in range(channels):
            temp = pattern.get_note(row_counter, channel_map[ch]) if row_counter < pattern.rows else PatternNote()
            if temp.effect == EFFECT_SPEED_TEMPO and temp.effect_parameter > 0:
                if temp.effect_parameter < 32:
                    state["speed"] = temp.effect_parameter
                else:
                    state["tempo"] = temp.effect_parameter

        # SEx (IT) / EEx (XM/MOD) - Pattern Delay: the row is played x extra times (only its
        # duration matters for DMM). Previously ignored, which shortened the song.
        row_repeat = 0
        if row_counter < pattern.rows:
            for ch in range(channels):
                t = pattern.get_note(row_counter, channel_map[ch])
                if t.effect == EFFECT_EXTENDED_EFFECTS and (t.effect_parameter >> 4) == EXT_EFFECT_PATTERN_DELAY \
                        and (t.effect_parameter & 0xF):
                    row_repeat = t.effect_parameter & 0xF
                    break

        for ch in range(channels):
            if row_counter == pattern.rows:
                pattern_end = True
            if not pattern_end:
                new_note = _apply_channel_volume(pattern.get_note(row_counter, channel_map[ch]),
                                                 chan_volume, ch)

            # SDx (IT) - Note Delay: the note named in this row's note
            # column doesn't actually start until `delay_tick` tracker
            # ticks into the row, not at the row's own start. Until now
            # this was silently dropped (the effect byte was parsed but
            # never consulted here), so a delayed note instead triggered
            # immediately, which both plays it too early and - just as
            # importantly - cuts short whatever was already sounding on
            # this channel right at the row boundary instead of letting
            # it ring on for the ticks it was actually supposed to.
            #
            # Fix: before the normal flush/swap below (which fires on
            # new_note.note != 0) runs, extend the *currently held* note's
            # delay by the pre-delay portion of this row, so it gets
            # flushed with its real, longer duration. `row_ticks` then
            # holds only the ticks *remaining* in this row once the new
            # note actually starts, and is used further down in place of
            # the full `state["speed"]` for this channel this row only.
            delay_tick = 0
            if not pattern_end and new_note.note != 0 and channel_started[ch] \
                    and new_note.effect == EFFECT_EXTENDED_EFFECTS \
                    and (new_note.effect_parameter >> 4) == EXT_EFFECT_NOTE_DELAY:
                delay_tick = min(new_note.effect_parameter & 0xF, state["speed"])
                if delay_tick > 0:
                    # Same tick->DMM-delay-unit conversion NOTE_CUT uses
                    # below (speed cancels out of (delay_tick/speed) *
                    # (speed*165/tempo)); a plain truncation, no fractional
                    # carry, matching NOTE_CUT's existing precision.
                    pre_ticks = int(delay_tick * DMM_REF_TEMPO / state["tempo"])
                    delay[ch] += pre_ticks
                    env_ticks[ch] += delay_tick

            if channel_started[ch]:
                if new_note.note != 0 or _is_stop_note_for_dmm(new_note) or pattern_end:
                    if delay[ch] > 0:
                        if note[ch].note == 0 and not _is_stop_note_for_dmm(note[ch]):
                            # a channel that never played anything in this pattern stays empty
                            # (like in the original DMM data) instead of being padded with rests
                            if ch_has_content[ch] or not pattern_end or (not any_content and ch == 0):
                                converted[ch].append(DMMNote(254, delay[ch] // 256, 0, delay[ch] % 256))
                        elif _is_stop_note_for_dmm(note[ch]):
                            if ch_sounded[ch]:
                                _emit_long(converted[ch], 24, 0, 0, delay[ch])
                            else:
                                converted[ch].append(DMMNote(254, delay[ch] // 256, 0, delay[ch] % 256))
                        else:
                            eff_it_instr = note[ch].it_instrument or remembered_it_instrument[ch]
                            eff_mapped_sample = note[ch].instrument or remembered_instrument[ch]
                            _flush_note(converted[ch], note[ch], env_ticks[ch], delay[ch], eff_it_instr,
                                        eff_mapped_sample, state["tempo"], it_volume_envelopes,
                                        envelope_ctx, traj[ch], volume_slides, cur_offset[ch],
                                        age_base[ch])
                            ch_sounded[ch] = True
                        delay[ch] = 0
                    if holdover[ch] is not None and len(holdover[ch]) > 4:
                        holdover[ch] = holdover[ch][:4] + (age_base[ch] + env_ticks[ch],)
                    note[ch] = new_note
                    env_ticks[ch] = 0
                    if new_note.note != 0 or _is_stop_note_for_dmm(new_note):
                        ch_has_content[ch] = True
                    if not pattern_end:
                        if new_note.note != 0:
                            hold_it = new_note.it_instrument or remembered_it_instrument[ch]
                            hold_inst = new_note.instrument or remembered_instrument[ch]
                            holdover[ch] = (new_note.note, hold_it, hold_inst,
                                            _note_start_level(new_note), 0) \
                                if _sample_is_looped(envelope_ctx, hold_inst) else None
                            age_base[ch] = 0
                        elif _is_stop_note_for_dmm(new_note):
                            holdover[ch] = None
            else:
                channel_started[ch] = True
                note[ch] = new_note
                env_ticks[ch] = 0
                if new_note.note != 0 or _is_stop_note_for_dmm(new_note):
                    ch_has_content[ch] = True
                if not pattern_end:
                    if new_note.note != 0:
                        hold_it = new_note.it_instrument or remembered_it_instrument[ch]
                        hold_inst = new_note.instrument or remembered_instrument[ch]
                        holdover[ch] = (new_note.note, hold_it, hold_inst,
                                        _note_start_level(new_note), 0) \
                            if _sample_is_looped(envelope_ctx, hold_inst) else None
                        age_base[ch] = 0
                    elif _is_stop_note_for_dmm(new_note):
                        holdover[ch] = None

            if not pattern_end and new_note.effect == EFFECT_EXTENDED_EFFECTS \
                    and (new_note.effect_parameter >> 4) == EXT_EFFECT_NOTE_CUT:
                # SCx (IT) / ECx (XM/MOD) - Note Cut: silence whatever is
                # currently sounding on this channel `cut_tick` tracker
                # ticks into *this* row, well before the row (and hence
                # the note's normal delay-until-next-trigger) actually
                # ends. Without this, a note that the source module cuts
                # short mid-row is instead held all the way to the next
                # real trigger - which is exactly what turns a rhythm of
                # short, separated hits into one long sustained tone once
                # DMM (which has no other way to end a note early) plays
                # it back.
                cut_tick = new_note.effect_parameter & 0xF
                # Like Impulse Tracker, ignore a cut that would land at/after the end of the row.
                if cut_tick < state["speed"] and note[ch].note != 0 and not _is_stop_note_for_dmm(note[ch]):
                    # cut_tick out of state["speed"] ticks in this row is,
                    # in DMM's own delay unit, cut_tick * 165 / tempo -
                    # notice `speed` cancels out of (cut_tick/speed) *
                    # (speed*165/tempo).
                    cut_delay = delay[ch] + int(cut_tick * DMM_REF_TEMPO / state["tempo"])
                    if cut_delay > 0:
                        eff_it_instr = note[ch].it_instrument or remembered_it_instrument[ch]
                        eff_mapped_sample = note[ch].instrument or remembered_instrument[ch]
                        _flush_note(converted[ch], note[ch], env_ticks[ch] + cut_tick, cut_delay,
                                    eff_it_instr, eff_mapped_sample, state["tempo"], it_volume_envelopes,
                                    envelope_ctx, traj[ch], volume_slides, cur_offset[ch], age_base[ch])
                        ch_sounded[ch] = True
                    note[ch] = PatternNote(note=NOTE_STOP)
                    traj[ch] = None
                    holdover[ch] = None
                    # The end-of-row accumulation below adds the WHOLE row again, but the first
                    # cut_tick ticks of it were already spent by the note itself; pre-subtract them,
                    # otherwise every cut stretches the channel by cut_tick ticks (channels drifted
                    # apart by up to a third of a second per pattern in real IT files).
                    delay[ch] = -int(cut_tick * DMM_REF_TEMPO / state["tempo"])
                    # The remaining ticks of this same row (from the cut
                    # point through to the row's end) are silence, and get
                    # folded into note[ch]'s (now a stop marker) delay by
                    # the normal end-of-row accumulation just below, same
                    # as any other gap between notes.

            if new_note.effect == EFFECT_PATTERN_BREAK:
                pattern_end_on_next_row = True

            if not pattern_end:
                # ---- volume trajectory of the sounding note ----
                # traj points: (tracker_tick, level 0..64, dmm_time, explicit). Mid-note "set volume"
                # cells are always tracked (they become 0xFF events, or are baked in "bake" mode);
                # volume slides only when volume_slides != "off".
                is_stop = _is_stop_note_for_dmm(new_note)
                dmm_now = delay[ch] + tick_carry[ch]
                if note[ch] is new_note and new_note.note != 0 and not is_stop:
                    lvl0 = _note_start_level(new_note)          # (re)triggered on this row
                    if volume_slides != "off":
                        chan_volume[ch] = lvl0
                    traj[ch] = [(0, lvl0, 0.0, True)]
                    so = new_note.sample_offset
                    if so == "mem":
                        so = offset_mem[ch]
                    elif so is not None:
                        offset_mem[ch] = so
                    cur_offset[ch] = (so or 0) if (envelope_ctx is not None and envelope_ctx.get("offsets")) else 0
                elif note[ch] is new_note and is_stop:
                    traj[ch] = None
                    cur_offset[ch] = 0
                if new_note.note == 0 and isinstance(new_note.sample_offset, int):
                    offset_mem[ch] = new_note.sample_offset
                if new_note.env_restart and not is_stop and traj[ch] is not None:
                    traj[ch].append((env_ticks[ch], _note_start_level(new_note), dmm_now, True, "E"))
                if new_note.note == 0:
                    setlvl = None
                    if new_note.effect == EFFECT_VOLUME:
                        setlvl = float(min(new_note.effect_parameter, 64))
                    elif 0x10 <= new_note.volume <= 0x50:
                        setlvl = float(new_note.volume - 0x10)
                    if setlvl is not None:
                        chan_volume[ch] = setlvl
                        if traj[ch] is not None:
                            traj[ch].append((env_ticks[ch], setlvl, dmm_now, True))
                        if holdover[ch] is not None:
                            holdover[ch] = _holdover_set_level(holdover[ch], setlvl)
                vs = new_note.vol_slide if volume_slides != "off" else None
                if vs is not None and not is_stop:
                    if vs == "mem":
                        vs = slide_mem[ch]
                    else:
                        slide_mem[ch] = vs
                    if vs:
                        per_tick, fine = vs
                        lvl = chan_volume[ch] if chan_volume[ch] is not None else 64.0
                        t0 = env_ticks[ch]
                        tr = traj[ch]
                        if tr is not None and (tr[-1][0] != t0 or tr[-1][1] != lvl):
                            tr.append((t0, lvl, dmm_now, False))
                        if fine:
                            lvl = min(64.0, max(0.0, lvl + fine))
                            if tr is not None:
                                tr.append((t0, lvl, dmm_now, False))
                        if per_tick:
                            ratio = DMM_REF_TEMPO / state["tempo"]
                            # ST3.00 "fast volume slides" also slide on the first tick of the row
                            for k in range(0 if state.get("fast_slides") else 1,
                                           state["speed"] * (1 + row_repeat)):
                                lvl = min(64.0, max(0.0, lvl + per_tick))
                                if tr is not None:
                                    tr.append((t0 + k, lvl, dmm_now + k * ratio, False))
                        chan_volume[ch] = lvl
                        if holdover[ch] is not None:
                            holdover[ch] = _holdover_set_level(holdover[ch], lvl)
                # Note retrigger (Qxy / E9x / Rxy): the sounding note is struck again every `interval`
                # ticks of this row (tick 0 is the normal start); each strike may also change the volume.
                rt = new_note.retrig
                if rt is not None and not is_stop:
                    if rt == "mem":
                        rt = retrig_mem[ch]
                    else:
                        retrig_mem[ch] = rt
                    tr = traj[ch]
                    row_ticks_total = state["speed"] * (1 + row_repeat)
                    if rt and tr is not None and volume_slides != "bake" and 0 < rt[0] < row_ticks_total:
                        interval, vcode = rt
                        ratio = DMM_REF_TEMPO / state["tempo"]
                        lvl = chan_volume[ch] if chan_volume[ch] is not None else 64.0
                        t0 = env_ticks[ch]
                        for k in range(interval, row_ticks_total, interval):
                            lvl = _retrig_level(vcode, lvl)
                            tr.append((t0 + k, lvl, dmm_now + k * ratio, True, "R"))
                        chan_volume[ch] = lvl
                        if holdover[ch] is not None:
                            holdover[ch] = _holdover_set_level(holdover[ch], lvl)
            if not pattern_end:
                if new_note.instrument != 0:
                    remembered_instrument[ch] = new_note.instrument
                    remembered_it_instrument[ch] = new_note.it_instrument
                row_ticks = state["speed"] * (1 + row_repeat) - delay_tick
                exact_ticks = row_ticks * DMM_REF_TEMPO / state["tempo"] + tick_carry[ch]
                inc = int(exact_ticks)  # exact_ticks is always >= 0, so this is a plain floor
                tick_carry[ch] = exact_ticks - inc
                delay[ch] += inc
                env_ticks[ch] += row_ticks
                if delay[ch] > 65535 - 256:
                    # A DMM rest is inst*256+delay (16 bit): flush before it could overflow.
                    if note[ch].note != 0 or _is_stop_note_for_dmm(note[ch]):
                        if _is_stop_note_for_dmm(note[ch]):
                            _emit_long(converted[ch], 24, 0, 0, 256)
                        else:
                            eff_it_instr = note[ch].it_instrument or remembered_it_instrument[ch]
                            eff_mapped_sample = note[ch].instrument or remembered_instrument[ch]
                            inst_val, vol = _resolve_note_for_dmm(note[ch], env_ticks[ch], it_volume_envelopes,
                                                                   eff_it_instr, eff_mapped_sample,
                                                                   state["tempo"], envelope_ctx, traj[ch] if volume_slides == "bake" else None,
                                                                   age_base[ch])
                            _emit_long(converted[ch], _dmm_note_number(note[ch].note), inst_val, vol, 256)
                            ch_sounded[ch] = True
                        delay[ch] -= 256
                        note[ch] = PatternNote()
                        traj[ch] = None
                    converted[ch].append(DMMNote(254, delay[ch] // 256, 0, delay[ch] % 256))
                    delay[ch] = 0
        if pattern_end:
            break
        if pattern_end_on_next_row:
            pattern_end = True

    dmm_notes = []
    for ch in range(DMM_8_CHANNELS):
        dmm_notes.extend(converted[ch])
        dmm_notes.append(DMMNote(255, 0, 128, 0))
    return dmm_notes


def _make_dmm_sample_name(s):
    allowed = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_")
    result = ""
    for c in s:
        if c in allowed:
            result += c
            if len(result) >= 6:
                break
    return result


def _base36_2(n):
    """Encodes 0..1295 as exactly 2 base-36 characters (0-9A-Z). Used as the
    trailing part of a generated 8-char DMM/WAD instrument name: with a
    6-char (or shorter) readable prefix, this guarantees a name that is at
    most 8 characters long *in total* and unique per instrument slot in the
    source module (up to 1296 slots - comfortably above DMM's own 255-
    instrument ceiling, and every real XM/S3M/MOD format's own limits).

    This matters because the real Doom 2D engine (and this project's own
    WAD viewer/player, via Wad.find8) only ever compares the *first 8
    characters* of a resource name when looking it up in a WAD - anything
    from the 9th character on is silently ignored. The DMM format itself
    stores up to 15 characters per instrument name, so it's easy to end up
    with two on-disk-valid, distinct-looking DMM instrument names (e.g. two
    9-character names that only differ in their 9th character) that are
    *indistinguishable* to the actual lookup the game performs, causing a
    WAD-import collision even though the .dmm file "looks" fine. Keeping
    every generated name at <=8 characters total avoids this class of bug
    entirely, rather than relying on the source module's own sample names
    happening to already be short/unique enough within their first 8 bytes."""
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    n = max(0, min(int(n), 35 * 36 + 35))
    return digits[n // 36] + digits[n % 36]


DMI_MAX_LENGTH = 0xFFFF  # the .DMI format's own sample-length field is a plain unsigned 16-bit word


UNROLL_PINGPONG = True


def _mixdown_stereo_samples(xm):
    """IT/S3M samples can be stereo; loaders keep them interleaved (L R L R ...). The DMI writer
    and the envelope baker both index the data as mono frames, which produced garbage (half the
    frames, L/R alternating). Convert every stereo sample to mono once, up front."""
    for inst in xm.instruments.values():
        for smp in inst.sample.values():
            if smp is not None and getattr(smp, "channels", 1) == 2 and smp.sample_length \
                    and len(smp.data) >= 2 * smp.sample_length:
                d = smp.data
                smp.data = [(d[2 * i] + d[2 * i + 1]) // 2 for i in range(smp.sample_length)]
                smp.channels = 1


def _build_dmi_bytes(sample, log=None):
    length = sample.sample_length
    samplerate = xm_to_sample_rate(sample.relative_note, sample.finetune)
    data = sample.data[:length]
    loop_start, loop_length = (sample.loop_start, sample.loop_length) if sample.loop else (0, 0)

    # DMI loops only run forward. A ping-pong loop is unrolled into a forward loop that contains
    # the forward pass followed by the reversed pass (turn-around samples not repeated), so it
    # keeps sounding like a ping-pong. Data after the loop end is unreachable anyway.
    if (UNROLL_PINGPONG and sample.loop and getattr(sample, "pingpong_loop", False)
            and loop_length > 2 and loop_start < length):
        loop_end = min(loop_start + loop_length, length)
        loop_length = loop_end - loop_start
        if loop_length > 2 and loop_end + loop_length - 2 <= DMI_MAX_LENGTH:
            data = list(data[:loop_end]) + list(data[loop_end - 2:loop_start:-1])
            length = len(data)
            loop_length = 2 * loop_length - 2
        elif loop_length > 2 and log is not None:
            # Unrolling would not fit into a DMI; downsampling the whole sample to make it fit
            # would hurt more than a forward-only loop does, so keep the forward loop.
            log(f"Warning: sample \"{sample.name or '(unnamed)'}\" has a ping-pong loop too long to "
                f"unroll into a DMI; it will loop forward only.\n")

    # Defensive fallback for samples not used by a generated DMM note. Used samples have already
    # been jointly fitted to the engine-rate and DMI-size limits before this writer is reached.
    # The DMI rate field is a 16-bit word as well: C5Speeds like 88200 used to wrap around
    # (& 0xFFFF) and play the sample two octaves too low. Same remedy as for a too-long sample.
    factor = max(-(-length // DMI_MAX_LENGTH), -(-samplerate // 65535), 1)
    if factor > 1:
        # A sample this long simply can't be represented in a .DMI file -
        # its own length field is a plain 16-bit word, the same hard limit
        # the original DOS-era format/engine always had. Real IT/XM/S3M
        # modules can and do carry samples far longer than that (a whole
        # jingle or long one-shot embedded as a single "instrument" isn't
        # unusual in demoscene-style modules).
        #
        # Simply chopping the sample off at DMI_MAX_LENGTH would silently
        # throw away everything past that point - for a sample more than
        # 2x over the limit, over half the sound. Instead this box-averages
        # every `factor` input frames into one output frame (a simple
        # anti-aliasing low-pass, not just dropping samples) and scales the
        # stored playback rate down by that same factor, so the shrunk
        # sample still plays back at the original pitch and takes the same
        # real-world time - i.e. the whole thing stays audible, just at
        # lower fidelity, rather than getting cut off partway through.
        new_length = -(-length // factor)      # ceil(length / factor), always <= DMI_MAX_LENGTH
        new_samplerate = max(1, round(samplerate / factor))
        if log is not None:
            name = sample.name or "(unnamed)"
            log(f"Warning: sample \"{name}\" ({length} samples, {samplerate}Hz) exceeds the "
                f".DMI format's {DMI_MAX_LENGTH}-sample / 65535Hz limits; downsampling {factor}x "
                f"({samplerate}Hz -> {new_samplerate}Hz) to keep the whole sample rather "
                f"than truncating it.\n")
        if _np is not None and length > 0 and len(data) >= length:
            # Exact integer fast path (sums can't round: unbounded integer
            # addition is order-independent, // is floor on both sides).
            # Short data falls through to the slow path to keep even its
            # failure mode identical.
            _arr = _np.asarray(data[:length], dtype=_np.int64)
            _starts = _np.arange(0, length, factor)
            _sums = _np.add.reduceat(_arr, _starts)
            _counts = _np.diff(_np.append(_starts, length))
            data = (_sums // _counts).tolist()
        else:
            data = [sum(data[i:i + factor]) // len(data[i:i + factor])
                    for i in range(0, length, factor)]
        loop_start //= factor
        loop_length //= factor
        length = new_length
        samplerate = new_samplerate
    if loop_start >= length:
        loop_start, loop_length = 0, 0
    else:
        loop_length = min(loop_length, length - loop_start)
    # sample.data holds 8-bit values already scaled by *256 (see decode_8bit); DMI
    # instrument data is plain signed 8-bit, so undo that scaling here.
    # Round (not truncate toward zero) when reducing 16-bit data; exact for 8-bit data.
    if _np is not None:
        # Integer-only, hence exact: arithmetic shift is floor like Python's
        # >>, clip matches max/min, & 0xFF takes the same low byte.
        _raw = ((_np.asarray(data, dtype=_np.int64) + 128) >> 8)
        _raw = _np.clip(_raw, -128, 127) & 0xFF
        raw = _raw.astype(_np.uint8).tobytes()
    else:
        raw = bytes((max(-128, min(127, (int(v) + 128) >> 8)) & 0xFF) for v in data)
    return struct.pack("<4H", length, samplerate & 0xFFFF, loop_start & 0xFFFF, loop_length & 0xFFFF) + raw


def _save_dmi(sample, path):
    ensure_dir_for_file(path)
    with open(path, "wb") as f:
        f.write(_build_dmi_bytes(sample))


def _pick_instrument_name(output_dir, base, preferred_index, dmi_bytes):
    """Return an <=8-char DMM/WAD instrument name (`base` + 2-char base36
    suffix) for the sample whose DMI bytes are `dmi_bytes`.

    The suffix normally just encodes this instrument's slot position in the
    *current* module (`preferred_index`), which is enough to keep names
    unique within a single conversion. It is NOT enough across several
    files converted independently (one run each) into the same output
    folder so they can later be packed into one WAD: two unrelated tracks
    can easily both have an unnamed instrument at the same slot number,
    both falling back to the same "DMIU.." name, and the second run would
    silently overwrite the first run's .dmi file on disk even though the
    two samples are different.

    To avoid that without needing a batch/multi-file mode or a separate
    session-state file, this treats the output directory itself as the
    source of truth: before finalising a name it checks whether a file
    with that name already exists there.
      - If no file exists yet -> use this name.
      - If a file exists and its bytes are identical -> reuse this name
        (this is the normal case when re-converting the *same* source file
        into the same folder again; keeps names stable/deterministic run
        to run).
      - If a file exists with *different* bytes -> it belongs to some other,
        previously converted track sharing this output folder. Move on to
        the next free base36 suffix (0..1295) instead, so the earlier
        file is never overwritten and the two instruments end up with
        genuinely distinct names.
    """
    tried = set()
    candidates = [preferred_index] + [i for i in range(36 * 36) if i != preferred_index]
    for idx in candidates:
        name = base + _base36_2(idx)
        if name in tried:
            continue
        tried.add(name)
        path = os.path.join(output_dir, name)
        if not os.path.isfile(path):
            return name
        with open(path, "rb") as f:
            existing = f.read()
        if existing == dmi_bytes:
            return name
    # All 1296 possible suffixes for this base are already taken by other,
    # different samples - astronomically unlikely in practice. Fall back to
    # the natural slot-based name rather than raising, matching old behaviour.
    return base + _base36_2(preferred_index)


def _convert_split_subsongs(xm, output_file, log, channel_selection, volume_slides,
                            sample_offsets):
    """Split converted driver: one DMM per detected subsong, DMIs shared.

    Returns a single path when there is just one song covering the whole
    order table with nothing to truncate (plain output name, same as
    split_subsongs=False), otherwise a list of "<base>_subN.dmm" paths.
    """
    full_channels = list(range(xm.number_of_channels))
    songs = _detect_subsongs(xm, full_channels)
    base = cut_extension(output_file)
    n = xm.track_length
    trunc_map = _subsong_truncations(xm, full_channels)

    if len(songs) == 1 and songs[0]["orders"] == list(range(n)) and not trunc_map:
        log("Subsong split: a single song covering the whole order table - one DMM as usual.\n")
        output_path, _dmi_names = _convert_loaded_module(
            xm, output_file, log, channel_selection, volume_slides, sample_offsets)
        return output_path

    results, shared_names = [], set()
    for i, song in enumerate(songs):
        view, _seq = _build_subsong_view(xm, song["orders"], trunc_map)
        view.restart_position = song["restart"]
        if len(songs) == 1:
            out = output_file
        else:
            out = f"{base}_sub{i + 1}.dmm"
        if song["loop_to"] is None:
            loop_info = "ends (DMM loops it from its own start)"
        else:
            loop_info = (f"loops to XM order {song['loop_to']} "
                         f"(pattern {xm.pattern_order[song['loop_to']]})")
        log(f"--- Subsong {i + 1}/{len(songs)}: from XM order {song['start']}, "
            f"{len(song['orders'])} orders, {view.number_of_patterns()} patterns, "
            f"{loop_info}.\n")
        output_path, dmi_names = _convert_loaded_module(
            view, out, log, channel_selection, volume_slides, sample_offsets)
        results.append(output_path)
        shared_names.update(dmi_names)
    if len(results) > 1:
        log(f"Subsong split done: {len(results)} DMM files sharing "
            f"{len(shared_names)} distinct DMI instruments in "
            f"\"{os.path.dirname(os.path.abspath(results[0])) or '.'}\" "
            f"(identical samples reuse one file - pack the union into one WAD).\n")
    return results[0] if len(results) == 1 else results


def convert_backward(input_file, output_file="", log=_default_log,
                      channel_selection="first", smart_channels=None, volume_slides="off",
                      bake_volume_slides=None, sample_offsets=False, split_subsongs=False):
    """XM, S3M, IT or MOD -> DMM (+ .DMI instrument files next to the output).
    Returns the output path, or a list of paths when split_subsongs yields
    several files.

    split_subsongs: off by default (one DMM with the full order table, speeds
    resolved along real playback paths - see _compute_pattern_entry_states).
    When True, position jumps (XM/MOD Bxx) and breaks (Cxx) split the module
    into its real playback songs, and each song becomes its own DMM
    ("<base>_sub1.dmm", "<base>_sub2.dmm", ...) with play order already
    expanded, mid-pattern jumps truncated, and speeds flowing naturally -
    no cross-subsong speed alignment is needed at all. Instrument samples
    are shared automatically: identical DMI bytes reuse the same file in
    the output folder (see _pick_instrument_name), so packing all parts
    plus the union of their DMIs into one WAD costs no duplicates. With a
    single song covering the whole table the output is one plain-named DMM
    exactly like split_subsongs=False would produce (minus jump cells).

    volume_slides: how to reproduce volume changes while a note is held. A DMM note has one
    volume, but the engine also has 0xFF "volume only" events that change the volume of the
    sounding note at any tick (SOUND.ASM; the game's own СОЛО song uses them).
      "off"     (default) volume slides (MOD/XM Axy, S3M/IT Dxy, ...) are ignored and IT volume
                envelopes are baked into extra instruments - the old behaviour.
      "events"  slides AND IT envelopes become 0xFF events inside the pattern: no extra
                instruments/.DMI files at all (a module with many enveloped IT instruments can
                need several times fewer instruments and DMI bytes than with "off"), +4 bytes
                per volume change (the format allows 65535 notes per song). Recommended.
      "bake"    slides and envelopes are rendered into extra one-off instruments: exact for
                complex shapes, but every such note adds a .DMI file (see SHAPED_* limits).
    Mid-note "set volume" cells (Cxx on an empty row, volume column) always become 0xFF events.
    bake_volume_slides: old boolean alias (True == "bake").

    sample_offsets: off by default. When True, a note with a sample offset (S3M/IT Oxx, MOD/XM 9xx)
    plays a copy of its instrument that starts at that offset (DMM cannot start a sample in the
    middle). One copy per distinct (instrument, offset), at most OFFSET_MAX_VARIANTS per song, so it
    costs extra instruments / WAD lumps; beyond the limit the offset is ignored.

    channel_selection: only matters when the source module has more than 8
    channels (DMM's hard limit). One of:
      "first"    (default) keeps the original behaviour of always using
                 channels 1-8 and dropping the rest.
      "count"    picks whichever 8 channels carry the most note events over
                 the real playback order, so channels that are silent or
                 used for engine-specific decoration (visualizer-only,
                 muted stems, etc.) don't take a slot away from a channel a
                 listener would actually miss - a real S3M was found where
                 an instrument used only on channel 12 of 13 was always
                 dropped even though 2 of the first 8 channels sat silent
                 for the whole song.
      "loudness" picks whichever 8 channels contribute the most volume-
                 weighted "loudness x how often it's heard" over the real
                 playback order. Unlike "count", this won't drop a channel
                 that only plays a handful of loud, prominent lead/bass
                 notes per bar in favour of a channel full of quiet filler/
                 arpeggio notes that simply happens to have more note
                 events - on a real IT file, "count" dropped exactly the
                 instrument a listener would call the loudest thing in the
                 mix, while "loudness" kept it.
    See _select_busiest_channels() for exactly how "count" and "loudness"
    each rank channels.

    smart_channels: deprecated boolean alias, kept for callers written
    against the older two-way choice - True means "loudness", False means
    "first". Only takes effect when channel_selection is left at its
    default; pass channel_selection explicitly to pick "count" or to
    override an smart_channels default.
    """
    if bake_volume_slides and volume_slides == "off":
        volume_slides = "bake"
    if volume_slides not in ("off", "events", "bake"):
        raise ConversionError(f'Unknown volume_slides "{volume_slides}" (expected "off", "events" or "bake")')
    if smart_channels is not None and channel_selection == "first":
        channel_selection = "loudness" if smart_channels else "first"
    if channel_selection not in ("first", "count", "loudness"):
        raise ConversionError(
            f'Unknown channel_selection "{channel_selection}" (expected '
            f'"first", "count", or "loudness").')
    if not os.path.isfile(input_file):
        raise ConversionError(f'File "{input_file}" not found!')
    ext = get_extension(input_file)
    if ext not in ("xm", "mod", "s3m", "it"):
        raise ConversionError(
            'Only .xm, .s3m, .it and .mod files are supported as input for backward conversion.')

    log(f':: Reading module: "{input_file}"...\n')
    xm = load_module(input_file)
    if xm is None:
        if ext == "xm":
            raise ConversionError(
                f'Couldn\'t load "{input_file}"! (must be a "modern" XM file, format version > 0x0103)')
        if ext == "s3m":
            raise ConversionError(
                f'Couldn\'t load "{input_file}"! (unrecognised or unsupported/corrupt S3M variant, '
                f'or an Adlib-only module with no channels)')
        if ext == "it":
            raise ConversionError(
                f'Couldn\'t load "{input_file}"! (unrecognised or corrupt IT file)')
        raise ConversionError(
            f'Couldn\'t load "{input_file}"! (unrecognised or unsupported/corrupt MOD variant)')

    _mixdown_stereo_samples(xm)
    if not output_file:
        output_file = cut_extension(input_file) + ".dmm"
    if split_subsongs:
        return _convert_split_subsongs(xm, output_file, log, channel_selection,
                                       volume_slides, sample_offsets)
    output_path, _dmi_names = _convert_loaded_module(
        xm, output_file, log, channel_selection, volume_slides, sample_offsets)
    return output_path


def _convert_loaded_module(xm, output_file, log, channel_selection, volume_slides,
                           sample_offsets):
    """Converts one already-loaded module object to DMM + DMI files.

    The module's own order table is converted as-is (for subsong splits the
    caller passes a pre-built song view whose order is already real play
    order). Returns (output path, [DMI instrument names referenced])."""
    output_dir = os.path.dirname(os.path.abspath(output_file)) or "."

    log(":: Converting module to DMM...\n")
    dmm_pattern_number = xm.number_of_patterns()
    if xm.number_of_channels > 8:
        if channel_selection in ("count", "loudness"):
            channel_map = _select_busiest_channels(xm, 8, by=channel_selection)
            metric_label = "note count" if channel_selection == "count" else "volume-weighted usage"
            log(f"Warning: Too many channels in module ({xm.number_of_channels}); "
                f"using the 8 most prominent channels by {metric_label} (1-based): "
                f"{[c + 1 for c in channel_map]}.\n")
        else:
            channel_map = list(range(8))
            log("Warning: Too many channels in module, only first 8 will be used.\n")
        xm.number_of_channels = 8
    else:
        channel_map = list(range(xm.number_of_channels))

    # NOTE: row-0 stop notes (C00 / volume-column 0 / note-off) are
    # deliberately NOT cleared here. The DMM engine silences every voice at
    # a pattern start anyway, so a leading stop is sonically a no-op - with
    # one vital exception: it must kill a holdover note carried over from
    # the previous pattern (see `holdover` in _pattern_to_dmm_notes). A real
    # case: BGM16.XM pattern 26 channel 3 starts with C00 while pattern 25
    # still rings there; clearing that stop (as this used to do) lets the
    # holdover seed drone through the whole pattern and even poisons the
    # entry-state walk past it. Keeping the stop converts to a harmless
    # leading silence and terminates the holdover chain correctly.

    envelope_ctx = {"xm": xm, "envelopes": xm.it_volume_envelopes, "cache": {}, "extra_count": 0,
                    "offsets": bool(sample_offsets)}

    slot_entries, slot_pattern, remapped_order, total_slots = _compute_pattern_entry_states(
        xm, dmm_pattern_number, log, channel_map, envelope_ctx, volume_slides)
    dmm_pattern_number = total_slots
    all_dmm_notes = []
    for slot in range(total_slots):
        all_dmm_notes.extend(
            _pattern_to_dmm_notes(xm.patterns[slot_pattern[slot]], xm.number_of_channels,
                                   slot_entries[slot], channel_map, xm.it_volume_envelopes, envelope_ctx,
                                   volume_slides))
    dmm_note_number = len(all_dmm_notes)
    if envelope_ctx.get("offset_count") or envelope_ctx.get("offset_skipped"):
        msg = f"Sample offsets (Oxx/9xx): {envelope_ctx.get('offset_count', 0)} instrument copy(ies) with a shifted start created."
        if envelope_ctx.get("offset_skipped"):
            msg += (f" {envelope_ctx['offset_skipped']} more distinct offset(s) ignored (limit "
                    f"{OFFSET_MAX_VARIANTS} copies per song) - those notes play from the start of the sample.")
        log(msg + "\n")
    _fit_dmm_instruments_to_engine(xm, all_dmm_notes, log)

    dmm_pattern_order_size = xm.track_length + 2
    dmm_pattern_order = list(remapped_order) + [255, xm.restart_position & 0xFF]

    # Format limits of the game's loader: the note count is a 16-bit word, the order list
    # (incl. its 255 marker and restart byte) and the pattern/instrument counts are single bytes.
    if dmm_note_number > 65535:
        raise ConversionError(
            f"The song needs {dmm_note_number} DMM notes but the format allows at most 65535"
            + (" - try volume_slides=\"off\" (0xFF volume events add notes)." if volume_slides == "events" else "."))
    if dmm_pattern_order_size > 255:
        raise ConversionError(f"The order list has {dmm_pattern_order_size - 2} entries; DMM allows at most 253.")
    if dmm_pattern_number > 255:
        raise ConversionError(f"{dmm_pattern_number} patterns; DMM allows at most 255.")
    if xm.number_of_instruments() > 255 or max(xm.instruments.keys(), default=0) > 255:
        # NOTE: number_of_instruments() itself only scans slots 1..255, so a module with 256+
        # instruments would look like exactly 255 to it while silently dropping every higher slot
        # (and any note pointing there would wrap to instrument 0 via & 0xFF) - hence the extra
        # max-key check, which sees the real table.
        raise ConversionError(
            f"{max(xm.number_of_instruments(), max(xm.instruments.keys(), default=0))} instruments "
            f"(counting multisample splits, baked volume shapes and sample-offset copies); DMM "
            f"allows at most 255 - use fewer instruments, volume_slides=\"events\" instead of "
            f"\"bake\", or disable sample_offsets.")

    dmm_instrument_number = xm.number_of_instruments()
    inst_rate = {}
    distinct_dmi = set()
    dmm_instruments = []
    for counter in range(dmm_instrument_number):
        inst = xm.instruments.get(counter + 1)
        name = ""
        if inst is not None and inst.sample.get(0) is not None:
            smp = inst.sample[0]
            if smp.sample_length != 0:
                # Base-36 slot suffix (see _base36_2) instead of a plain
                # 2-digit decimal counter: guarantees the *whole* generated
                # name is <=8 characters (matching the real WAD-lookup
                # limit exactly, so nothing gets silently truncated later)
                # and unique across every instrument slot in this module,
                # even when several source instruments share the same
                # sanitised 6-character name prefix (or have no name at
                # all). _pick_instrument_name() additionally checks the
                # output folder itself for name collisions with *other*
                # already-converted files (see its docstring), so this
                # stays deterministic for a single file and safe when
                # several files are converted one-by-one into the same
                # folder to be packed into one WAD together.
                base = _make_dmm_sample_name(smp.name)
                if not base:
                    base = _make_dmm_sample_name(inst.name)
                if not base:
                    base = "DMIU"
                dmi_bytes = _build_dmi_bytes(smp, log=log)
                inst_rate[counter + 1] = struct.unpack_from("<H", dmi_bytes, 2)[0]
                distinct_dmi.add(dmi_bytes)
                name = _pick_instrument_name(output_dir, base, counter + 1, dmi_bytes)
                ensure_dir_for_file(os.path.join(output_dir, name))
                with open(os.path.join(output_dir, name), "wb") as f:
                    f.write(dmi_bytes)
        dmm_instruments.append(name)

    _report_engine_limits(all_dmm_notes, inst_rate, log)
    if envelope_ctx.get("holdover_count"):
        log(f"Pattern-boundary continuity: re-triggered {envelope_ctx['holdover_count']} note(s) that "
            f"were still sounding across a pattern boundary (the engine silences the sounding voice "
            f"whenever a new pattern starts unless something retriggers it there - see README).\n")
    lumps = len(distinct_dmi) + 1
    free = WAD_MAX_LUMPS - ORIGINAL_WAD_LUMPS
    log(f"WAD budget: this song needs {lumps} lumps (1 DMM + {len(distinct_dmi)} distinct DMI; DMM2WAD merges "
        f"identical DMIs). The game has {WAD_MAX_LUMPS} lump slots for all WADs together, the original "
        f"doom2d.wad uses {ORIGINAL_WAD_LUMPS} -> {free} free"
        f"{' - THIS SONG ALONE EXCEEDS THEM' if lumps > free else ''}. Pack with DMM2WAD start_number >= "
        f"{ORIGINAL_WAD_LAST_DMI + 1}, otherwise the new DMI0001.. override the original game's instruments.\n")

    ensure_dir_for_file(output_file)
    with open(output_file, "wb") as f:
        f.write(b"DMM\x00\x00")
        f.write(bytes([dmm_pattern_number & 0xFF]))
        f.write(struct.pack("<H", dmm_note_number))
        for n in all_dmm_notes:
            f.write(bytes([n.note & 0xFF, n.instrument & 0xFF, n.volume & 0xFF, n.delay & 0xFF]))
        f.write(bytes([dmm_pattern_order_size & 0xFF]))
        f.write(bytes([b & 0xFF for b in dmm_pattern_order]))
        f.write(bytes([dmm_instrument_number & 0xFF]))
        for name in dmm_instruments:
            nb = name.encode("ascii", errors="replace")[:15]
            nb = nb + b"\x00" * (15 - len(nb))
            f.write(bytes([0]) + nb)

    log(f':: Saved DMM to "{output_file}"\n')
    return output_file, [name for name in dmm_instruments if name]
