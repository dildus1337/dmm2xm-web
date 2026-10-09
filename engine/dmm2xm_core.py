"""
dmm2xm_core.py
==============

Python port of "dmm2xm" (Doom2D DMM <-> FastTracker XM converter) by Grom PE
& Artem Vasilev, originally written in Object Pascal (dmm2xm.dpr, BeRoXM.pas,
WadFile.pas, streams.pas, helpers.inc).

This module re-implements the conversion algorithm (not just calls the old
.exe): reading DOOM2D.WAD/PWAD-IWAD lumps, reading DMM songs and DMI
instruments, and writing FastTracker II .XM modules (forward conversion),
as well as reading .XM modules and writing them back to .DMM + .DMI files
(backward conversion).

Notes on fidelity / known limitations of this port, compared to the
original Windows-only dmm2xm.exe v1.5.1:

  * Backward conversion only accepts .xm input (not legacy .mod - the
    original tool's amiga-MOD loader is not ported here).
  * Only "modern" XM files (format version > 0x0103, i.e. the kind that
    FastTracker II, OpenMPT, MilkyTracker etc. actually write) are
    supported for backward conversion.
  * ADPCM4-compressed XM samples are not decoded (very rare, not used by
    dmm2xm itself).

The forward direction (DMM/WAD -> XM), which is what conv_wad.bat and the
main advertised use-case of the tool need, is implemented to match the
original algorithm closely, including the same quantization/note-packing
logic, the same per-instrument stereo panning table, and the same
hand-written per-sample loop-point bugfixes for the stock Doom2D music.
"""

import os
import re
import copy
import math
import random
import struct
import types

VERSION_TEXT = "dmm2xm v1.5.1 [13 Apr 2023] by Grom PE & Artem Vasilev (wormsbiysk)"

DMM_8_CHANNELS = 8
MAX_SAMPLES = 11

NOTE_STOP = 97

EFFECT_ARPEGGIO = 0x00
EFFECT_VOLUME = 0x0C
EFFECT_PATTERN_BREAK = 0x0D
EFFECT_EXTENDED_EFFECTS = 0x0E
EFFECT_SPEED_TEMPO = 0x0F
EFFECT_EXTRA_FINE_PORTA = 0x21

EXT_EFFECT_NOTE_CUT = 0xC
EXT_EFFECT_PATTERN_DELAY = 0xE

# DMM ticks <-> XM/IT/S3M tempo. DMM runs at 66 ticks/s and a tracker tick lasts 2.5/tempo s,
# so tempo 165 == exactly 66 Hz (2.5 * 66). Forward conversion writes an XM with tempo
# DMM_REF_TEMPO and speed == quantization (one row = `quantization` DMM ticks), backward
# conversion converts tracker ticks to DMM ticks as  ticks * DMM_REF_TEMPO / tempo. Both
# directions MUST use this same constant, otherwise dmm -> xm -> dmm drifts.
# Measured against real modules: 165 keeps converted songs within ~0.1 s of their true length;
# 166 made them ~0.6 % (about 1.6 s per 4.5 minutes) too long.
# NOTE (original source, SOUND.ASM): the DOS engine advances the music every (sfreq >> 6) output
# samples, so its tick rate is sfreq/(sfreq>>6) = 64.10 Hz at 11025 Hz (64.0-64.5 over the
# selectable rates), not 66. A song converted with 165 therefore plays ~3 % slower in the original
# game than in a 66 Hz player (4:38 -> ~4:46). 160.25 (= 2.5 * 64.10) would be the value for the
# original engine; keep 165 if the target player really runs DMM at 66 ticks/s.
DMM_REF_TEMPO = 165
EXT_EFFECT_NOTE_DELAY = 0xD

# PAN_SETTING[total_instruments-1][instrument_index] (0-based instrument index)
PAN_SETTING = [
    [128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [120, 136, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [112, 128, 144, 0, 0, 0, 0, 0, 0, 0, 0],
    [102, 120, 136, 152, 0, 0, 0, 0, 0, 0, 0],
    [96, 112, 128, 144, 160, 0, 0, 0, 0, 0, 0],
    [88, 102, 120, 136, 152, 168, 0, 0, 0, 0, 0],
    [80, 96, 112, 128, 144, 160, 176, 0, 0, 0, 0],
    [72, 88, 102, 120, 136, 152, 168, 184, 0, 0, 0],
    [64, 80, 96, 112, 128, 144, 160, 176, 192, 0, 0],
    [56, 72, 88, 102, 120, 136, 152, 168, 184, 200, 0],
    [48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208],
]

# Hand-picked fixes for known Doom2D instruments with broken/odd loop points
# in the original DMI data. name -> (check_field, check_value, fixes dict)
SAMPLE_FIXES = [
    ("DMI0011", "loop_start", 3326, {"pingpong": True, "loop_length": 5375}),
    ("DMI0063", "loop_start", 720, {"loop_length": 6982}),
    ("DMI0018", "loop_length", 14868, {"pingpong": True, "loop_start": 2168, "loop_length": 8338}),
    ("DMI0055", "loop_start", 3394, {"pingpong": True, "loop_start": 1897, "loop_length": 4377}),
    ("DMI0059", "loop_length", 14864, {"pingpong": True, "loop_start": 2168, "loop_length": 8595}),
    ("DMI0040", "loop_start", 4010, {"pingpong": True}),
    ("DMI0004", "loop_start", 14856, {"loop_length": 7712}),
    ("DMI0071", "loop_start", 24, {"pingpong": True, "loop_start": 428, "loop_length": 7525}),
    ("DMI0022", "loop_start", 4250, {"pingpong": True, "loop_start": 905, "loop_length": 4606}),
    ("DMI0050", "loop_length", 20000, {"pingpong": True, "loop_length": 20946}),
    ("DMI0074", "loop_start", 6614, {"pingpong": True, "loop_start": 6339, "loop_length": 738}),
    ("DMI0045", "loop_start", 1000, {"pingpong": True, "loop_start": 80, "loop_length": 9272}),
]


# ---------------------------------------------------------------------------
# small helpers replicating Pascal integer semantics
# ---------------------------------------------------------------------------

def trunc_div(a, b):
    """Pascal 'div' truncates toward zero (unlike Python's floor //)."""
    q = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        q = -q
    return q


def trunc_mod(a, b):
    return a - trunc_div(a, b) * b


def wrap_s8(v):
    return ((v + 128) % 256) - 128


def wrap_s16(v):
    return ((v + 32768) % 65536) - 32768


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def cut_extension(filename):
    dot = filename.rfind(".")
    sep = max(filename.rfind("/"), filename.rfind("\\"))
    if dot == -1 or dot < sep:
        return filename
    return filename[:dot]


def get_extension(filename):
    dot = filename.rfind(".")
    sep = max(filename.rfind("/"), filename.rfind("\\"))
    if dot == -1 or dot < sep:
        return ""
    return filename[dot + 1:].lower()


def ensure_dir_for_file(path):
    """Creates the parent directory of `path` if it doesn't exist yet
    (like `mkdir -p`), so writing to a since-deleted/not-yet-created output
    folder doesn't fail with FileNotFoundError."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)


def oem_encode(name):
    """Encode a (possibly Cyrillic) name the way Windows ANSI->OEM conversion
    would for looking it up inside a WAD file (Doom2D WAD lump names are
    stored in DOS codepage 866)."""
    try:
        return name.encode("cp866")
    except UnicodeEncodeError:
        return name.encode("ascii", errors="ignore")


def oem_decode(raw_bytes):
    try:
        return raw_bytes.decode("cp866")
    except UnicodeDecodeError:
        return raw_bytes.decode("latin1")


# ---------------------------------------------------------------------------
# WAD reading
# ---------------------------------------------------------------------------

class WadFile:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            self.data = f.read()
        self.valid = False
        self.records = []  # list of (offset, size, name_bytes)
        if len(self.data) >= 12 and self.data[0:1] in (b"I", b"P") and self.data[1:4] == b"WAD":
            num_lumps, table_ofs = struct.unpack_from("<II", self.data, 4)
            ok = True
            recs = []
            off = table_ofs
            try:
                for _ in range(num_lumps):
                    offset, size, name = struct.unpack_from("<II8s", self.data, off)
                    recs.append((offset, size, name))
                    off += 16
            except struct.error:
                ok = False
            if ok:
                self.valid = True
                self.records = recs

    def get_file_index(self, name_str):
        target = oem_encode(name_str)
        for i, (_offset, _size, name) in enumerate(self.records):
            record_name = name.split(b"\x00", 1)[0]
            if record_name == target:
                return i
        return -1

    def read_at(self, offset, size):
        return self.data[offset:offset + size]


def list_dmm_in_wad(path):
    """Return list of (display_name) of valid DMM lumps found in a WAD."""
    wad = WadFile(path)
    if not wad.valid:
        return None
    names = []
    for offset, size, name in wad.records:
        if size < 80:
            continue
        header = wad.read_at(offset, 5)
        if len(header) == 5 and header[0:3] == b"DMM" and header[3] == 0 and header[4] == 0:
            raw_name = name.split(b"\x00", 1)[0]
            names.append(oem_decode(raw_name))
    return names


# ---------------------------------------------------------------------------
# DMM reading / writing
# ---------------------------------------------------------------------------

class DMMNote:
    __slots__ = ("note", "instrument", "volume", "delay")

    def __init__(self, note=0, instrument=0, volume=0, delay=0):
        self.note = note
        self.instrument = instrument
        self.volume = volume
        self.delay = delay


def is_finish_note(n):
    return n.note == 255 and n.instrument == 0 and n.volume == 128 and n.delay == 0


def parse_dmm_bytes(data):
    if len(data) < 5 or data[0:3] != b"DMM" or data[3] != 0 or data[4] != 0:
        return None
    pos = 5
    try:
        pattern_number = data[pos]; pos += 1
        note_number = struct.unpack_from("<H", data, pos)[0]; pos += 2
        notes = []
        for _ in range(note_number):
            note, instrument, volume, delay = data[pos], data[pos + 1], data[pos + 2], data[pos + 3]
            notes.append(DMMNote(note, instrument, volume, delay))
            pos += 4
        pattern_order_size = data[pos]; pos += 1
        pattern_order = list(data[pos:pos + pattern_order_size]); pos += pattern_order_size
        instrument_number = data[pos]; pos += 1
        instruments = []
        for _ in range(instrument_number):
            pos += 1  # unused byte
            name_bytes = data[pos:pos + 15]; pos += 15
            name = name_bytes.split(b"\x00", 1)[0].decode("ascii", errors="replace")
            instruments.append(name)
    except (IndexError, struct.error):
        return None
    return {
        "pattern_number": pattern_number,
        "notes": notes,
        "pattern_order_size": pattern_order_size,
        "pattern_order": pattern_order,
        "instrument_number": instrument_number,
        "instruments": instruments,
    }


def parse_dmi_bytes(data, clamp_size=None):
    if len(data) < 8:
        return 0, 0, 0, 0, []
    length, samplerate, loopstart, looplength = struct.unpack_from("<4H", data, 0)
    avail = len(data) - 8
    if clamp_size is not None:
        avail = min(avail, clamp_size - 8)
    if length > avail:
        length = avail
    if length < 0:
        length = 0
    sample_bytes = data[8:8 + length]
    samples = list(struct.unpack("<%db" % length, sample_bytes)) if length > 0 else []
    return length, samplerate, loopstart, looplength, samples


def apply_sample_fixes(name, loop_start_raw, loop_length_raw, sample):
    for fname, field, val, fixes in SAMPLE_FIXES:
        if name != fname:
            continue
        current = loop_start_raw if field == "loop_start" else loop_length_raw
        if current == val:
            if fixes.get("pingpong"):
                sample.pingpong_loop = True
            if "loop_start" in fixes:
                sample.loop_start = fixes["loop_start"]
            if "loop_length" in fixes:
                sample.loop_length = fixes["loop_length"]


# ---------------------------------------------------------------------------
# XM data model
# ---------------------------------------------------------------------------

class PatternNote:
    __slots__ = ("note", "instrument", "volume", "effect", "effect_parameter", "it_instrument",
                 "keep_volume", "vol_slide", "retrig", "sample_offset", "env_restart")

    def __init__(self, note=0, instrument=0, volume=0, effect=0, effect_parameter=0, it_instrument=0,
                 keep_volume=False):
        # True when `instrument` was only inherited from the channel's memory (the row shows no
        # instrument column). Such a note must NOT reset the channel volume to the sample default.
        self.keep_volume = keep_volume
        # Volume slide carried by this cell, normalised across formats (MOD/XM Axy, 5xy, 6xy,
        # EAx/EBx, XM volume column; S3M/IT Dxy/Kxy/Lxy, IT volume column):
        #   None                - no slide
        #   (per_tick, fine)    - level change (0..64 scale) applied on every tick but the first,
        #                         and once on the first tick, respectively
        #   "mem"               - "repeat the last slide of this channel" (D00 / XM A00)
        self.vol_slide = None
        # Note retrigger carried by this cell (S3M/IT Qxy, MOD/XM E9x, XM Rxy): None, or
        # (interval_in_ticks, volume_change_code 0..15), or "mem" (= "repeat the last one", Q00).
        self.retrig = None
        # Sample start offset (S3M/IT Oxx, MOD/XM 9xx) in frames: None, an int, or "mem" (= "repeat the
        # last offset of this channel", Oxx with xx = 0).
        self.sample_offset = None
        # True on an XM cell that names an instrument but has no note: FT2 resets the channel
        # volume to the instrument's default and restarts its volume envelope (the sample keeps
        # playing). Chip-tune style songs put the instrument on every row for a rhythmic "gate".
        self.env_restart = False
        self.note = note
        self.instrument = instrument
        self.volume = volume
        self.effect = effect
        self.effect_parameter = effect_parameter
        # Raw IT instrument number (1-based) that triggered this note, or 0
        # if not applicable (MOD/S3M/XM, or an IT file not using the
        # instrument-based sample scheme). This is *not* the same thing as
        # `instrument` above: `instrument` is the already-resolved sample
        # slot (what every other format also puts there), while an IT
        # instrument's volume envelope lives on the instrument itself, not
        # the sample - several instruments can point at the same sample
        # with different envelopes, so the sample slot alone isn't enough
        # to look the envelope back up later. Only it_module_load ever sets
        # this; every other loader leaves it at 0, and the envelope-aware
        # code in _pattern_to_dmm_notes simply finds no matching envelope
        # in that case and behaves exactly as before.
        self.it_instrument = it_instrument

    def copy(self):
        n = PatternNote(self.note, self.instrument, self.volume, self.effect, self.effect_parameter,
                        self.it_instrument, self.keep_volume)
        n.vol_slide = self.vol_slide
        n.retrig = self.retrig
        n.sample_offset = self.sample_offset
        n.env_restart = self.env_restart
        return n


def vol_slide_from_xm_effect(effect, param, memory_on_zero):
    """MOD/XM effect column -> PatternNote.vol_slide. Axy/5xy/6xy: x up, y down per tick;
    EAx/EBx: fine slide up/down on the first tick. `memory_on_zero`: XM's A00 repeats the last
    slide, ProTracker's does nothing."""
    if effect in (0xA, 0x5, 0x6):
        if param == 0:
            return "mem" if (memory_on_zero and effect == 0xA) else None
        hi, lo = param >> 4, param & 15
        return (hi if hi else -lo, 0)
    if effect == 0xE:
        sub, val = param >> 4, param & 15
        if val and sub == 0xA:
            return (0, val)
        if val and sub == 0xB:
            return (0, -val)
    return None


def vol_slide_from_s3m_param(param):
    """S3M/IT Dxy (also the volume half of Kxy/Lxy): D0y down, Dx0 up (both per tick),
    DFy fine down, DxF fine up (first tick only), D00 = repeat the last one."""
    if param == 0:
        return "mem"
    hi, lo = param >> 4, param & 15
    if hi == 0:
        return (-lo, 0)
    if lo == 0:
        return (hi, 0)
    if lo == 0xF:
        return (0, hi)
    if hi == 0xF:
        return (0, -lo)
    return None


def sample_offset_from_param(param):
    """Oxx / 9xx: the note starts xx*256 frames into the sample; 00 repeats the last offset."""
    return param * 256 if param else "mem"


def retrig_from_param(param, memory_on_zero=True):
    """S3M/IT Qxy, XM Rxy: x = volume change on every retrigger, y = interval in ticks."""
    if param == 0:
        return "mem" if memory_on_zero else None
    interval, vcode = param & 0x0F, param >> 4
    return (interval, vcode) if interval else None


def retrig_from_xm_effect(effect, param):
    """MOD/XM effect column -> PatternNote.retrig: E9x = retrigger every x ticks (no volume change),
    XM 0x1B (Rxy) = multi retrigger with volume change."""
    if effect == 0xE and (param >> 4) == 0x9:
        return (param & 0x0F, 0) if (param & 0x0F) else None
    if effect == 0x1B:
        return retrig_from_param(param)
    return None


def combine_vol_slides(a, b):
    if a is None:
        return b
    if b is None or a == "mem" or b == "mem":
        return a
    return (a[0] + b[0], a[1] + b[1])


def make_note(note, instrument=0, volume=0, effect=0, effect_parameter=0):
    return PatternNote(note, instrument, volume, effect, effect_parameter)


class Pattern:
    def __init__(self):
        self.rows = 0
        self.channels = 0
        self.cells = []

    def resize(self, new_rows, new_channels):
        old_rows, old_channels, old_cells = self.rows, self.channels, self.cells
        self.rows = new_rows
        self.channels = new_channels
        if new_rows > 0 and new_channels > 0:
            new_cells = [PatternNote() for _ in range((new_rows + 1) * new_channels)]
            for row in range(min(new_rows, old_rows)):
                for ch in range(min(new_channels, old_channels)):
                    new_cells[row * new_channels + ch] = old_cells[row * old_channels + ch]
            self.cells = new_cells
        else:
            self.cells = []

    def get_note(self, row, channel):
        cell = row * self.channels + channel
        if 0 <= cell < len(self.cells):
            return self.cells[cell]
        return PatternNote()

    def set_note(self, row, channel, note):
        cell = row * self.channels + channel
        if 0 <= cell < len(self.cells):
            self.cells[cell] = note

    def is_empty(self):
        for row in range(self.rows):
            for ch in range(self.channels):
                n = self.get_note(row, ch)
                if n.note or n.instrument or n.volume or n.effect or n.effect_parameter:
                    return False
        return True


class Sample:
    def __init__(self):
        self.data = []
        self.sample_length = 0
        self.bits = 8
        self.channels = 1
        self.name = ""
        self.loop = False
        self.pingpong_loop = False
        self.loop_start = 0
        self.loop_length = 0
        self.volume = 64
        self.panning = 128
        self.relative_note = 0
        self.finetune = 0
        self.adpcm4 = False


class Envelope:
    def __init__(self):
        self.active = False
        self.loop = False
        self.sustain = False
        self.points = [(0, 0)] * 12
        self.number_of_points = 0
        self.loop_start_point = 0
        self.loop_end_point = 0
        self.sustain_point = 0


def xm_envelope_to_it_style(env):
    """Adapts an XM-native Envelope (fixed 12-point array, single sustain point) into the
    ITVolumeEnvelope shape the DMM-conversion volume/shape code already understands, so a plain
    XM instrument's attack/decay/sustain gets applied exactly like an IT one does - previously it
    was silently ignored for XM/MOD/S3M source files (their PatternNote.it_instrument is never
    set, so the IT-only envelope lookup in dmm2xm_convert.py never found anything for them),
    which let every note hold at a flat volume for as long as it was sounding instead of decaying
    the way its own instrument says it should. Returns None when there is nothing to apply."""
    if not env.active or env.number_of_points <= 0:
        return None
    out = ITVolumeEnvelope()
    out.active = True
    out.points = list(env.points[:env.number_of_points])
    if out.points[0][0] != 0:
        out.points[0] = (0, out.points[0][1])
    n = len(out.points)
    out.sustain = env.sustain and 0 <= env.sustain_point < n
    out.sustain_start = out.sustain_end = env.sustain_point
    out.loop = env.loop and 0 <= env.loop_start_point < n and 0 <= env.loop_end_point < n
    out.loop_start, out.loop_end = env.loop_start_point, env.loop_end_point
    return out


class ITVolumeEnvelope:
    """A raw IT-style volume envelope, kept separate from the XM-oriented
    Envelope class above (which only supports a single fixed sustain point
    and a 12-point cap, matching XM's own on-disk format). IT allows up to
    25 points and a full sustain *loop* (a start/end pair, not just one
    point) as well as an independent regular loop - both of which need to
    be kept intact to correctly work out how loud a note really sounds
    while it's held, rather than just at the instant it's triggered. This
    is only ever used by the DMM-conversion volume calculation in
    dmm2xm_convert.py; it plays no part in xm_module_save's XM export.
    """
    __slots__ = ("active", "loop", "sustain", "points",
                 "loop_start", "loop_end", "sustain_start", "sustain_end")

    def __init__(self):
        self.active = False
        self.loop = False
        self.sustain = False
        self.points = []            # [(tick, value 0..64), ...], tick ascending, points[0][0] == 0
        self.loop_start = 0         # node indices into `points`
        self.loop_end = 0
        self.sustain_start = 0
        self.sustain_end = 0


class Instrument:
    def __init__(self):
        self.name = ""
        self.sample = {}  # index (0-based) -> Sample
        self.sample_map = [0] * 97  # 1..96 used
        self.fade_out = 0
        self.vibrato_type = 0
        self.vibrato_sweep = 0
        self.vibrato_depth = 0
        self.vibrato_rate = 0
        self.volume_envelope = Envelope()
        self.panning_envelope = Envelope()


class XMModule:
    def __init__(self):
        self.name = ""
        self.comment = ""
        self.instruments = {}  # 1..255 -> Instrument
        self.patterns = [Pattern() for _ in range(256)]
        self.pattern_order = [0] * 256
        self.global_volume = 64
        self.master_volume = 128
        self.start_speed = 6
        self.start_tempo = 125
        self.track_length = 0
        self.restart_position = 0
        self.linear_slides = True
        self.extended_filter_range = False
        self.number_of_channels = 0
        # raw IT instrument number (1-based) -> ITVolumeEnvelope. Populated
        # only by it_module_load (new-format IT instruments, cmwt >= 0x200);
        # empty for every other format/loader, and for old-format IT files.
        # Looked up via PatternNote.it_instrument, purely for the
        # duration-aware volume calculation in dmm2xm_convert.py.
        self.it_volume_envelopes = {}

    def number_of_patterns(self):
        result = 0
        for i in range(256):
            if self.patterns[i].rows > 0:
                result = i + 1
        return result

    def number_of_instruments(self):
        result = 0
        for i in range(1, 256):
            if self.instruments.get(i) is not None:
                result = i
        return result


def sample_rate_to_xm(sample_rate, emulate_bug):
    if sample_rate != 0:
        finetune = round(math.log2(sample_rate / 8363) * 1536)
    else:
        finetune = 0
    if emulate_bug:
        finetune += 64
    if finetune < 0:
        relative_note = trunc_div(finetune - 63, 128)
    else:
        relative_note = trunc_div(finetune + 64, 128)
    finetune -= relative_note * 128
    if finetune < 0:
        finetune = -(trunc_mod(-finetune, 128))
    else:
        finetune = trunc_mod(finetune, 128)
    relative_note = clamp(relative_note, -127, 127)
    return relative_note, finetune


def xm_to_sample_rate(relative_note, finetune):
    return round(8363 * (2 ** (((relative_note * 128) + finetune) / 1536)))


def encode_8bit(data):
    out = bytearray(len(data))
    last = 0
    for i, v in enumerate(data):
        v = clamp(trunc_div(int(v), 256), -128, 127)
        d = wrap_s8(v - last)
        out[i] = d & 0xFF
        last = v
    return bytes(out)


def encode_16bit(data):
    out = bytearray(len(data) * 2)
    last = 0
    for i, v in enumerate(data):
        v = clamp(int(v), -32768, 32767)
        d = wrap_s16(v - last)
        struct.pack_into("<h", out, i * 2, d)
        last = v
    return bytes(out)


def decode_8bit(raw, length, channels):
    values = list(struct.unpack("<%db" % (length * channels), raw)) if length else []
    if channels == 2:
        data = [0] * (length * 2)
        last = 0
        for i in range(length):
            last = wrap_s8(values[i] + last)
            data[i * 2] = last * 256
        last = 0
        for i in range(length):
            last = wrap_s8(values[length + i] + last)
            data[i * 2 + 1] = last * 256
    else:
        data = [0] * length
        last = 0
        for i in range(length):
            last = wrap_s8(values[i] + last)
            data[i] = last * 256
    return data


def decode_16bit(raw, length, channels):
    values = list(struct.unpack("<%dh" % (length * channels), raw)) if length else []
    if channels == 2:
        data = [0] * (length * 2)
        last = 0
        for i in range(length):
            last = wrap_s16(values[i] + last)
            data[i * 2] = last
        last = 0
        for i in range(length):
            last = wrap_s16(values[length + i] + last)
            data[i * 2 + 1] = last
    else:
        data = [0] * length
        last = 0
        for i in range(length):
            last = wrap_s16(values[i] + last)
            data[i] = last
    return data


# ---------------------------------------------------------------------------
# Amiga MOD loading (backward conversion also accepts .mod, like the
# original tool). Ported from BeRoXM.pas's LoadMOD/MOD2XMNote/MOD2XMFineTune.
#
# NOTE ON A BUG IN THE ORIGINAL: the original Pascal reads a 5-byte tag to
# check for an "ADPCM" marker before each sample's raw data; if the tag does
# NOT match (the overwhelming majority of real MOD files), it only rewinds
# the stream by 4 of the 5 bytes it just consumed ("Dec(DataPosition,4)"),
# which would misalign every non-ADPCM sample's data by one byte and
# corrupt the decoded audio. This Python port intentionally fixes that
# (rewinds the full 5 bytes / simply doesn't consume them speculatively)
# rather than reproducing the corruption.
# ---------------------------------------------------------------------------

_AMIGA_MOD_PERIODS = [
    1712, 1616, 1524, 1440, 1356, 1280, 1208, 1140, 1076, 1016, 960, 906,
    856, 808, 762, 720, 678, 640, 604, 570, 538, 508, 480, 453,
    428, 404, 381, 360, 339, 320, 302, 285, 269, 254, 240, 226,
    214, 202, 190, 180, 170, 160, 151, 143, 135, 127, 120, 113,
    107, 101, 95, 90, 85, 80, 75, 71, 67, 63, 60, 56,
]


def mod2xm_note(period):
    if period <= 0:
        return 0
    for note in range(1, 61):
        if period >= _AMIGA_MOD_PERIODS[note - 1]:
            return note + 24
    return 0


def mod2xm_finetune(finetune):
    result = finetune
    if result < 0:
        result = 0
    if result > 15:
        result = 15
    if result > 7:
        result = -(8 - (result - 8))
    return result * 16


def _mod_channel_count_from_tag(tag):
    """Returns channel count, 0 if unrecognized, or -1 for an unsupported
    packed ('PPxx') variant that should abort loading entirely."""
    if len(tag) < 4:
        return 0
    if tag[0:2] == b"PP":
        return -1
    if tag in (b"M.K.", b"M!K!", b"M&K!", b"N.T."):
        return 4
    if tag in (b"OCTA", b"OKTA", b"CD81", b"WOW!"):
        return 8
    T, D, Z, F, L, X, O, C, H, N, E = (ord(c) for c in "TDZFLXOCHNE")
    if tag[0] == T and tag[1] == D and tag[2] == Z:
        return tag[3] - 48
    if tag[0] == F and tag[1] == L and tag[2] == T:
        v = tag[3] - 48
        return v if 1 <= v <= 8 else 0
    if tag[0] == E and tag[1] == X and tag[2] == O:
        v = tag[3] - 48
        return v if 1 <= v <= 8 else 0
    if tag[1] == C and tag[2] == H and tag[3] == N:
        return tag[0] - 48
    if tag[2] == C and tag[3] == N:
        return (tag[0] - 48) * 10 + (tag[1] - 48)
    if tag[2] == C and tag[3] == H:
        return (tag[0] - 48) * 10 + (tag[1] - 48)
    if tag[0] == E and tag[1] == X:
        return (tag[2] - 48) * 10 + (tag[3] - 48)
    return 0


def _parse_mod_sample(data, pos):
    name = data[pos:pos + 22]
    length_words = struct.unpack_from(">H", data, pos + 22)[0]
    finetune_raw = data[pos + 24]
    volume = data[pos + 25]
    loop_start_words = struct.unpack_from(">H", data, pos + 26)[0]
    loop_length_words = struct.unpack_from(">H", data, pos + 28)[0]
    return name, length_words, finetune_raw, volume, loop_start_words, loop_length_words


def _read_adpcm4_sample_mono(data, pos, size):
    table = list(data[pos:pos + 16])
    pos += 16
    half_length = (size + 1) // 2
    buf = [0] * size
    delta = 0
    idx = 0
    for _ in range(half_length):
        a_byte = data[pos]; pos += 1
        delta = wrap_s8(delta + table[a_byte & 0xF])
        if idx < size:
            buf[idx] = delta
            idx += 1
        delta = wrap_s8(delta + table[a_byte >> 4])
        if idx < size:
            buf[idx] = delta
            idx += 1
    return [v * 256 for v in buf], pos


def mod_module_load(path):
    with open(path, "rb") as f:
        raw = f.read()
    data_size = len(raw)
    # Zero-pad generously so header/tag lookups on short/garbage files never
    # raise (mirrors the original Read()'s "zero-fill past EOF" behaviour);
    # the actual size-consistency checks below always compare against
    # data_size, never against the padded length.
    data = raw + b"\x00" * 4096

    old_format = False
    title = data[0:20]
    header31_tag = data[1080:1084]

    chn = _mod_channel_count_from_tag(header31_tag)
    if chn == -1:
        return None  # unsupported packed ("PPxx") MOD variant

    if chn == 0:
        old_format = True
        how_many_samples = 15
        how_many_channels = 4
        track_length = data[20 + 15 * 30]
        restart_position = data[20 + 15 * 30 + 1]
        pattern_order = list(data[20 + 15 * 30 + 2:20 + 15 * 30 + 2 + 128])
        pattern_start_position = 20 + 15 * 30 + 2 + 128  # 600
    else:
        how_many_samples = 31
        how_many_channels = chn
        track_length = data[20 + 31 * 30]
        restart_position = data[20 + 31 * 30 + 1]
        pattern_order = list(data[20 + 31 * 30 + 2:20 + 31 * 30 + 2 + 128])
        pattern_start_position = 20 + 31 * 30 + 2 + 128 + 4  # 1084

    if restart_position > track_length:
        restart_position = 0

    xm = XMModule()
    xm.name = title.split(b"\x00", 1)[0].decode("cp1251", errors="replace")
    xm.track_length = track_length
    xm.restart_position = restart_position
    xm.pattern_order = pattern_order + [0] * (256 - len(pattern_order))

    total_sample_length = 0
    samples = []
    for counter in range(1, how_many_samples + 1):
        pos = 20 + (counter - 1) * 30
        name_b, length_w, finetune_raw, volume, loop_start_w, loop_length_w = \
            _parse_mod_sample(data, pos)
        length_bytes = length_w << 1
        loop_start_bytes = loop_start_w << 1
        loop_length_bytes = loop_length_w << 1

        inst = Instrument()
        inst.fade_out = 65536
        inst.volume_envelope.active = False
        inst.panning_envelope.active = False
        inst.vibrato_rate = 0
        smp = Sample()
        smp.bits = 8
        smp.channels = 1
        smp.sample_length = length_bytes
        smp.loop_start = loop_start_bytes
        smp.loop_length = loop_length_bytes
        smp.loop = False
        smp.finetune = mod2xm_finetune(finetune_raw)
        smp.volume = volume
        smp.panning = 128
        smp.name = name_b.split(b"\x00", 1)[0].decode("cp1251", errors="replace")
        inst.sample[0] = smp
        xm.instruments[counter] = inst
        total_sample_length += length_bytes
        samples.append(smp)

    pattern_b = 0
    for c in range(128):
        po = xm.pattern_order[c]
        if po <= 127 and po > pattern_b:
            pattern_b = po
    pattern_b += 1

    if how_many_channels == 4:
        if pattern_start_position + (pattern_b * 8 * 4 * 64) + total_sample_length == data_size:
            how_many_channels = 8

    denom = (((1024 // 4) * how_many_channels) // 64) * 64
    pattern_a = trunc_div(data_size - pattern_start_position - total_sample_length, denom) if denom else 0

    if old_format or (pattern_a > 64 and pattern_b <= 64):
        pattern_a = pattern_b

    if pattern_a == pattern_b:
        how_many_patterns = pattern_a
    elif pattern_b < pattern_a:
        how_many_patterns = pattern_b
    elif pattern_b > 64 and (header31_tag != b"M!K!" and not old_format):
        how_many_patterns = pattern_a
    else:
        how_many_patterns = pattern_b
    if how_many_patterns > 128:
        how_many_patterns = 128

    if pattern_start_position + (how_many_patterns * how_many_channels * 4 * 64) + total_sample_length != data_size:
        return None

    pos = pattern_start_position
    for counter in range(how_many_patterns):
        xm.patterns[counter].resize(64, how_many_channels)
        for row in range(64):
            for ch in range(how_many_channels):
                buf = data[pos:pos + 4]
                pos += 4
                b1, b2, b3, b4 = buf[0], buf[1], buf[2], buf[3]
                note = PatternNote()
                note.note = mod2xm_note(((b1 & 0x0F) << 8) | b2)
                note.instrument = (b1 & 0xF0) | (b3 >> 4)
                note.volume = 0
                note.effect = b3 & 0xF
                note.effect_parameter = b4
                note.vol_slide = vol_slide_from_xm_effect(note.effect, note.effect_parameter, False)
                note.retrig = retrig_from_xm_effect(note.effect, note.effect_parameter)
                if note.effect == 0x9:
                    note.sample_offset = sample_offset_from_param(note.effect_parameter)
                xm.patterns[counter].set_note(row, ch, note)

    # ProTracker rule (same trap as fixed for S3M at s3m_module_load):
    # a MOD pattern cell carries no per-note volume; triggering a note
    # without an explicit Cxx command resets the channel to the sample's
    # own default volume (sample header byte). Without this fill the
    # shared DMM-note conversion below would read volume 0 as "unset"
    # and bake every such note at maximum (127) instead of the balance
    # the tracker author intended.
    for counter in range(how_many_patterns):
        for row in range(64):
            for ch in range(how_many_channels):
                note = xm.patterns[counter].get_note(row, ch)
                # (a sample number WITHOUT a period counts too: it resets the volume to the sample
                # default without retriggering - chip-tunes use it for rhythmic volume gating)
                if note.instrument != 0 \
                        and note.volume == 0 and note.effect != EFFECT_VOLUME:
                    smp = samples[note.instrument - 1] \
                        if 1 <= note.instrument <= len(samples) else None
                    if smp is not None:
                        note.volume = 0x10 + clamp(smp.volume, 0, 64)
                        xm.patterns[counter].set_note(row, ch, note)

    for counter in range(1, how_many_samples + 1):
        smp = xm.instruments[counter].sample[0]
        if smp.sample_length > 1:
            if smp.loop_start > smp.sample_length:
                smp.loop_start = 0
                smp.loop_length = 0
            if smp.loop_start + smp.loop_length > smp.sample_length:
                smp.loop_length = smp.sample_length - smp.loop_start
                if smp.loop_length < 2:
                    smp.loop_length = 2
            smp.loop = smp.loop_length > 2
            if smp.sample_length > 0:
                if data[pos:pos + 5] == b"ADPCM":
                    pos += 5
                    smp.data, pos = _read_adpcm4_sample_mono(data, pos, smp.sample_length)
                else:
                    raw_bytes = data[pos:pos + smp.sample_length]
                    pos += smp.sample_length
                    vals = struct.unpack("<%db" % len(raw_bytes), raw_bytes) if raw_bytes else []
                    smp.data = [v * 256 for v in vals]

    xm.number_of_channels = how_many_channels
    xm.linear_slides = False
    return xm


# ---------------------------------------------------------------------------
# Scream Tracker 3 (.s3m) loading.
#
# S3M is structurally closer to XM/MOD than IT is (fixed 64-row patterns,
# a simple channel/note/volume/command cell format, uncompressed PCM
# samples), so it reuses the same internal XMModule/PatternNote/Sample
# representation as the MOD and XM loaders above. Two format-specific traps
# to note (see also the MOD default-volume fix elsewhere in this file):
#
#   * Sample length/loop points in the instrument header are given directly
#     in *sample frames*, unlike XM/MOD which give them in bytes/words -
#     no _sample_frames_from_disk_length() conversion is needed here.
#   * 8-bit (and, rarely, 16-bit) sample data is plain, *unsigned* PCM by
#     default (governed by the file's global "Ffi" flag), not delta-coded
#     like XM, and not signed like MOD/DMI - needs an explicit unsigned ->
#     signed conversion.
#   * Like MOD, S3M pattern cells have no per-note volume unless one is
#     explicitly written; triggering a note without one resets the channel
#     to the sample's own default volume (header byte 0x1C). This is the
#     exact same trap that was fixed for MOD - it is handled the same way
#     here, at load time, rather than in the shared DMM-note conversion code.
# ---------------------------------------------------------------------------

S3M_CMD_SET_SPEED = 1    # 'A'
S3M_CMD_PATTERN_BREAK = 3   # 'C'
S3M_CMD_SET_TEMPO = 20   # 'T'


def _decode_s3m_pcm(raw, frames, channels, bits, unsigned):
    """Decodes plain (non delta-coded) PCM sample data as stored in .s3m
    files, converting to the same internal convention used everywhere else
    in this module: 8-bit values pre-scaled x256 (to share value range with
    16-bit data), 16-bit values used as-is, non-interleaved stereo (all of
    the left channel's frames, followed by all of the right channel's)."""
    if bits == 16:
        if unsigned:
            raw_vals = struct.unpack_from("<%dH" % (frames * channels), raw)
            vals = [v - 32768 for v in raw_vals]
        else:
            vals = list(struct.unpack_from("<%dh" % (frames * channels), raw))
        if channels == 2:
            data = [0] * (frames * 2)
            for i in range(frames):
                data[i * 2] = vals[i]
                data[i * 2 + 1] = vals[frames + i]
            return data
        return vals
    else:
        if unsigned:
            vals = [b - 128 for b in raw[:frames * channels]]
        else:
            vals = list(struct.unpack_from("<%db" % (frames * channels), raw))
        if channels == 2:
            data = [0] * (frames * 2)
            for i in range(frames):
                data[i * 2] = vals[i] * 256
                data[i * 2 + 1] = vals[frames + i] * 256
            return data
        return [v * 256 for v in vals]


def s3m_module_load(path):
    with open(path, "rb") as f:
        raw = f.read()
    # Generous zero-padding so a truncated/malformed file never raises
    # instead of just producing a shorter/quieter result (same philosophy
    # as mod_module_load's padding).
    data = raw + b"\x00" * 65536

    if data[0x2C:0x30] != b"SCRM":
        return None  # not an S3M file

    title = data[0:20]
    try:
        ord_num, ins_num, pat_num, s3m_flags, s3m_cwtv, ffi = struct.unpack_from("<6H", data, 0x20)
    except struct.error:
        return None
    if ord_num > 256 or ins_num > 255 or pat_num > 255:
        return None

    initial_speed = data[0x31] or 6
    initial_tempo = data[0x32] or 125
    unsigned_samples = (ffi != 1)  # Ffi==1 means signed; 2 (or anything else) means unsigned

    chan_settings = data[0x40:0x60]
    # Channel setting: 0-7 left PCM, 8-15 right PCM, 16-31 Adlib (melody/drums - no PCM data, they can
    # never make a sound here), bit 7 = channel disabled (muted), 255 = unused. Only enabled PCM
    # channels are real; the others used to be converted as silent/muted-but-played channels and
    # could take one of the eight DMM channel slots.
    used_channels = [c for c in range(32) if chan_settings[c] < 16]
    if not used_channels:
        return None
    chan_map = {c: i for i, c in enumerate(used_channels)}
    number_of_channels = len(used_channels)

    pos = 0x60
    order_bytes = data[pos:pos + ord_num]
    pos += ord_num

    pattern_order = []
    for b in order_bytes:
        if b == 255:  # "--" end of song marker
            break
        if b == 254:  # "++" skip marker, not a real pattern
            continue
        pattern_order.append(b)
    track_length = len(pattern_order)

    try:
        inst_ptrs = list(struct.unpack_from("<%dH" % ins_num, data, pos))
    except struct.error:
        return None
    pos += ins_num * 2
    try:
        pat_ptrs = list(struct.unpack_from("<%dH" % pat_num, data, pos))
    except struct.error:
        return None

    xm = XMModule()
    xm.name = title.split(b"\x00", 1)[0].decode("cp1251", errors="replace")
    xm.track_length = track_length
    xm.restart_position = 0  # S3M has no dedicated restart-position field
    xm.pattern_order = pattern_order + [0] * (256 - len(pattern_order))
    xm.number_of_channels = number_of_channels
    xm.linear_slides = False
    xm.start_speed = initial_speed
    xm.start_tempo = initial_tempo
    # ST3.00 "fast volume slides" (header flag 0x40, or file made by ST3.00 = cwt/v 0x1300): Dxy also
    # slides on the first tick of every row, i.e. `speed` steps per row instead of `speed - 1`.
    xm.fast_volume_slides = bool(s3m_flags & 0x40) or s3m_cwtv == 0x1300

    samples = [None] * (ins_num + 1)  # 1-based, like xm.instruments
    for counter in range(1, ins_num + 1):
        ptr = inst_ptrs[counter - 1]
        inst = Instrument()
        inst.fade_out = 65536
        inst.volume_envelope.active = False
        inst.panning_envelope.active = False
        smp = Sample()
        smp.panning = 128
        if ptr != 0:
            try:
                base = ptr * 16
                ihdr = data[base:base + 80]
                itype = ihdr[0]
                # The name field sits at a fixed offset in every S3M
                # instrument header regardless of type, and some real-world
                # files (e.g. old-school "credits split across empty
                # instrument slots" tricks) put meaningful text there even
                # on type=0 ("empty"/no-sample) entries. Read it
                # unconditionally so that text isn't silently lost - it
                # doesn't end up in the DMM output either way (only slots
                # with actual sample data get a name written there), but
                # it's still useful for anyone inspecting the loaded module.
                name_b = ihdr[0x30:0x4C]
                smp.name = name_b.split(b"\x00", 1)[0].decode("cp1251", errors="replace")
                if itype == 1:  # 1 = PCM sample; anything else (Adlib etc.) unsupported
                    hmem = ihdr[0x0D]
                    memseg = struct.unpack_from("<H", ihdr, 0x0E)[0]
                    length, loopbegin, loopend = struct.unpack_from("<3I", ihdr, 0x10)
                    vol = ihdr[0x1C]
                    pack = ihdr[0x1E]
                    sflags = ihdr[0x1F]
                    c2spd = struct.unpack_from("<I", ihdr, 0x20)[0]
                    smp.bits = 16 if (sflags & 4) else 8
                    smp.channels = 2 if (sflags & 2) else 1
                    smp.volume = clamp(vol, 0, 64)
                    smp.relative_note, smp.finetune = sample_rate_to_xm(c2spd, False)
                    if pack == 0 and 0 < length <= MAX_SAMPLE_SIZE:
                        data_off = ((hmem << 16) | memseg) * 16
                        bytes_per_frame = (2 if smp.bits == 16 else 1) * smp.channels
                        disk_bytes = length * bytes_per_frame
                        sample_raw = data[data_off:data_off + disk_bytes]
                        smp.data = _decode_s3m_pcm(sample_raw, length, smp.channels, smp.bits, unsigned_samples)
                        smp.sample_length = length
                        if (sflags & 1) and loopend > loopbegin:
                            smp.loop = True
                            smp.loop_start = loopbegin
                            smp.loop_length = loopend - loopbegin
                    # else: packed (compressed) sample data - not supported,
                    # left as an empty (zero-length) sample.
                # else: Adlib/FM or unrecognised instrument type - no PCM
                # data to read; left as an empty (zero-length) sample.
            except (struct.error, IndexError):
                smp = Sample()
                smp.panning = 128
        inst.sample[0] = smp
        xm.instruments[counter] = inst
        samples[counter] = smp

    s3m_jumps = []   # (pattern, row, channel, target) of every Bxx, in file order
    for p in range(pat_num):
        xm.patterns[p].resize(64, number_of_channels)
        ptr = pat_ptrs[p] if p < len(pat_ptrs) else 0
        if ptr == 0:
            continue  # missing pattern -> 64 empty rows
        reader = _ByteReader(data)
        reader.seek(ptr * 16 + 2)  # skip the 2-byte "packed length" field
        row = 0
        while row < 64:
            what = reader.read_byte()
            if what == 0:
                row += 1
                continue
            s3m_chan = what & 31
            has_note = bool(what & 32)
            has_vol = bool(what & 64)
            has_cmd = bool(what & 128)

            note_byte = inst_byte = vol_byte = cmd_byte = info_byte = 0
            if has_note:
                note_byte = reader.read_byte()
                inst_byte = reader.read_byte()
            if has_vol:
                vol_byte = reader.read_byte()
            if has_cmd:
                cmd_byte = reader.read_byte()
                info_byte = reader.read_byte()

            ch = chan_map.get(s3m_chan)
            if ch is None:
                continue  # cell on a disabled/unused channel - bytes already consumed, discard

            note = PatternNote()
            if has_note:
                if note_byte == 255:
                    note.note = 0
                elif note_byte == 254:
                    note.note = NOTE_STOP
                else:
                    octave, semitone = note_byte >> 4, note_byte & 0x0F
                    note.note = octave * 12 + semitone + 1
                note.instrument = inst_byte

            if has_vol:
                note.volume = 0x10 + clamp(vol_byte, 0, 64)
            elif has_note and note.instrument != 0 and note.note != NOTE_STOP:
                # note.note == 0 here means "instrument number without a note": ST3 resets the
                # channel volume to the sample's default (no retrigger). Songs use it as a gate,
                # e.g. "... 01" on even rows and a volume-column 8 on odd rows; ignoring it left
                # a steady quiet drone instead of the pulsing original.
                smp = samples[note.instrument] if note.instrument < len(samples) else None
                if smp is not None:
                    note.volume = 0x10 + clamp(smp.volume, 0, 64)

            if has_cmd:
                if cmd_byte == S3M_CMD_SET_SPEED:
                    note.effect = EFFECT_SPEED_TEMPO
                    note.effect_parameter = min(info_byte, 31)
                    if not info_byte:       # A00 is ignored
                        note.effect = 0
                elif cmd_byte == S3M_CMD_SET_TEMPO:
                    note.effect = EFFECT_SPEED_TEMPO
                    note.effect_parameter = info_byte
                    if info_byte < 32:      # T0x/T1x are tempo slides, not tempo values
                        note.effect = 0
                        note.effect_parameter = 0
                elif cmd_byte == S3M_CMD_PATTERN_BREAK:
                    note.effect = EFFECT_PATTERN_BREAK
                elif cmd_byte == 17:                 # Q - retrigger note (+ volume change)
                    note.retrig = retrig_from_param(info_byte)
                elif cmd_byte == 15:                 # O - sample offset
                    note.sample_offset = sample_offset_from_param(info_byte)
                elif cmd_byte in (4, 11, 12):        # D, K (vibrato+vol), L (porta+vol)
                    note.vol_slide = vol_slide_from_s3m_param(info_byte)
                elif cmd_byte == 19:                 # S - the same three sub-commands IT/MOD keep
                    sub, val = info_byte >> 4, info_byte & 0x0F
                    if sub == EXT_EFFECT_NOTE_CUT:   # SCx - note cut (used to be dropped: notes rang on)
                        note.effect = EFFECT_EXTENDED_EFFECTS
                        note.effect_parameter = val | (EXT_EFFECT_NOTE_CUT << 4)
                    elif sub == EXT_EFFECT_NOTE_DELAY:   # SDx - note delay
                        note.effect = EFFECT_EXTENDED_EFFECTS
                        note.effect_parameter = val | (EXT_EFFECT_NOTE_DELAY << 4)
                    elif sub == EXT_EFFECT_PATTERN_DELAY and val:   # SEx - pattern delay
                        note.effect = EFFECT_EXTENDED_EFFECTS
                        note.effect_parameter = val | (EXT_EFFECT_PATTERN_DELAY << 4)
                elif cmd_byte == 2:                  # B - position jump (only the loop-back case is used, below)
                    s3m_jumps.append((p, row, ch, info_byte))
                # all other commands (arpeggio, slides, vibrato, position
                # jump, ...) have no DMM equivalent and are dropped - this
                # matches the existing MOD/XM conversion, which only ever
                # looks at Set Speed/Tempo and Pattern Break either way.

            xm.patterns[p].set_note(row, ch, note)

    # Song loop: a Bxx in the first-executed row of the last pattern that jumps backwards is what makes
    # an S3M song repeat (no restart field in the format). The value after DMM's 255 marker is an
    # order index, like XM's restart position. The B parameter counts the raw order list, "++"
    # skip markers included.
    if pattern_order:
        raw_to_compact = {}
        n_valid = 0
        for i, b in enumerate(order_bytes):
            if b == 255:
                break
            raw_to_compact[i] = n_valid
            if b != 254:
                n_valid += 1
        last_pat = pattern_order[-1]
        jumps = sorted((row_, ch_, tgt) for (p_, row_, ch_, tgt) in s3m_jumps if p_ == last_pat)
        if jumps:
            target = min(raw_to_compact.get(jumps[0][2], track_length - 1), track_length - 1)
            if target < track_length - 1:
                xm.restart_position = target

    return xm


# ---------------------------------------------------------------------------
# Impulse Tracker (.it) loading.
#
# IT is structurally the most different of the three "extra" formats loaded
# above (MOD and S3M both map an "instrument" directly onto a single sample,
# exactly like DMM itself, so they reuse the shared XMModule/Instrument/
# Sample representation almost mechanically). IT instead has a real
# indirection layer: a pattern event names an *instrument*, and each
# instrument carries its own 120-entry Note-Sample/Keyboard Table that picks
# a *sample* (and can remap the note) per incoming note - the same
# instrument can legitimately trigger different samples on different notes
# (drum-kit-style instruments are common in real IT modules). Since this
# tool's "instrument" slots are 1:1 with actual samples everywhere else
# (matching DMM's own one-sample-per-instrument model), every IT *sample*
# becomes one XMModule instrument slot here, and each pattern note's raw IT
# instrument+note pair is resolved through the owning instrument's keyboard
# table *at load time* into a direct (resolved sample, resolved note) pair -
# see _it_resolve_note_instrument() below. This is deliberately more
# faithful than reusing the generic backward-conversion "remembered
# instrument" fallback (the one MOD/S3M/XM rely on for "note without an
# explicit instrument this row") would be: that fallback assumes an
# instrument *number* alone is a stable, note-independent identity, which
# is not true for IT drum-kit instruments, so this resolves the sample
# explicitly on every note trigger instead of leaving it to be guessed
# downstream.
#
# The other IT-specific piece of work is IT214/IT215 sample decompression:
# real-world .it files very often store sample data as adaptive-bit-width
# delta-compressed blocks rather than plain PCM. _IT214Decompressor below is
# a Python port of Ben "GreaseMonkey" Russell's public-domain reference
# decompressor (itself matching OpenMPT's ITCompression.cpp / the original
# itsex.c algorithm), verified against real compressed samples from actual
# .it files: decompressing every compressed sample in sequence lands byte-
# exact on the following sample's own on-disk offset (and the very last
# sample ends exactly at EOF), which would not happen if the bit-unpacking
# were even slightly wrong.
# ---------------------------------------------------------------------------

IT_EFFECT_SET_SPEED = 1     # 'A'
IT_EFFECT_PATTERN_BREAK = 3     # 'C'
IT_EFFECT_S = 19    # 'S' - "special" commands, S0x..SFx
IT_EFFECT_SET_TEMPO = 20    # 'T'


class _IT214DecompressError(Exception):
    pass


class _IT214Decompressor:
    """Decompresses one on-disk IT214/IT215-compressed sample block (up to
    0x8000 samples for 8-bit / 0x4000 samples for 16-bit - itsex.c's own
    chunking limit; the caller loops this once per block). Python port of
    munch.py's IT214Decompressor (public domain); see the module-level note
    above this section for provenance and verification.

    On malformed/truncated input this pads the rest of the block with the
    last known sample value rather than raising, matching this project's
    general policy elsewhere (mod_module_load, s3m_module_load) of
    degrading gracefully instead of crashing the whole conversion over one
    bad sample.
    """

    def __init__(self, data, length, is16):
        self.data = data
        self.dpos = 0
        self.bpos = 0
        self.brem = 8

        self.fetch_a = 4 if is16 else 3
        self.lower_b = -8 if is16 else -4
        self.upper_b = 7 if is16 else 3
        self.widthtop = 17 if is16 else 9
        self.width = self.widthtop
        self.unpack_mask = 0xFFFF if is16 else 0xFF

        self.length = length
        self.running_count = 0
        self.unpacked_root = 0
        self.unpacked_data = []
        try:
            self._unpack()
        except _IT214DecompressError:
            while self.running_count < length:
                self.unpacked_data.append(self.unpacked_root)
                self.running_count += 1

    def _end_of_block(self):
        return self.dpos >= len(self.data)

    def _read(self, width):
        v = 0
        vpos = 0
        vmask = (1 << width) - 1
        while width >= self.brem:
            if self.dpos >= len(self.data):
                raise _IT214DecompressError("unbalanced block end")
            v |= (self.data[self.dpos] >> self.bpos) << vpos
            vpos += self.brem
            width -= self.brem
            self.dpos += 1
            self.brem = 8
            self.bpos = 0
        if width > 0:
            if self.dpos >= len(self.data):
                raise _IT214DecompressError("unbalanced block end")
            v |= (self.data[self.dpos] >> self.bpos) << vpos
            v &= vmask
            self.brem -= width
            self.bpos += width
        return v

    def _write(self, width, value, topbit):
        self.running_count += 1
        self.length -= 1
        v = value
        if v & topbit:
            v -= topbit * 2
        self.unpacked_root = (self.unpacked_root + v) & self.unpack_mask
        self.unpacked_data.append(self.unpacked_root)

    def _change_width(self, width):
        width += 1
        if width >= self.width:
            width += 1
        self.width = width

    def _unpack(self):
        while self.length > 0 and not self._end_of_block():
            if self.width == 0 or self.width > self.widthtop:
                raise _IT214DecompressError("invalid bit width")
            v = self._read(self.width)
            topbit = 1 << (self.width - 1)
            if self.width <= 6:  # MODE A: width-change marker is the sole
                                  # value equal to topbit at this width
                if v == topbit:
                    self._change_width(self._read(self.fetch_a))
                else:
                    self._write(self.width, v, topbit)
            elif self.width < self.widthtop:  # MODE B: a small range of
                                               # values just below topbit
                                               # means "change width"
                if topbit + self.lower_b <= v <= topbit + self.upper_b:
                    self._change_width(v - (topbit + self.lower_b))
                else:
                    self._write(self.width, v, topbit)
            else:  # MODE C (width == widthtop): top bit itself flags a
                   # width change vs. a (always non-negative) literal value
                if v & topbit:
                    self.width = (v & ~topbit) + 1
                else:
                    self._write(self.width - 1, v & ~topbit, 0)


def _it214_decompress_channel(data, pos, num_frames, is16):
    """Decompresses one full sample channel's worth of IT214/IT215-packed
    data (i.e. every consecutive on-disk block for that channel). Returns
    (list_of_unsigned_wrapped_ints, new_pos) - values are still the raw
    reconstructed on-disk byte/word patterns at this point (see
    _it_unsigned_to_signed for the final unsigned/signed interpretation)."""
    values = []
    remaining = num_frames
    maxgrablen = 0x4000 if is16 else 0x8000
    n = len(data)
    while remaining > 0:
        if pos + 2 > n:
            break
        blkcomplen = struct.unpack_from("<H", data, pos)[0]
        pos += 2
        block_len = min(remaining, maxgrablen)
        block_bytes = data[pos:pos + blkcomplen]
        pos += blkcomplen
        dec = _IT214Decompressor(block_bytes, block_len, is16)
        values.extend(dec.unpacked_data)
        remaining -= block_len
    if len(values) < num_frames:
        pad = values[-1] if values else 0
        values.extend([pad] * (num_frames - len(values)))
    return values[:num_frames], pos


def _it_unsigned_to_signed(v, bits, unsigned):
    """Same unsigned<->signed PCM reinterpretation as _decode_s3m_pcm above,
    just operating on already-decompressed ints instead of raw on-disk
    bytes: 'unsigned' on-disk samples need the usual +/-128 (or +/-32768)
    bias flip, while samples already stored as signed two's-complement just
    need reinterpreting into Python's native int range."""
    if bits == 16:
        return (v - 32768) if unsigned else (v - 65536 if v >= 32768 else v)
    return (v - 128) if unsigned else (v - 256 if v >= 128 else v)


def _it_decompress_pcm(data, ptr, num_frames, channels, bits, is_signed_on_disk, is215):
    """Decompresses IT214/IT215 sample data starting at byte offset `ptr`,
    returning sample.data in this module's usual internal convention (8-bit
    values pre-scaled x256, 16-bit values as-is, stereo interleaved) plus
    the file offset right after the last block read."""
    is16 = (bits == 16)
    pos = ptr
    unsigned = not is_signed_on_disk
    chan_values = []
    for _c in range(channels):
        raw_values, pos = _it214_decompress_channel(data, pos, num_frames, is16)
        if is215:
            # IT215 samples are delta-coded a *second* time on top of the
            # IT214 bitstream's own per-sample delta coding (see
            # itmod/OpenMPT's "IT214 + cvt 0x04 (delta) = IT215" note) -
            # each block undoes this independently, matching how it was
            # independently re-applied per block when compressing.
            base = 0
            mask = 0xFFFF if is16 else 0xFF
            undelta = [0] * len(raw_values)
            for i, v in enumerate(raw_values):
                base = (base + v) & mask
                undelta[i] = base
            raw_values = undelta
        chan_values.append([_it_unsigned_to_signed(v, bits, unsigned) for v in raw_values])

    scale = 256 if bits == 8 else 1
    if channels == 2:
        out = [0] * (num_frames * 2)
        left, right = chan_values[0], chan_values[1]
        for i in range(num_frames):
            out[i * 2] = left[i] * scale
            out[i * 2 + 1] = right[i] * scale
        return out, pos
    out = [v * scale for v in chan_values[0]]
    return out, pos


def _it_read_keyboard_table(data, ptr):
    """Reads an IT instrument's 120-entry Note-Sample/Keyboard Table (240
    bytes of (note, sample) pairs at offset 0x40 within the instrument
    header). This offset and layout is identical between the old (cmwt <
    0x200) and new Impulse Instrument Format headers, so - unlike every
    other field in an IT instrument header - it can be read without caring
    which of the two instrument header versions the file actually uses."""
    raw = data[ptr + 0x40: ptr + 0x40 + 240]
    if len(raw) < 240:
        raw = raw + b"\x00" * (240 - len(raw))
    return [(raw[i * 2], raw[i * 2 + 1]) for i in range(120)]


def _it_parse_volume_envelope(data, base):
    """Parses one 82-byte IT envelope structure (new-format instrument
    header only) starting at `base`, laid out per ITTECH.TXT as:
        0x00 Flags (bit0 on, bit1 loop, bit2 sustain-loop)
        0x01 Num (node count, up to 25)
        0x02 LpB / 0x03 LpE     (regular loop, node indices)
        0x04 SLB / 0x05 SLE     (sustain loop, node indices)
        0x06 25 * (value: signed byte 0..64, tick: uint16 LE)
    Returns None if the data is out of range or clearly not a real
    envelope (no points), otherwise an ITVolumeEnvelope."""
    if base + 6 > len(data):
        return None
    env_flags = data[base]
    num = data[base + 1]
    if num == 0 or num > 25:
        return None
    lpb, lpe, slb, sle = data[base + 2], data[base + 3], data[base + 4], data[base + 5]
    points = []
    try:
        for i in range(num):
            off = base + 6 + i * 3
            value = struct.unpack_from("<b", data, off)[0]
            tick = struct.unpack_from("<H", data, off + 1)[0]
            points.append((tick, clamp(value, 0, 64)))
    except struct.error:
        return None
    # Points are supposed to already be tick-ascending in the file, but a
    # damaged/unusual file could violate that; sort defensively since the
    # duration-aware volume code below assumes ascending order, and also
    # force the very first point to tick 0 (matching how IT itself always
    # starts an envelope's playback there regardless of what's on disk).
    points.sort(key=lambda p: p[0])
    points[0] = (0, points[0][1])
    env = ITVolumeEnvelope()
    env.points = points
    env.active = bool(env_flags & 1)
    env.loop = bool(env_flags & 2)
    env.sustain = bool(env_flags & 4)
    env.loop_start = clamp(lpb, 0, num - 1)
    env.loop_end = clamp(lpe, 0, num - 1)
    env.sustain_start = clamp(slb, 0, num - 1)
    env.sustain_end = clamp(sle, 0, num - 1)
    # loop_start == loop_end (or sustain_start == sustain_end) is legal and
    # common - it just means the envelope holds at that single node's value
    # for as long as the loop stays entered; _envelope_average_over_hold
    # (dmm2xm_convert.py) handles that degenerate case directly.
    return env if env.active else None


def _it_parse_pattern(data, ptr):
    """Decompresses one IT pattern's channel-mask-packed cell data. Returns
    (rows, cells) where cells maps (row, 0-based channel) -> a dict with
    whichever of 'note'/'instr'/'vol'/'effect' keys were actually *shown*
    that row (present via a fresh read or the format's own "reuse last
    value for this channel" memory bit) - fields never mentioned this row
    are simply absent from the dict, exactly like every other row/channel
    combination that never appears in `cells` at all represents an
    entirely empty cell. A pointer of 0 means a 64-row empty pattern, per
    the format spec."""
    if ptr == 0:
        return 64, {}
    if ptr + 8 > len(data):
        return 64, {}
    length, rows = struct.unpack_from("<HH", data, ptr)
    rows = clamp(rows, 1, 65535)
    buf = data[ptr + 8: ptr + 8 + length]
    n = len(buf)
    pos = 0
    last_mask = {}
    last_note = {}
    last_instr = {}
    last_vol = {}
    last_effect = {}
    cells = {}
    row = 0
    # Guards against a corrupt/adversarial length field turning this into
    # an effectively infinite loop - same defensive spirit as the
    # zero-padding used elsewhere in this file for malformed MOD/S3M input.
    guard_max = n * 8 + 1000
    guard = 0
    while row < rows and pos < n:
        guard += 1
        if guard > guard_max:
            break
        ch_byte = buf[pos]
        pos += 1
        if ch_byte == 0:
            row += 1
            continue
        ch = (ch_byte - 1) & 63
        if ch_byte & 0x80:
            if pos >= n:
                break
            mask = buf[pos]
            pos += 1
            last_mask[ch] = mask
        else:
            mask = last_mask.get(ch, 0)

        if mask & 0x01:
            if pos >= n:
                break
            last_note[ch] = buf[pos]
            pos += 1
        if mask & 0x02:
            if pos >= n:
                break
            last_instr[ch] = buf[pos]
            pos += 1
        if mask & 0x04:
            if pos >= n:
                break
            last_vol[ch] = buf[pos]
            pos += 1
        if mask & 0x08:
            if pos + 1 >= n:
                break
            last_effect[ch] = (buf[pos], buf[pos + 1])
            pos += 2

        if mask & 0x0F == 0 and mask & 0xF0 == 0:
            continue
        cell = cells.setdefault((row, ch), {})
        if mask & 0x11 and ch in last_note:
            cell["note"] = last_note[ch]
            # The instrument backing this note-trigger is whatever is
            # *currently* selected for this channel, regardless of whether
            # the instrument column happens to be shown this exact row -
            # same "reuse the last selected instrument" semantics as every
            # other tracker format. Keep this separate from "instr" below
            # (which only reflects this row's own mask bits) so a note
            # shown without a fresh/reused instrument column this row still
            # resolves against the right instrument instead of silently
            # losing it.
            if ch in last_instr:
                cell["cur_instr"] = last_instr[ch]
        if mask & 0x22 and ch in last_instr:
            cell["instr"] = last_instr[ch]
        if mask & 0x44 and ch in last_vol:
            cell["vol"] = last_vol[ch]
        if mask & 0x88 and ch in last_effect:
            cell["effect"] = last_effect[ch]
    return rows, cells


def _it_resolve_note_instrument(raw_note, raw_instr, use_instruments, it_keyboard, ins_num):
    """Turns a raw (IT note byte, IT instrument/sample number) pair from a
    pattern cell into a (this-module's-note-numbering, resolved sample
    slot) pair. See the "Impulse Tracker (.it) loading" section docstring
    above for why this resolves the sample explicitly here rather than
    leaving instrument-number passthrough to the generic backward-
    conversion "remembered instrument" fallback used by MOD/S3M/XM.

    Note numbering: IT's raw note byte (0-119, C-0 -> B-9) uses IT's own
    octave numbering, where middle C is "C-5" (IT sample headers even name
    their base-rate field "C5Speed" - "number of bytes a second for C-5").
    MOD/S3M/XM (and this codebase's own shared absolute note numbering,
    inherited from the S3M/MOD loaders above) instead call the very same
    physical pitch "C-4" - one octave lower by NAME only, not by actual
    pitch. So a raw IT note has to be shifted down by a full octave (12
    semitones) when converted into this shared numbering, on top of the
    usual +1 (this codebase's absolute note 1 = C-0): raw_note - 11 rather
    than raw_note + 1. Getting this wrong doesn't just transpose playback
    by an octave - since sample_rate_to_xm()/xm_to_sample_rate() derive
    each sample's *base* playback rate from C5Speed under the same "C-4 is
    the reference note" assumption the whole engine relies on elsewhere,
    a missing shift here means every note plays a full octave higher than
    it should, which is exactly what an un-shifted `raw_note + 1` produces.
    """
    if raw_note is None:
        return None, 0
    if raw_note >= 120:
        # 255 = note off, 254 = note cut, anything else >= 120 is the rare
        # "note fade" pseudo-note (not available in the editor, per
        # ITTECH.TXT) - none of DMM's model distinguishes these, so all map
        # to the one "stop" concept DMM/XM notes have.
        return NOTE_STOP, 0
    if not use_instruments:
        return max(1, raw_note - 11), (raw_instr or 0)
    if raw_instr and 1 <= raw_instr <= ins_num:
        kb = it_keyboard.get(raw_instr)
        mapped_note, mapped_sample = kb[raw_note] if kb else (raw_note, 0)
    else:
        mapped_note, mapped_sample = raw_note, 0
    if mapped_note > 119:
        # A keyboard table entry may itself point past B-9 on a corrupt/
        # malformed instrument - fall back to the untransposed note rather
        # than producing an out-of-range value.
        mapped_note = raw_note
    return max(1, mapped_note - 11), mapped_sample


def _it_resolve_volume(raw_vol, gain):
    """Resolves a raw IT volume-column byte (or None, meaning "not shown
    this row") into a this-codebase "set volume" byte (0x10 + level 0-64),
    scaled by `gain` (see it_module_load: combined song/instrument/sample
    global-volume multiplier). Returns None (as opposed to 0, "explicitly
    silent") when there is nothing to scale, so the caller can tell "no
    volume-column value this row" apart from "volume-column value was 0"."""
    if raw_vol is None:
        return None
    if 0 <= raw_vol <= 64:
        return 0x10 + clamp(round(raw_vol * gain), 0, 64)
    # Panning (mapped onto 128-192) and the various fine-slide/slide/
    # portamento/vibrato volume-column shorthands (65-127, 193-212) have no
    # DMM equivalent and are dropped, exactly like the MOD/S3M loaders drop
    # every pattern command with no DMM equivalent.
    return None


def _it_resolve_effect(raw_effect):
    if raw_effect is None:
        return 0, 0
    cmd, param = raw_effect
    if cmd == IT_EFFECT_SET_SPEED:
        # A00 is ignored by Impulse Tracker (it used to become speed 1).
        return (EFFECT_SPEED_TEMPO, min(param, 31)) if param else (0, 0)
    if cmd == IT_EFFECT_SET_TEMPO:
        # T00-T0F / T10-T1F are tempo *slides*, not tempo values. Only 0x20..0xFF is a tempo
        # (this used to turn every slide into tempo 32 for the rest of the song).
        return (EFFECT_SPEED_TEMPO, param) if param >= 32 else (0, 0)
    if cmd == IT_EFFECT_PATTERN_BREAK:
        return EFFECT_PATTERN_BREAK, 0
    if cmd == IT_EFFECT_S:
        sub, val = (param >> 4) & 0xF, param & 0xF
        if sub == EXT_EFFECT_NOTE_CUT:      # SCx - Note Cut
            return EFFECT_EXTENDED_EFFECTS, val | (EXT_EFFECT_NOTE_CUT << 4)
        if sub == EXT_EFFECT_NOTE_DELAY:    # SDx - Note Delay
            return EFFECT_EXTENDED_EFFECTS, val | (EXT_EFFECT_NOTE_DELAY << 4)
        if sub == EXT_EFFECT_PATTERN_DELAY and val:   # SEx - Pattern Delay (same as XM EEx)
            return EFFECT_EXTENDED_EFFECTS, val | (EXT_EFFECT_PATTERN_DELAY << 4)
        return 0, 0
    # All other commands (arpeggio, slides, vibrato, position jump, pattern
    # loop, other S sub-commands, ...) have no DMM equivalent and are
    # dropped - same as the MOD/S3M loaders.
    return 0, 0


def it_module_load(path):
    """Loads an Impulse Tracker .it file into an XMModule. Returns None if
    the file isn't a recognisable IT module. Every IT *sample* becomes one
    XMModule instrument slot (matching DMM's own one-sample-per-instrument
    model - see the module docstring above for why, in "use instruments"
    mode, pattern notes are resolved to a specific sample at load time
    instead of just carrying the raw IT instrument number through)."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 0xC0 or data[0:4] != b"IMPM":
        return None
    songname = data[4:30]
    try:
        ordnum, insnum, smpnum, patnum, cwtv, cmwt, flags, special = struct.unpack_from("<8H", data, 32)
        gv, mv, ispd, itpo = data[48], data[49], data[50], data[51]
    except (struct.error, IndexError):
        return None
    if insnum > 0xFF or smpnum > 0xFF or patnum > 0xFF:
        return None

    pos = 0xC0
    order_bytes = data[pos:pos + ordnum]
    pos += ordnum
    try:
        ins_ptrs = struct.unpack_from("<%dI" % insnum, data, pos) if insnum else ()
        pos += insnum * 4
        smp_ptrs = struct.unpack_from("<%dI" % smpnum, data, pos) if smpnum else ()
        pos += smpnum * 4
        pat_ptrs = struct.unpack_from("<%dI" % patnum, data, pos) if patnum else ()
    except struct.error:
        return None

    use_instruments = bool(flags & 0x04)

    pattern_order = []
    for b in order_bytes:
        if b == 255:  # "---" end-of-song marker
            break
        if b == 254:  # "+++" skip marker, not a real pattern
            continue
        pattern_order.append(b)
    track_length = len(pattern_order)

    xm = XMModule()
    xm.name = songname.split(b"\x00", 1)[0].decode("cp1251", errors="replace")
    xm.track_length = track_length
    xm.restart_position = 0  # IT has no dedicated restart-position field
    xm.pattern_order = pattern_order + [0] * (256 - len(pattern_order))
    xm.linear_slides = bool(flags & 0x08)
    xm.start_speed = ispd or 6
    xm.start_tempo = itpo or 125

    it_keyboard = {}
    it_instrument_gain = {}
    if use_instruments:
        for i, ptr in enumerate(ins_ptrs, start=1):
            if ptr == 0:
                continue
            it_keyboard[i] = _it_read_keyboard_table(data, ptr)
            # GbV (per-instrument Global Volume, 0->128) only exists in the
            # "new" instrument header (cmwt >= 0x200); the old format has
            # no such field and every instrument implicitly means "128",
            # i.e. no attenuation.
            if cmwt >= 0x200 and ptr + 0x19 <= len(data):
                it_instrument_gain[i] = clamp(data[ptr + 0x18], 0, 128) / 128.0
            else:
                it_instrument_gain[i] = 1.0
            # Volume envelope, likewise only present (in the node-based
            # form parsed here) on new-format instrument headers. Old-format
            # (cmwt < 0x200) instruments use a different, much simpler
            # envelope layout that isn't handled here - they're left with
            # no envelope (same as before this was added), which just
            # means their note volume is used as-is, unscaled.
            if cmwt >= 0x200:
                env = _it_parse_volume_envelope(data, ptr + 0x130)
                if env is not None:
                    xm.it_volume_envelopes[i] = env

    song_gain = clamp(gv, 0, 128) / 128.0

    samples = [None] * (smpnum + 1)  # 1-based, like xm.instruments
    sample_gain = [1.0] * (smpnum + 1)
    for i, ptr in enumerate(smp_ptrs, start=1):
        inst = Instrument()
        inst.fade_out = 65536
        inst.volume_envelope.active = False
        inst.panning_envelope.active = False
        smp = Sample()
        smp.panning = 128
        if ptr != 0 and ptr + 0x50 <= len(data):
            gvl = data[ptr + 17]
            flg = data[ptr + 18]
            vol = data[ptr + 19]
            name_b = data[ptr + 20:ptr + 46]
            cvt = data[ptr + 46]
            try:
                length, lpbeg, lpend, c5speed, susbeg, susend, smpptr = \
                    struct.unpack_from("<7I", data, ptr + 0x30)
            except struct.error:
                length = 0
            smp.name = name_b.split(b"\x00", 1)[0].decode("cp1251", errors="replace")
            smp.volume = clamp(vol, 0, 64)
            # GvL (per-*sample* Global Volume, 0->64 - despite the
            # confusingly reused name, this is a separate multiplier from
            # the per-instrument GbV above) attenuates every note played
            # through this sample, not just the "no explicit volume this
            # row" default. Real .it files commonly set this below 64 to
            # balance a specific sample into the mix - dropping it entirely
            # would make exactly those samples play back too loud relative
            # to the rest of the track.
            sample_gain[i] = clamp(gvl, 0, 64) / 64.0
            bits = 16 if (flg & 2) else 8
            channels = 2 if (flg & 4) else 1
            smp.bits = bits
            smp.channels = channels
            smp.relative_note, smp.finetune = sample_rate_to_xm(c5speed, False)
            if (flg & 1) and length > 0:
                compressed = bool(flg & 8)
                is215 = bool(cvt & 4)
                is_signed = bool(cvt & 1)
                if compressed:
                    if smpptr < len(data):
                        pcm, _endpos = _it_decompress_pcm(
                            data, smpptr, length, channels, bits, is_signed, is215)
                        smp.data = pcm
                        smp.sample_length = length
                else:
                    bytes_per_frame = (2 if bits == 16 else 1) * channels
                    raw = data[smpptr:smpptr + length * bytes_per_frame]
                    smp.data = _decode_s3m_pcm(raw, length, channels, bits, unsigned=not is_signed)
                    smp.sample_length = length
                # DMM notes are never "released" (there is no note-off, only cut/stop), so while a
                # note sounds an IT sample is always in its *sustain* state: the sustain loop, if
                # enabled, takes precedence over the normal loop (as it does in Impulse Tracker for
                # a held note). Previously susbeg/susend were read but never used.
                if smp.sample_length and (flg & 0x20) and susend > susbeg and susbeg < smp.sample_length:
                    smp.loop_start = susbeg
                    smp.loop_length = min(susend, smp.sample_length) - susbeg
                    smp.loop = smp.loop_length > 0
                    smp.pingpong_loop = bool(flg & 0x80)
                elif smp.sample_length and (flg & 0x10) and lpend > lpbeg:
                    smp.loop_start = lpbeg
                    smp.loop_length = min(lpend, smp.sample_length) - lpbeg
                    smp.loop = smp.loop_length > 0
                    smp.pingpong_loop = bool(flg & 0x40)
        inst.sample[0] = smp
        xm.instruments[i] = inst
        samples[i] = smp

    # Channel Pan byte: bit 7 set = channel disabled/muted in the tracker; its notes must not be
    # converted (nor take part in "loudest channels" selection).
    muted = [bool(data[0x40 + i] & 0x80) for i in range(64)]

    parsed = []
    max_channel = -1
    for ptr in pat_ptrs:
        rows, cells = _it_parse_pattern(data, ptr)
        for key in [k for k in cells if muted[k[1]]]:
            del cells[key]
        parsed.append((rows, cells))
        for (_r, ch) in cells:
            if ch > max_channel:
                max_channel = ch

    # The channel's selected instrument persists across pattern boundaries (the parser above
    # only remembers it within one pattern). Walk the patterns in play order and hand the
    # remembered instrument to notes that show no instrument column.
    chan_instr = {}
    seen_patterns = set()
    for p in list(pattern_order) + list(range(len(parsed))):
        if p in seen_patterns or p >= len(parsed):
            continue
        seen_patterns.add(p)
        for (r, ch) in sorted(parsed[p][1]):
            f = parsed[p][1][(r, ch)]
            if "instr" in f:
                chan_instr[ch] = f["instr"]
            elif "note" in f and "cur_instr" not in f and ch in chan_instr:
                f["cur_instr"] = chan_instr[ch]

    # Song loop: a Bxx (position jump) in the first-executed row of the last pattern that jumps
    # backwards is what makes an IT song loop. IT has no restart field, so it was always 0.
    if pattern_order:
        raw_to_compact = {}
        n_valid = 0
        for i, b in enumerate(order_bytes):
            if b == 255:
                break
            raw_to_compact[i] = n_valid
            if b != 254:
                n_valid += 1
        last_cells = parsed[pattern_order[-1]][1] if pattern_order[-1] < len(parsed) else {}
        jumps = sorted((r, ch, f["effect"][1]) for (r, ch), f in last_cells.items()
                       if "effect" in f and f["effect"][0] == 2)
        if jumps:
            target = min(raw_to_compact.get(jumps[0][2], track_length - 1), track_length - 1)
            if target < track_length - 1:
                # The value after DMM's 255 marker is an order index (engine: curseq = seq[i+1]),
                # which is exactly what XM's restart position is too.
                xm.restart_position = target
    number_of_channels = clamp(max_channel + 1, 1, 64) if max_channel >= 0 else 1
    xm.number_of_channels = number_of_channels

    for p, (rows, cells) in enumerate(parsed):
        pat = xm.patterns[p]
        pat.resize(max(rows, 1), number_of_channels)
        for (r, ch), fields in cells.items():
            if ch >= number_of_channels:
                continue
            note = PatternNote()
            has_something = False
            mapped_sample = 0
            note_gain = song_gain
            if "note" in fields:
                raw_instr = fields.get("cur_instr")
                note.keep_volume = "instr" not in fields
                mapped_note, mapped_sample = _it_resolve_note_instrument(
                    fields["note"], raw_instr, use_instruments, it_keyboard, insnum)
                if mapped_note is not None:
                    note.note = mapped_note
                    if mapped_sample:
                        note.instrument = mapped_sample
                        note_gain *= sample_gain[mapped_sample] if mapped_sample < len(sample_gain) else 1.0
                        if use_instruments and raw_instr:
                            note_gain *= it_instrument_gain.get(raw_instr, 1.0)
                            note.it_instrument = raw_instr
                    has_something = True
            if "vol" in fields:
                vol_byte = _it_resolve_volume(fields["vol"], note_gain)
                if vol_byte is not None:
                    note.volume = vol_byte
                v = fields["vol"]
                # volume-column slides: 65-74 fine up, 75-84 fine down, 85-94 up, 95-104 down
                if 65 <= v <= 74 and v > 65:
                    note.vol_slide = (0, v - 65)
                elif 75 <= v <= 84 and v > 75:
                    note.vol_slide = (0, -(v - 75))
                elif 85 <= v <= 94 and v > 85:
                    note.vol_slide = (v - 85, 0)
                elif 95 <= v <= 104 and v > 95:
                    note.vol_slide = (-(v - 95), 0)
                has_something = True
            if "effect" in fields:
                note.effect, note.effect_parameter = _it_resolve_effect(fields["effect"])
                if note.effect:
                    has_something = True
                if fields["effect"][0] in (4, 11, 12):        # D, K, L
                    note.vol_slide = combine_vol_slides(note.vol_slide,
                                                        vol_slide_from_s3m_param(fields["effect"][1]))
                elif fields["effect"][0] == 17:               # Q - retrigger note (+ volume change)
                    note.retrig = retrig_from_param(fields["effect"][1])
                elif fields["effect"][0] == 15:               # O - sample offset
                    note.sample_offset = sample_offset_from_param(fields["effect"][1])
            if note.vol_slide is not None or note.retrig is not None or note.sample_offset is not None:
                has_something = True
            # Same default-volume behaviour as mod_module_load/s3m_module_load:
            # a note triggered without an explicit volume/panning this row
            # falls back to the resolved sample's own default volume (here
            # additionally scaled by the same song/instrument/sample gain
            # as an explicit volume would have been), rather than being
            # read as volume 0 (silence) or left for a downstream "no
            # volume set" default that knows nothing about this sample.
            if (note.note and note.instrument and note.volume == 0 and note.effect != EFFECT_VOLUME
                    and not note.keep_volume):
                smp = samples[note.instrument] if note.instrument < len(samples) else None
                if smp is not None:
                    note.volume = 0x10 + clamp(round(smp.volume * note_gain), 0, 64)
            if has_something:
                pat.set_note(r, ch, note)

    return xm


def load_module(path):
    """Load a modern .xm file, a Scream Tracker 3 .s3m file, an Impulse
    Tracker .it file, or an Amiga .mod file. Returns an XMModule, or None
    if the file could not be recognised/loaded. Format is sniffed from
    content, not the extension."""
    with open(path, "rb") as f:
        head = f.read(0x30)
    if head[0:17] == b"Extended Module: ":
        return xm_module_load(path)
    if len(head) >= 0x30 and head[0x2C:0x30] == b"SCRM":
        return s3m_module_load(path)
    if head[0:4] == b"IMPM":
        return it_module_load(path)
    return mod_module_load(path)


# ---------------------------------------------------------------------------
# XM saving (used by forward DMM/WAD -> XM conversion)
# ---------------------------------------------------------------------------

def xm_module_save(xm, path):
    n_patterns = xm.number_of_patterns()
    n_instruments = xm.number_of_instruments()

    name_b = xm.name.encode("cp1251", errors="replace")[:20].ljust(20, b"\x00")
    tracker_b = b"FastTracker v2.00   "[:20].ljust(20, b"\x00")
    flags = (1 if xm.linear_slides else 0) | (0x1000 if xm.extended_filter_range else 0)
    header = struct.pack(
        "<17s20sB20sHI8H",
        b"Extended Module: ", name_b, 0x1A, tracker_b, 0x0104, 276,
        xm.track_length, xm.restart_position, xm.number_of_channels,
        n_patterns, n_instruments, flags, xm.start_speed, xm.start_tempo,
    )
    header += bytes(xm.pattern_order[:256]).ljust(256, b"\x00")

    buf = bytearray()
    buf += header

    for p in range(n_patterns):
        pattern = xm.patterns[p]
        pdata = bytearray()
        if not pattern.is_empty():
            for row in range(pattern.rows):
                for ch in range(pattern.channels):
                    note = pattern.get_note(row, ch)
                    if (note.note != 0 and note.instrument != 0 and note.volume > 0xF
                            and (note.effect != EFFECT_ARPEGGIO or note.effect_parameter != 0)):
                        pdata += bytes([note.note & 0xFF, note.instrument & 0xFF,
                                        note.volume & 0xFF, note.effect & 0xFF,
                                        note.effect_parameter & 0xFF])
                    else:
                        f = 0x80
                        if note.note != 0:
                            f |= 1
                        if note.instrument != 0:
                            f |= 2
                        if note.volume > 0xF:
                            f |= 4
                        if note.effect != EFFECT_ARPEGGIO:
                            f |= 8
                        if note.effect_parameter != 0:
                            f |= 16
                        pdata.append(f)
                        if f & 1:
                            pdata.append(note.note & 0xFF)
                        if f & 2:
                            pdata.append(note.instrument & 0xFF)
                        if f & 4:
                            pdata.append(note.volume & 0xFF)
                        if f & 8:
                            pdata.append(note.effect & 0xFF)
                        if f & 16:
                            pdata.append(note.effect_parameter & 0xFF)
        buf += struct.pack("<IBHH", 9, 0, pattern.rows, len(pdata))
        buf += pdata

    for counter in range(1, n_instruments + 1):
        inst = xm.instruments.get(counter)
        n_samples = 0
        if inst is not None:
            for sc in range(256):
                if inst.sample.get(sc) is not None:
                    n_samples = sc + 1
        ihdr_size = (29 + 214) if n_samples > 0 else 29
        iname = inst.name.encode("cp1251", errors="replace")[:21] if inst is not None else b""
        iname = iname.ljust(22, b"\x00")
        buf += struct.pack("<I22sBH", ihdr_size, iname, 0, n_samples)

        if n_samples > 0:
            sample_map = bytes(inst.sample_map[1:97]).ljust(96, b"\x00") if len(inst.sample_map) >= 97 \
                else bytes(96)
            ve_num = min(inst.volume_envelope.number_of_points, 12)
            pe_num = min(inst.panning_envelope.number_of_points, 12)
            vol_env_words, pan_env_words = [], []
            for sc in range(12):
                t, v = inst.volume_envelope.points[sc]
                vol_env_words += [t & 0xFFFF, v & 0xFFFF]
                t2, v2 = inst.panning_envelope.points[sc]
                pan_env_words += [t2 & 0xFFFF, v2 & 0xFFFF]
            vol_flags = ((1 if inst.volume_envelope.active else 0)
                         | (2 if inst.volume_envelope.sustain else 0)
                         | (4 if inst.volume_envelope.loop else 0))
            pan_flags = ((1 if inst.panning_envelope.active else 0)
                         | (2 if inst.panning_envelope.sustain else 0)
                         | (4 if inst.panning_envelope.loop else 0))
            buf += struct.pack(
                "<I96s24H24H14BHH",
                40, sample_map, *vol_env_words, *pan_env_words,
                ve_num, pe_num,
                inst.volume_envelope.sustain_point, inst.volume_envelope.loop_start_point,
                inst.volume_envelope.loop_end_point,
                inst.panning_envelope.sustain_point, inst.panning_envelope.loop_start_point,
                inst.panning_envelope.loop_end_point,
                vol_flags, pan_flags,
                inst.vibrato_type, inst.vibrato_sweep, inst.vibrato_depth, inst.vibrato_rate,
                inst.fade_out, 0,
            )
            samples_list = [inst.sample.get(sc) for sc in range(n_samples)]
            for smp in samples_list:
                shdr = bytearray(40)
                if smp is not None:
                    sample_length, loop_start, loop_length = smp.sample_length, smp.loop_start, smp.loop_length
                    sflags = 0
                    if smp.pingpong_loop:
                        sflags |= 2
                    elif smp.loop:
                        sflags |= 1
                    if smp.bits == 16:
                        sflags |= 16
                        sample_length *= 2; loop_start *= 2; loop_length *= 2
                    if smp.channels == 2:
                        sflags |= 32
                        sample_length *= 2; loop_start *= 2; loop_length *= 2
                    name_bytes = smp.name.encode("cp1251", errors="replace")[:22].ljust(22, b"\x00")
                    struct.pack_into(
                        "<3IBbBBbB22s", shdr, 0,
                        sample_length, loop_start, loop_length,
                        smp.volume, clamp(smp.finetune, -128, 127), sflags, smp.panning,
                        clamp(smp.relative_note, -128, 127), 0, name_bytes,
                    )
                buf += shdr
            for smp in samples_list:
                if smp is None:
                    continue
                if smp.channels == 1:
                    buf += encode_8bit(smp.data) if smp.bits == 8 else encode_16bit(smp.data)
                else:
                    left = smp.data[0::2]
                    right = smp.data[1::2]
                    if smp.bits == 8:
                        buf += encode_8bit(left)
                        buf += encode_8bit(right)
                    else:
                        buf += encode_16bit(left)
                        buf += encode_16bit(right)

    if xm.comment:
        comment = xm.comment.replace("\n", "")
        cb = comment.encode("cp1251", errors="replace")
        buf += b"text" + struct.pack("<I", len(cb)) + cb

    ensure_dir_for_file(path)
    with open(path, "wb") as f:
        f.write(buf)


# ---------------------------------------------------------------------------
# XM loading (used by backward XM -> DMM conversion). Supports the full
# range of on-disk XM format versions the original BeRoXM handled:
#   version <= 0x0102 : old (8-byte) pattern header, deferred sample data
#   version == 0x0103 : new (9-byte) pattern header, but still deferred
#                        sample data (instrument headers -> patterns ->
#                        one block of raw sample data for every instrument)
#   version  > 0x0103 : "modern" layout used by FastTracker II onwards
#                        (patterns -> each instrument's headers+data)
# ---------------------------------------------------------------------------

MAX_SAMPLE_SIZE = 0x40000000


class _ByteReader:
    """Sequential little-endian byte reader that zero-pads past EOF, mirroring
    the original tool's stream Read() (which zero-fills any bytes it could
    not actually read instead of raising an error)."""

    def __init__(self, data):
        self.data = data
        self.pos = 0

    def read(self, n):
        chunk = self.data[self.pos:self.pos + n]
        if len(chunk) < n:
            chunk = chunk + b"\x00" * (n - len(chunk))
        self.pos += n
        return chunk

    def read_byte(self):
        return self.read(1)[0]

    def seek(self, pos):
        self.pos = pos


def _sample_frames_from_disk_length(disk_length, bits, channels):
    """Converts the on-disk byte-count convention used by the XM sample
    header's SampleLength/LoopStart/LoopLength fields into the logical
    per-channel sample-frame count."""
    frames = disk_length
    if bits == 16:
        frames //= 2
    if channels == 2:
        frames //= 2
    return frames


def _read_sample_pcm(reader, smp):
    """Reads and decodes one sample's raw PCM data, applying the same
    ADPCM4 / bit-depth / stereo-channel handling and length/loop-point
    conversions as the original LoadInstrumentSamplesData / Read8BitSample /
    Read16BitSample / ReadADPCM4Sample."""
    disk_length = smp.sample_length
    if disk_length > MAX_SAMPLE_SIZE:
        disk_length = MAX_SAMPLE_SIZE
        if smp.loop_start > disk_length:
            smp.loop_start = 0
            smp.loop = False

    if disk_length <= 0:
        smp.sample_length = 0
        smp.data = []
        return

    if smp.adpcm4:
        # ADPCM4 is only ever used for 8-bit mono samples; disk_length here
        # is already the logical (final) sample count, not a byte count.
        smp.data, reader.pos = _read_adpcm4_sample_mono(reader.data, reader.pos, disk_length)
        smp.sample_length = disk_length
        return

    raw = reader.read(disk_length)
    frames = _sample_frames_from_disk_length(disk_length, smp.bits, smp.channels)
    if smp.bits == 16:
        smp.data = decode_16bit(raw, frames, smp.channels)
    else:
        smp.data = decode_8bit(raw, frames, smp.channels)
    smp.sample_length = frames
    smp.loop_start = _sample_frames_from_disk_length(smp.loop_start, smp.bits, smp.channels)
    smp.loop_length = _sample_frames_from_disk_length(smp.loop_length, smp.bits, smp.channels)


def _load_pattern_data(reader, pattern, rows, channels):
    for row in range(rows):
        for ch in range(channels):
            b0 = reader.read_byte()
            note = PatternNote()
            if (b0 & 128) == 0:
                note.note = b0
                note.instrument = reader.read_byte()
                note.volume = reader.read_byte()
                note.effect = reader.read_byte()
                note.effect_parameter = reader.read_byte()
            else:
                if b0 & 1:
                    note.note = reader.read_byte()
                if b0 & 2:
                    note.instrument = reader.read_byte()
                if b0 & 4:
                    note.volume = reader.read_byte()
                if b0 & 8:
                    note.effect = reader.read_byte()
                if b0 & 16:
                    note.effect_parameter = reader.read_byte()
            if note.instrument == 0xFF:
                note.instrument = 0
            # XM has no separate "IT instrument" concept: the instrument number IS the envelope
            # key (see xm_envelope_to_it_style / xm.it_volume_envelopes, populated in
            # _load_instruments). Mirroring it here lets the existing IT-only envelope lookup in
            # dmm2xm_convert.py find it for XM notes too, the same way it already does for IT.
            note.it_instrument = note.instrument
            vs = vol_slide_from_xm_effect(note.effect, note.effect_parameter, True)
            vcol = note.volume
            if 0x60 <= vcol <= 0x6F and vcol & 15:
                vs = combine_vol_slides(vs, (-(vcol & 15), 0))
            elif 0x70 <= vcol <= 0x7F and vcol & 15:
                vs = combine_vol_slides(vs, (vcol & 15, 0))
            elif 0x80 <= vcol <= 0x8F and vcol & 15:
                vs = combine_vol_slides(vs, (0, -(vcol & 15)))
            elif 0x90 <= vcol <= 0x9F and vcol & 15:
                vs = combine_vol_slides(vs, (0, vcol & 15))
            note.vol_slide = vs
            note.retrig = retrig_from_xm_effect(note.effect, note.effect_parameter)
            if note.effect == 0x9:
                note.sample_offset = sample_offset_from_param(note.effect_parameter)
            pattern.set_note(row, ch, note)


def _load_patterns(reader, xm, n_patterns, version):
    for p in range(n_patterns):
        if version <= 0x0102:
            phdr = reader.read(8)
            rows = phdr[5] + 1
            packed_size = struct.unpack_from("<H", phdr, 6)[0]
        else:
            phdr = reader.read(9)
            rows = struct.unpack_from("<H", phdr, 5)[0]
            packed_size = struct.unpack_from("<H", phdr, 7)[0]
        xm.patterns[p].resize(rows, xm.number_of_channels)
        if packed_size > 0:
            next_pos = reader.pos + packed_size
            if rows > 1024 or rows == 0:
                reader.seek(next_pos)
                continue
            _load_pattern_data(reader, xm.patterns[p], rows, xm.number_of_channels)
            reader.seek(next_pos)


def _parse_instrument_extra_header(exhdr, inst):
    base = 4 + 96 + 48 + 48
    sample_map = list(exhdr[4:4 + 96])
    vol_env_words = struct.unpack_from("<24H", exhdr, 4 + 96)
    pan_env_words = struct.unpack_from("<24H", exhdr, 4 + 96 + 48)
    ve_points = [[vol_env_words[sc * 2], vol_env_words[sc * 2 + 1] & 0xFF] for sc in range(12)]
    pe_points = [[pan_env_words[sc * 2], pan_env_words[sc * 2 + 1] & 0xFF] for sc in range(12)]
    for sc in range(1, 12):
        if ve_points[sc][0] < ve_points[sc - 1][0]:
            ve_points[sc][0] = ve_points[sc][0] & 0xFF
            ve_points[sc][0] += ve_points[sc - 1][0] & 0xFF00
            if ve_points[sc][0] < ve_points[sc - 1][0]:
                ve_points[sc][0] += 0x100
        if pe_points[sc][0] < pe_points[sc - 1][0]:
            pe_points[sc][0] = pe_points[sc][0] & 0xFF
            pe_points[sc][0] += pe_points[sc - 1][0] & 0xFF00
            if pe_points[sc][0] < pe_points[sc - 1][0]:
                pe_points[sc][0] += 0x100
    (ve_num, pe_num, ve_sus, ve_ls, ve_le, pe_sus, pe_ls, pe_le,
     vol_flags, pan_flags, vtype, vsweep, vdepth, vrate) = struct.unpack_from("<14B", exhdr, base)
    fade_out = struct.unpack_from("<H", exhdr, base + 14)[0]

    inst.fade_out = fade_out
    inst.sample_map = [0] + sample_map
    inst.volume_envelope.points = [tuple(pt) for pt in ve_points]
    inst.panning_envelope.points = [tuple(pt) for pt in pe_points]
    inst.volume_envelope.number_of_points = ve_num
    inst.volume_envelope.loop_start_point = ve_ls
    inst.volume_envelope.loop_end_point = ve_le
    inst.volume_envelope.sustain_point = ve_sus
    inst.panning_envelope.number_of_points = pe_num
    inst.panning_envelope.loop_start_point = pe_ls
    inst.panning_envelope.loop_end_point = pe_le
    inst.panning_envelope.sustain_point = pe_sus
    inst.volume_envelope.active = bool(vol_flags & 1)
    inst.volume_envelope.sustain = bool(vol_flags & 2)
    inst.volume_envelope.loop = bool(vol_flags & 4)
    inst.panning_envelope.active = bool(pan_flags & 1)
    inst.panning_envelope.sustain = bool(pan_flags & 2)
    inst.panning_envelope.loop = bool(pan_flags & 4)
    if inst.volume_envelope.number_of_points == 0:
        inst.volume_envelope.active = False
    if inst.panning_envelope.number_of_points == 0:
        inst.panning_envelope.active = False
    if inst.volume_envelope.sustain_point > 12:
        inst.volume_envelope.sustain = False
    if inst.panning_envelope.sustain_point > 12:
        inst.panning_envelope.sustain = False
    if inst.volume_envelope.loop_start_point >= inst.volume_envelope.loop_end_point:
        inst.volume_envelope.loop = False
    if inst.panning_envelope.loop_start_point >= inst.panning_envelope.loop_end_point:
        inst.panning_envelope.loop = False
    inst.vibrato_type, inst.vibrato_sweep = vtype, vsweep
    inst.vibrato_depth, inst.vibrato_rate = vdepth, vrate


def _load_instruments(reader, xm, n_instruments, defer_sample_data):
    """Reads all instrument headers (+ sample headers). If defer_sample_data
    is False, also reads each instrument's raw sample data immediately
    (the "modern", version > 0x0103 on-disk layout). If True, sample data is
    *not* read here - the caller must call _load_deferred_sample_data()
    afterwards (the old, version <= 0x0103 on-disk layout, where all raw
    sample bytes for every instrument come in one block after the patterns).

    The 4-byte "instrument size" field at the start of each instrument's
    header is the ONLY reliable source for how many bytes the (base header +
    "extended"/envelope header) block actually occupies on disk - real-world
    files disagree on this (234 bytes of envelope/keymap data is the common
    "modern" FT2 case, giving instrument size 263, but older/simpler writers
    can legitimately use a shorter extended block). Likewise, the "sample
    header size" field at the very start of that extended block is the only
    reliable source for how many bytes each of the instrument's sample
    headers occupies. Trusting fixed constants for either of these two sizes
    silently desyncs the reader for any file that doesn't happen to match
    the assumed size exactly, corrupting every envelope/sample-header/
    instrument read afterwards for the rest of the file."""
    pending = []  # list of (instrument_index) with samples still to be read, in order
    for i in range(1, n_instruments + 1):
        ihdr = reader.read(29)
        isize, iname_b, itype, cnt = struct.unpack("<I22sBH", ihdr)
        header_start = reader.pos - 29
        if isize < 29:
            isize = 29
        inst = Instrument()
        inst.name = iname_b.split(b"\x00", 1)[0].decode("cp1251", errors="replace")
        if cnt == 0:
            reader.seek(header_start + isize)
            xm.instruments[i] = inst
            continue

        extended_size = isize - 29
        exhdr_raw = reader.read(extended_size)
        # Pad defensively so _parse_instrument_extra_header can safely unpack
        # its fixed field offsets (up to 234 bytes) even for files whose
        # extended header is shorter (older/simpler envelope-less layouts);
        # any padding bytes just decode as zeroed/absent envelope data.
        exhdr = exhdr_raw.ljust(234, b"\x00")
        _parse_instrument_extra_header(exhdr, inst)
        # Trust the file's own declared instrument size for where the
        # sample headers actually start, regardless of extended_size.
        reader.seek(header_start + isize)

        sample_header_size = struct.unpack_from("<I", exhdr, 0)[0]
        if sample_header_size <= 0:
            sample_header_size = 40

        samples = []
        for sc in range(cnt):
            shdr_start = reader.pos
            shdr_raw = reader.read(min(sample_header_size, 40))
            shdr = shdr_raw.ljust(40, b"\x00")
            slen, lstart, llen, vol, ftune, sflags, pan, relnote, reserved = \
                struct.unpack_from("<3IBbBBbB", shdr, 0)
            sname = shdr[18:40].split(b"\x00", 1)[0].decode("cp1251", errors="replace")
            smp = Sample()
            smp.name = sname
            smp.sample_length = slen
            smp.loop_start = lstart
            smp.loop_length = llen
            smp.volume = vol
            smp.finetune = ftune
            smp.relative_note = relnote
            smp.panning = pan
            smp.bits = 16 if (sflags & 16) else 8
            smp.channels = 2 if (sflags & 32) else 1
            smp.loop = bool(sflags & 3)
            smp.pingpong_loop = bool(sflags & 2)
            smp.adpcm4 = (reserved == 0xAD) and ((sflags & 0x30) == 0)
            inst.sample[sc] = smp
            samples.append(smp)
            # Resync per-sample too, in case sample_header_size differs from
            # the 40 bytes we actually parsed (e.g. a file declaring extra
            # reserved bytes per sample header).
            reader.seek(shdr_start + sample_header_size)

        if defer_sample_data:
            pending.append(i)
        else:
            for smp in samples:
                _read_sample_pcm(reader, smp)
        env = xm_envelope_to_it_style(inst.volume_envelope)
        if env is not None:
            xm.it_volume_envelopes[i] = env
        xm.instruments[i] = inst

    return pending


def _load_deferred_sample_data(reader, xm, pending_instrument_indices):
    for i in pending_instrument_indices:
        inst = xm.instruments.get(i)
        if inst is None:
            continue
        for sc in sorted(inst.sample.keys()):
            _read_sample_pcm(reader, inst.sample[sc])


def xm_module_load(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 80 or data[0:17] != b"Extended Module: ":
        return None  # not an XM file

    name_b = data[17:37]
    version = struct.unpack_from("<H", data, 58)[0]
    size = struct.unpack_from("<I", data, 60)[0]
    trlen, restart, nchan, npat, ninst, flags, speed, tempo = struct.unpack_from("<8H", data, 64)
    if nchan > 255 or ninst > 255 or nchan == 0:
        return None

    pattern_order = list(data[80:80 + 256])
    pattern_order += [0] * (256 - len(pattern_order))

    xm = XMModule()
    xm.name = name_b.split(b"\x00", 1)[0].decode("cp1251", errors="replace")
    xm.track_length = trlen
    xm.restart_position = restart
    xm.number_of_channels = nchan
    xm.linear_slides = bool(flags & 0x1)
    xm.extended_filter_range = bool(flags & 0x1000)
    xm.start_speed = speed or 6
    xm.start_tempo = tempo or 125
    xm.pattern_order = pattern_order

    reader = _ByteReader(data)
    reader.seek(60 + size)  # right after the (variable-size, but normally 276) header block

    if version <= 0x0103:
        pending = _load_instruments(reader, xm, ninst, defer_sample_data=True)
        _load_patterns(reader, xm, npat, version)
        _load_deferred_sample_data(reader, xm, pending)
    else:
        _load_patterns(reader, xm, npat, version)
        _load_instruments(reader, xm, ninst, defer_sample_data=False)

    _expand_xm_multisample_instruments(xm)
    _apply_xm_default_volumes(xm)
    return xm


def _expand_xm_multisample_instruments(xm):
    """Splits XM instruments whose note->sample keyboard map actually sends
    played notes to several different samples into one slot per used sample
    (like it_module_load already does for IT drum kits), rewriting those
    notes' instrument numbers. An instrument whose played notes all hit a
    single non-zero sample just gets that sample repointed to sample[0] in
    place, keeping its slot number.

    Without this, backward conversion (which is one-sample-per-instrument
    like DMM itself) silently plays sample[0] for every note: BGM12.XM
    instrument 19 maps everything to sample[1] (19 KB of audio, default
    volume 37) while its sample[0] is empty, so all of its notes converted
    to silence even though FT2/OpenMPT clearly plays them. Single-sample
    instruments (sample map all zeros) are left byte-for-byte untouched.
    """
    for inst_no in sorted(xm.instruments):
        inst = xm.instruments[inst_no]
        if len(inst.sample) <= 1:
            continue
        used = {}  # sample_idx -> [(pattern, row, channel)]
        for p in range(xm.number_of_patterns()):
            pat = xm.patterns[p]
            for row in range(pat.rows):
                for ch in range(pat.channels):
                    n = pat.get_note(row, ch)
                    if n.instrument != inst_no or not n.note or n.note == NOTE_STOP:
                        continue
                    idx = inst.sample_map[n.note] if 0 <= n.note < len(inst.sample_map) else 0
                    if idx not in inst.sample:
                        idx = 0
                    if idx not in inst.sample:
                        continue  # broken map entry: leave the cell as-is (silent, as before)
                    used.setdefault(idx, []).append((p, row, ch))
        if not used:
            continue
        first = min(used)
        by_sample = {s: inst.sample[s] for s in used}
        # Normalize the keeper slot to a plain single-sample instrument.
        inst.sample = {0: by_sample[first]}
        inst.sample_map = [0] * 97
        for other in sorted(s for s in used if s != first):
            nxt = max(xm.instruments) + 1
            ni = Instrument()
            ni.name = inst.name
            ni.sample_map = [0] * 97
            ni.sample = {0: by_sample[other]}
            ni.fade_out = inst.fade_out
            ni.vibrato_type = inst.vibrato_type
            ni.vibrato_sweep = inst.vibrato_sweep
            ni.vibrato_depth = inst.vibrato_depth
            ni.vibrato_rate = inst.vibrato_rate
            ni.volume_envelope = copy.copy(inst.volume_envelope)
            ni.panning_envelope = copy.copy(inst.panning_envelope)
            xm.instruments[nxt] = ni
            if inst_no in xm.it_volume_envelopes:
                xm.it_volume_envelopes[nxt] = xm.it_volume_envelopes[inst_no]
            for (p, row, ch) in used[other]:
                cell = xm.patterns[p].get_note(row, ch)
                cell.instrument = nxt
                cell.it_instrument = nxt
                xm.patterns[p].set_note(row, ch, cell)


def _apply_xm_default_volumes(xm):
    """FT2 itself plays a freshly triggered note at the INSTRUMENT'S OWN sample default volume
    (header byte) when the row carries no volume column / Cxx of its own, not at full volume -
    the same rule mod_module_load/s3m_module_load/it_module_load already apply; the XM loader was
    missing it, so every such note played at full volume regardless of how quiet its instrument
    was actually set up to be (some real-world samples default well below half volume specifically
    so they don't need an explicit volume on every note). Run as its own pass over the pattern
    data, once patterns AND instruments are both loaded - which one an XM file stores first
    differs by version, so this can't safely happen while patterns are still being parsed.
    """
    for p in range(xm.number_of_patterns()):
        pat = xm.patterns[p]
        for row in range(pat.rows):
            for ch in range(pat.channels):
                note = pat.get_note(row, ch)
                if not note.instrument or note.note == NOTE_STOP:
                    continue
                inst = xm.instruments.get(note.instrument)
                smp = inst.sample.get(0) if inst is not None else None
                if smp is None:
                    continue
                no_set_volume = (note.volume == 0 or note.volume > 0x50) and note.effect != EFFECT_VOLUME
                if note.note:
                    if no_set_volume and not note.keep_volume:
                        note.volume = 0x10 + clamp(smp.volume, 0, 64)
                        pat.set_note(row, ch, note)
                else:
                    # instrument number without a note (see PatternNote.env_restart): default volume
                    # again (unless the row sets one itself) and the envelope starts over
                    if no_set_volume:
                        note.volume = 0x10 + clamp(smp.volume, 0, 64)
                    note.env_restart = True
                    pat.set_note(row, ch, note)

