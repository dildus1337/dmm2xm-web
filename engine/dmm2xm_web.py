"""Browser bridge for dmm2xm. The conversion code itself is untouched."""

import json
import os
import shutil
import traceback
import zipfile

from dmm2xm_convert import (
    ConversionError,
    ForwardOptions,
    convert_backward,
    convert_forward,
)
from dmm2xm_core import VERSION_TEXT, list_dmm_in_wad

WORK = "/work"
OUT = "/work/out"
LOG_PATH = "/work/log.txt"
ZIP_PATH = "/work/result.zip"


def version():
    return VERSION_TEXT


def _log_collect(logs):
    def log(msg):
        if msg is None:
            return
        text = str(msg)
        if not text.endswith("\n"):
            text += "\n"
        logs.append(text)
    return log


def _reset_out():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT, exist_ok=True)
    for path in (LOG_PATH, ZIP_PATH):
        if os.path.exists(path):
            os.remove(path)


def _safe_stem(name):
    base = os.path.basename(str(name).replace("\\", "/"))
    stem, _ext = os.path.splitext(base)
    cleaned = "".join(ch if (ch.isalnum() or ch in "._- ") else "_" for ch in stem).strip(" .")
    return cleaned or "song"


def _write_log(logs):
    os.makedirs(WORK, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as handle:
        handle.write("".join(logs))


def _zip_out(logs):
    with open(os.path.join(OUT, "log.txt"), "w", encoding="utf-8") as handle:
        handle.write("".join(logs))
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for root, _dirs, files in os.walk(OUT):
            for name in files:
                full = os.path.join(root, name)
                arc = os.path.relpath(full, OUT)
                archive.write(full, arc)


def list_wad(path):
    names = list_dmm_in_wad(path)
    if names is None:
        raise ConversionError("Это не корректный WAD-файл.")
    return list(names)


def _convert_backward(src, dest_dir, stem, options, log):
    os.makedirs(dest_dir, exist_ok=True)
    output = os.path.join(dest_dir, stem + ".dmm")
    return convert_backward(
        src,
        output,
        log=log,
        channel_selection=options.get("channel_selection") or "loudness",
        volume_slides=options.get("volume_slides") or "events",
        sample_offsets=bool(options.get("sample_offsets")),
        split_subsongs=bool(options.get("split_subsongs")),
    )


def _convert_forward_one(src, song_name, dest_dir, stem, options, log, wad=None):
    os.makedirs(dest_dir, exist_ok=True)
    opts = ForwardOptions()
    opts.input_file = src
    opts.dmm_in_wad_name = song_name or ""
    opts.output_file = os.path.join(dest_dir, stem + ".xm")
    opts.quantization = int(options.get("quantization") or 4)
    opts.max_rows = int(options.get("max_rows") or 256)
    opts.no_stereo = bool(options.get("no_stereo"))
    opts.emulate_bug = bool(options.get("emulate_bug"))
    opts.preloaded_wad = wad
    return convert_forward(opts, log=log)


def convert_json(raw):
    """Run a job. Input files are already on the virtual FS. Writes result.zip and log.txt."""
    logs = []
    log = _log_collect(logs)
    _reset_out()
    log(VERSION_TEXT + "\n")
    log("WebAssembly bridge · original Python engine\n\n")
    try:
        job = json.loads(raw)
        kind = job.get("kind")
        options = job.get("options") or {}
        if kind == "backward":
            items = job.get("items") or []
            if not items:
                raise ConversionError("Нет входных модулей.")
            multi = len(items) > 1
            for index, item in enumerate(items, 1):
                stem = _safe_stem(item.get("name") or "song")
                dest = os.path.join(OUT, f"{index:02d}_{stem}") if multi else OUT
                log(f"--- {item.get('name') or stem} -> DMM ---\n")
                _convert_backward(item["path"], dest, stem, options, log)
                log("\n")
        elif kind == "forward":
            items = job.get("items") or []
            if not items:
                raise ConversionError("Нет входных DMM/WAD.")
            wad_cache = {}
            multi = len(items) > 1
            for index, item in enumerate(items, 1):
                stem = _safe_stem(item.get("song") or item.get("name") or "song")
                dest = os.path.join(OUT, f"{index:02d}_{stem}") if multi else OUT
                label = item.get("song") or item.get("name") or stem
                log(f"--- {label} -> XM ---\n")
                wad = None
                if item.get("wad"):
                    wad_path = item["path"]
                    if wad_path not in wad_cache:
                        from dmm2xm_core import WadFile
                        wad_cache[wad_path] = WadFile(wad_path)
                    wad = wad_cache[wad_path]
                _convert_forward_one(
                    item["path"], item.get("song") or "", dest, stem, options, log, wad=wad
                )
                log("\n")
        else:
            raise ConversionError(f"Неизвестный режим: {kind}")
        log("Готово.\n")
    except ConversionError as exc:
        log(f"Error: {exc}\n")
    except Exception:
        log("Unexpected error:\n" + traceback.format_exc())
    _write_log(logs)
    _zip_out(logs)
    return "".join(logs)
