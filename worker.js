/* dmm2xm web worker: original Python on CPython/WebAssembly (Pyodide). */
const PYODIDE_INDEX = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";

let pyodide = null;

function baseUrl() {
  return self.location.href.replace(/[^/]+$/, "");
}

async function readText(path) {
  const response = await fetch(baseUrl() + path);
  if (!response.ok) {
    throw new Error("Не удалось загрузить " + path + " (" + response.status + ")");
  }
  return response.text();
}

async function boot() {
  importScripts(PYODIDE_INDEX + "pyodide.js");
  self.postMessage({ type: "status", text: "Загрузка CPython WASM…" });
  pyodide = await loadPyodide({ indexURL: PYODIDE_INDEX });
  self.postMessage({ type: "status", text: "Загрузка движка dmm2xm…" });
  const core = await readText("engine/dmm2xm_core.py");
  const convert = await readText("engine/dmm2xm_convert.py");
  const api = await readText("engine/dmm2xm_web.py");
  pyodide.FS.mkdirTree("/engine");
  pyodide.FS.mkdirTree("/work");
  pyodide.FS.writeFile("/engine/dmm2xm_core.py", core);
  pyodide.FS.writeFile("/engine/dmm2xm_convert.py", convert);
  pyodide.FS.writeFile("/engine/dmm2xm_web.py", api);
  pyodide.runPython("import sys; sys.path.insert(0, '/engine'); import dmm2xm_web");
  const version = pyodide.runPython("dmm2xm_web.version()");
  self.postMessage({ type: "ready", version: String(version) });
}

function writeInput(name, bytes) {
  const safe = "in_" + Math.random().toString(16).slice(2) + "_" + name.replace(/[^\w.\-]+/g, "_");
  const path = "/work/" + safe;
  pyodide.FS.writeFile(path, bytes);
  return path;
}

self.onmessage = async (event) => {
  const msg = event.data || {};
  try {
    if (msg.type === "init") {
      await boot();
      return;
    }
    if (!pyodide) throw new Error("Движок ещё не загружен.");

    if (msg.type === "list-wad") {
      const path = writeInput("input.wad", msg.bytes);
      pyodide.globals.set("WAD_PATH", path);
      const names = JSON.parse(pyodide.runPython("import json; json.dumps(dmm2xm_web.list_wad(WAD_PATH))"));
      self.postMessage({ type: "wad-list", id: msg.id, names, path });
      return;
    }

    if (msg.type === "convert") {
      pyodide.FS.mkdirTree("/work/bundle");
      const items = (msg.items || []).map((item) => {
        if (item.role === "dmi") {
          const base = String(item.name || "inst.dmi").split(/[\\/]/).pop();
          const path = "/work/bundle/" + base;
          pyodide.FS.writeFile(path, item.bytes);
          return { name: base, path, role: "dmi" };
        }
        const path = item.path || writeInput(item.name || "input.bin", item.bytes);
        if (!item.wad && item.bytes) {
          const base = String(item.name || "song.dmm").split(/[\\/]/).pop();
          const bundled = "/work/bundle/" + base;
          pyodide.FS.writeFile(bundled, item.bytes);
          return { name: item.name, song: item.song || "", path: bundled, wad: false };
        }
        return {
          name: item.name,
          song: item.song || "",
          path,
          wad: Boolean(item.wad)
        };
      }).filter((item) => item.role !== "dmi");
      const job = { kind: msg.kind, options: msg.options || {}, items };
      pyodide.globals.set("JOB", JSON.stringify(job));
      const log = pyodide.runPython("dmm2xm_web.convert_json(JOB)");
      const zip = pyodide.FS.readFile("/work/result.zip");
      const copy = new Uint8Array(zip);
      self.postMessage({ type: "result", id: msg.id, log: String(log), zip: copy }, [copy.buffer]);
      return;
    }
  } catch (err) {
    self.postMessage({
      type: "error",
      id: msg.id,
      message: err && err.message ? err.message : String(err)
    });
  }
};
