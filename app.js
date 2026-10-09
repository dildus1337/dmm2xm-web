const state = {
  worker: null,
  ready: false,
  back: [],
  fwd: [],
  seq: 1
};

const $ = (id) => document.getElementById(id);

function setStatus(text, kind) {
  const node = $("status");
  node.textContent = text;
  node.className = "status" + (kind ? " " + kind : "");
}

function log(text) {
  $("log").textContent = text;
}

function extOf(name) {
  const m = /\.([a-z0-9]+)$/i.exec(name || "");
  return m ? m[1].toLowerCase() : "";
}

function kb(n) {
  if (n < 1024) return n + " B";
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
  return (n / 1024 / 1024).toFixed(2) + " MB";
}

function renderFiles(list, node, kind) {
  node.innerHTML = "";
  list.forEach((item, index) => {
    const row = document.createElement("div");
    row.className = "item";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = item.enabled !== false;
    box.addEventListener("change", () => { item.enabled = box.checked; });
    const name = document.createElement("div");
    name.textContent = item.song ? item.file + " / " + item.song : item.name;
    const meta = document.createElement("small");
    meta.textContent = kb(item.size || 0);
    row.append(box, name, meta);
    if (kind === "fwd" && item.songs) {
      const songs = document.createElement("div");
      songs.className = "songs";
      songs.style.gridColumn = "1 / -1";
      item.songs.forEach((song) => {
        const s = document.createElement("label");
        s.className = "item";
        const c = document.createElement("input");
        c.type = "checkbox";
        c.checked = song.enabled !== false;
        c.addEventListener("change", () => { song.enabled = c.checked; });
        const t = document.createElement("div");
        t.textContent = song.name;
        s.append(c, t);
        songs.append(s);
      });
      row.append(songs);
    }
    node.append(row);
    void index;
  });
}

function wireDrop(node, onFiles) {
  node.addEventListener("dragover", (e) => { e.preventDefault(); node.classList.add("hot"); });
  node.addEventListener("dragleave", () => node.classList.remove("hot"));
  node.addEventListener("drop", (e) => {
    e.preventDefault();
    node.classList.remove("hot");
    onFiles(e.dataTransfer.files);
  });
}

async function fileRecord(file) {
  return {
    name: file.name,
    size: file.size,
    bytes: new Uint8Array(await file.arrayBuffer()),
    enabled: true
  };
}

async function addBack(files) {
  const next = [];
  for (const file of files) {
    const ext = extOf(file.name);
    if (!["xm", "mod", "s3m", "it"].includes(ext)) continue;
    next.push(await fileRecord(file));
  }
  state.back.push(...next);
  renderFiles(state.back, $("list-back"));
}

async function addFwd(files) {
  for (const file of files) {
    const ext = extOf(file.name);
    if (!["dmm", "wad", "dmi"].includes(ext)) continue;
    const rec = await fileRecord(file);
    if (ext === "wad") {
      rec.songs = [];
      state.fwd.push(rec);
      renderFiles(state.fwd, $("list-fwd"), "fwd");
      askWad(rec);
    } else {
      state.fwd.push(rec);
    }
  }
  renderFiles(state.fwd, $("list-fwd"), "fwd");
}

function askWad(rec) {
  const id = state.seq++;
  rec.req = id;
  setStatus("Читаю WAD…");
  const copy = rec.bytes.slice();
  state.worker.postMessage({ type: "list-wad", id, bytes: copy }, [copy.buffer]);
}

function selectedBack() {
  return state.back.filter((item) => item.enabled !== false);
}

function selectedFwd() {
  const items = [];
  state.fwd.forEach((item) => {
    if (extOf(item.name) === "dmi") {
      items.push({ name: item.name, role: "dmi", bytes: item.bytes, enabled: true });
      return;
    }
    if (item.songs) {
      item.songs.filter((s) => s.enabled !== false).forEach((song) => {
        items.push({ name: item.name, song: song.name, path: item.path, wad: true, enabled: true });
      });
    } else if (item.enabled !== false && extOf(item.name) === "dmm") {
      items.push(item);
    }
  });
  return items;
}

function optionsBack() {
  return {
    channel_selection: document.querySelector("input[name=ch]:checked").value,
    volume_slides: $("slides").value,
    sample_offsets: $("offsets").checked,
    split_subsongs: $("subsongs").checked
  };
}

function optionsFwd() {
  return {
    quantization: Number($("quant").value),
    max_rows: Number($("rows").value),
    no_stereo: $("mono").checked,
    emulate_bug: $("bug").checked
  };
}

function run(kind) {
  if (!state.ready) return;
  const items = kind === "backward" ? selectedBack() : selectedFwd();
  if (!items.length) {
    setStatus("Нечего конвертировать.", "bad");
    return;
  }
  const id = state.seq++;
  setStatus("Конвертация в WASM…");
  $("run-back").disabled = true;
  $("run-fwd").disabled = true;
  const payloadItems = items.map((item) => ({
    name: item.name,
    song: item.song || "",
    path: item.path || "",
    wad: Boolean(item.wad),
    role: item.role || "",
    bytes: item.path ? null : (item.bytes ? item.bytes.slice() : null)
  }));
  const transfer = [];
  payloadItems.forEach((item) => {
    if (item.bytes) transfer.push(item.bytes.buffer);
  });
  state.worker.postMessage({
    type: "convert",
    id,
    kind,
    options: kind === "backward" ? optionsBack() : optionsFwd(),
    items: payloadItems
  }, transfer);
  if (kind === "backward") {
    state.back.forEach((item) => { if (!item.path) item.bytes = null; });
  }
}

function addDownload(name, bytes) {
  const blob = new Blob([bytes], { type: "application/zip" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.innerHTML = "<span>" + name + "</span><span>" + kb(bytes.byteLength) + "</span>";
  $("downloads").prepend(a);
}

function boot() {
  state.worker = new Worker("worker.js");
  state.worker.onmessage = (event) => {
    const msg = event.data || {};
    if (msg.type === "status") setStatus(msg.text);
    if (msg.type === "ready") {
      state.ready = true;
      $("engine").textContent = msg.version;
      setStatus("Движок готов.", "ok");
      log(msg.version + "\nЖдёт файл.");
    }
    if (msg.type === "wad-list") {
      const rec = state.fwd.find((item) => item.req === msg.id);
      if (rec) {
        rec.path = msg.path;
        rec.songs = msg.names.map((name) => ({ name, enabled: true }));
        renderFiles(state.fwd, $("list-fwd"), "fwd");
      }
      setStatus("В WAD: " + msg.names.length + " DMM.", "ok");
    }
    if (msg.type === "result") {
      $("run-back").disabled = false;
      $("run-fwd").disabled = false;
      log(msg.log);
      const failed = /^(Error:|Unexpected error:)/m.test(msg.log);
      setStatus(failed ? "Конвертация завершилась с ошибкой." : "Готово. Архив ниже.", failed ? "bad" : "ok");
      const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
      addDownload("dmm2xm-" + stamp + ".zip", msg.zip);
    }
    if (msg.type === "error") {
      $("run-back").disabled = false;
      $("run-fwd").disabled = false;
      setStatus(msg.message, "bad");
      log(msg.message);
    }
  };
  state.worker.postMessage({ type: "init" });
}

document.querySelectorAll(".tabs button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((b) => b.classList.remove("active"));
    button.classList.add("active");
    $("panel-back").classList.toggle("hidden", button.dataset.tab !== "back");
    $("panel-fwd").classList.toggle("hidden", button.dataset.tab !== "fwd");
    $("panel-about").classList.toggle("hidden", button.dataset.tab !== "about");
  });
});

wireDrop($("drop-back"), addBack);
wireDrop($("drop-fwd"), addFwd);
$("pick-back").addEventListener("change", (e) => addBack(e.target.files));
$("pick-fwd").addEventListener("change", (e) => addFwd(e.target.files));
$("clear-back").addEventListener("click", () => { state.back = []; renderFiles(state.back, $("list-back")); });
$("clear-fwd").addEventListener("click", () => { state.fwd = []; renderFiles(state.fwd, $("list-fwd"), "fwd"); });
$("run-back").addEventListener("click", () => run("backward"));
$("run-fwd").addEventListener("click", () => run("forward"));

boot();

