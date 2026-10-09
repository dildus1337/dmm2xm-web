# dmm2xm web

Веб-версия конвертера Doom 2D DMM ↔ XM/MOD/S3M/IT.

Исходники — архив из темы [Новая версия dmm2xm](https://doom2d.org/forum/viewtopic.php?f=8&t=1081&start=40) (Y-Dr. Now, Google Drive `14UhMGpqh_dWVVBiKnWYqyUPKFoJzY2_5`, файлы от 2026-10-04). Алгоритм не переписывался: `dmm2xm_core.py` и `dmm2xm_convert.py` выполняются как есть на CPython, собранном в WebAssembly (Pyodide 0.27.7).

## Запуск

Нужен обычный HTTP, не `file://`.

```bash
cd dmm2xm-web
python3 -m http.server 8080
```

Открыть http://localhost:8080 . Первый запуск скачивает рантайм WASM с jsDelivr.

## Что умеет

- XM, MOD, S3M, IT → `.dmm` + `.dmi`
- выбор 8 каналов: первые / по числу нот / по громкости
- слайды и огибающие: выкл / события 0xFF / запекание
- смещения семпла Oxx/9xx и нарезка subsongs
- `.dmm` и песни из WAD → `.xm`
- квантование, лимит строк, моно, эмуляция бага SB 44 кГц
- результат — zip с журналом
