#!/usr/bin/env bash
# Sonda di sola lettura: quali motori JavaScript o browser esistono su lagrange0 per eseguire la logica e l'autotest
# dei pacchetti di revisione offline (revisioni/*/logica.js, test.js). Elenca programmi di sistema, JDK, librerie
# condivise (JavaScriptCore, SpiderMonkey, V8, QuickJS, Duktape) e moduli Python (Qt QML, GObject) dell'interprete
# di sistema e del venv adottato. Non installa, non scarica e non scrive nulla fuori dal log.
# Uso: sbatch slurm/js_engine_probe.sbatch
#SBATCH --job-name=lvl-js-probe
#SBATCH --partition=mat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/home/prignano/vision_limite/outputs/logs/%x-%j.out
#SBATCH --error=/home/prignano/vision_limite/outputs/logs/%x-%j.err

set -uo pipefail
: "${SLURM_JOB_ID:?Submit through SLURM}"
source /home/prignano/vision_limite/config/paths.env
export SEED=0 CONFIG_PATH=-
cd "$PROJECT_ROOT"
source slurm/prologue.sh
lagrange_prologue cpu
echo "== programmi"
for p in node nodejs deno bun qjs quickjs d8 js js78 js91 js102 js115 jsc rhino jjs gjs cjs seed chromium \
         chromium-browser google-chrome google-chrome-stable firefox firefox-esr epiphany midori surf phantomjs \
         wkhtmltopdf java; do
  w=$(command -v "$p" 2>/dev/null) && echo "$p -> $w"
done
echo "== /usr/lib/jvm"
ls /usr/lib/jvm 2>&1
ls -d /usr/lib/jvm/*/bin/jjs /usr/lib/jvm/*/jre/bin/jjs 2>/dev/null
command -v java >/dev/null && java -version 2>&1
echo "== /opt, /snap/bin"
ls /opt 2>&1
ls /snap/bin 2>&1 | head -20
echo "== librerie condivise"
ldconfig -p 2>/dev/null | grep -i -E 'javascriptcore|mozjs|libv8|libnode|duktape|quickjs|webkit|qt5qml|qt6qml|libQt.*Qml' || echo "nessuna"
probe_python() {
  "$1" - <<'EOF'
import importlib, sys
print(sys.executable, sys.version.split()[0])
for mod in ("PyQt5.QtQml", "PyQt6.QtQml", "PySide2.QtQml", "PySide6.QtQml", "PyQt5.QtWebEngineWidgets",
            "PySide6.QtWebEngineCore", "gi", "quickjs", "py_mini_racer", "dukpy", "js2py", "execjs", "playwright",
            "selenium", "pyppeteer", "STPyV8", "pythonmonkey", "esprima", "pyjsparser", "wasmtime"):
    try:
        importlib.import_module(mod)
        print("  importabile:", mod)
    except Exception as exc:
        print("  no:", mod, type(exc).__name__)
try:
    import gi
    for ns, vers in (("JavaScriptCore", ("4.0", "4.1", "6.0")), ("WebKit2", ("4.0", "4.1")), ("WebKit", ("6.0",))):
        for v in vers:
            try:
                gi.require_version(ns, v)
                importlib.import_module("gi.repository." + ns)
                print("  gi:", ns, v)
            except Exception as exc:
                pass
except Exception:
    pass
EOF
}
echo "== python di sistema"
for py in /usr/bin/python3; do [[ -x $py ]] && probe_python "$py"; done
echo "== python del venv"
[[ -x $RUN_PYTHON ]] && probe_python "$RUN_PYTHON"
ledger_note ok "js_engine_probe: elenco dei motori JavaScript e browser su lagrange0 nel log"
