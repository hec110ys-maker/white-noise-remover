# PyInstaller spec for the Windows build (lite: no AI / PyTorch, fully offline).
# Build from the project root:  pyinstaller packaging/WhiteNoiseRemover.spec
# Expects ffmpeg.exe / ffprobe.exe in packaging/ffmpeg/ (the CI workflow downloads them).
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).parent
FFMPEG_DIR = ROOT / "packaging" / "ffmpeg"
exe_suffix = ".exe" if sys.platform == "win32" else ""

datas = [(str(ROOT / "models" / "bd.rnnn"), "models"), (str(ROOT / "assets" / "icon.png"), "assets")]
binaries = []
for tool in ("ffmpeg", "ffprobe"):
    f = FFMPEG_DIR / f"{tool}{exe_suffix}"
    if f.exists():
        binaries.append((str(f), "bin"))

hiddenimports = ["run_cli", "PySide6.QtMultimedia"]
for pkg in ("noisereduce",):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=[
        "pytest", "tkinter", "IPython",
        # AI engine is left out of the lite build
        "torch", "torchaudio", "df", "libdf",
        "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
        "PySide6.Qt3DCore", "PySide6.QtQuick3D", "PySide6.QtPdf",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="WhiteNoiseRemover",
    console=False,  # GUI app: no black console window
    icon=str(ROOT / "assets" / "icon.ico"),
    upx=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    upx=False,
    name="WhiteNoiseRemover",
)
