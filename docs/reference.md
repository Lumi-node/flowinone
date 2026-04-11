# FlowInOne API Reference

This document provides a reference for the actual API functions and classes available in the specified modules of the Pillow (PIL) library, as used in the context of FlowInOne. Only documented entities are included—no additional functions or classes have been invented.

---

## `.venv/lib/python3.13/site-packages/PIL/AvifImagePlugin.py`

### `get_codec_version(codec_name: str) -> str | None`
Returns the version string of the specified AVIF codec if available, or `None` otherwise. Used to check codec support.

**Example:**
```python
version = get_codec_version("libavif")
print(version)  # e.g., "0.9.1"
```

### `class AvifImageFile(ImageFile.ImageFile)`
Represents an AVIF image file. Inherits from `ImageFile.ImageFile` and supports standard PIL image operations.

#### `seek(self, frame: int) -> None`
Moves to the given frame (page) in a multi-frame AVIF file.

**Example:**
```python
image = AvifImageFile("animation.avif")
image.seek(1)  # Go to second frame
```

#### `load(self) -> Image.core.PixelAccess | None`
Decodes and loads pixel data from the current frame.

**Example:**
```python
pixels = image.load()
print(pixels[0, 0])  # Access first pixel
```

---

## `.venv/lib/python3.13/site-packages/PIL/BdfFontFile.py`

### `bdf_char(data: bytes) -> tuple[int, int, int, int, bytes]`
Parses a single BDF character definition from raw byte data and returns its dimensions, offset, and bitmap.

**Example:**
```python
char_data = b"STARTCHAR charname..."
width, height, dx, dy, bitmap = bdf_char(char_data)
```

### `class BdfFontFile(FontFile.FontFile)`
Parses and represents a BDF (Bitmap Distribution Format) font file.

**Example:**
```python
with open("font.bdf", "rb") as f:
    font = BdfFontFile(f)
```

---

## `.venv/lib/python3.13/site-packages/PIL/BlpImagePlugin.py`

### `class Format(IntEnum)`
Enumeration of BLP container formats (e.g., BLP1, BLP2).

### `class Encoding(IntEnum)`
Enumeration of texture encodings used in BLP files (e.g., DXT1, DXT3).

### `class AlphaEncoding(IntEnum)`
Enumeration of alpha channel encoding methods in BLP.

### `unpack_565(i: int) -> tuple[int, int, int]`
Converts a 16-bit RGB565 value into an `(r, g, b)` tuple.

**Example:**
```python
r, g, b = unpack_565(0b1111100000011111)
```

### `decode_dxt1(data: bytes, w: int, h: int) -> bytearray`
Decodes DXT1-compressed texture data into raw RGB bytes.

**Example:**
```python
rgb_data = decode_dxt1(compressed_bytes, 64, 64)
```

### `decode_dxt3(data: bytes) -> tuple[bytearray, bytearray, bytearray, bytearray]`
Decodes DXT3-compressed data into four component bytearrays (RGBA).

**Example:**
```python
r, g, b, a = decode_dxt3(dxt3_bytes)
```

---

## `.venv/lib/python3.13/site-packages/PIL/BmpImagePlugin.py`

### `class BmpImageFile(ImageFile.ImageFile)`
Implements support for BMP image files. Handles loading and parsing of BMP headers and pixel data.

**Example:**
```python
image = BmpImageFile("image.bmp")
image.load()
```

---

## `.venv/lib/python3.13/site-packages/PIL/BufrStubImagePlugin.py`

### `register_handler(handler: ImageFile.StubHandler | None) -> None`
Registers or unregisters a handler for BUFR file stubs (used for lazy loading).

**Example:**
```python
register_handler(None)  # Unregister current handler
```

### `class BufrStubImageFile(ImageFile.StubImageFile)`
Stub class for BUFR meteorological data files; defers actual decoding until load time.

**Example:**
```python
stub = BufrStubImageFile("data.bufr")
```

---

## `.venv/lib/python3.13/site-packages/PIL/ContainerIO.py`

### `class ContainerIO(IO[AnyStr])`
A file-like wrapper around in-memory data (e.g., bytes or strings), allowing it to be used as a stream.

#### `isatty(self) -> bool`
Returns `False`. Indicates this is not a TTY device.

#### `seekable(self) -> bool`
Returns `True` if the underlying buffer supports seeking.

#### `seek(self, offset: int, mode: int = io.SEEK_SET) -> int`
Moves the read/write pointer within the buffer.

**Example:**
```python
io_obj = ContainerIO(b"hello world")
io_obj.seek(6)
```

#### `tell(self) -> int`
Returns current position in the stream.

#### `readable(self) -> bool`
Returns `True` if the stream can be read.

#### `read(self, n: int = -1) -> AnyStr`
Reads up to `n` bytes from the stream.

#### `readline(self, n: int = -1) -> AnyStr`
Reads a single line from the stream.

#### `readlines(self, n: int | None = -1) -> list[AnyStr]`
Reads all lines into a list.

#### `writable(self) -> bool`
Returns `True` if the stream supports writing.

#### `write(self, b: AnyStr) -> NoReturn`
Raises an error—writing is not supported.

#### `writelines(self, lines: Iterable[AnyStr]) -> NoReturn`
Raises an error—writing is not supported.

#### `truncate(self, size: int | None = None) -> int`
Truncates the underlying buffer to the given size.

#### `fileno(self) -> int`
Raises `OSError`—no file descriptor is available.

#### `flush(self) -> None`
No-op. Included for file interface compatibility.

---

## `.venv/lib/python3.13/site-packages/PIL/CurImagePlugin.py`

### `class CurImageFile(BmpImagePlugin.BmpImageFile)`
Implements support for Windows `.cur` cursor files, including hotspot information.

**Example:**
```python
cursor = CurImageFile("icon.cur")
cursor.load()
```

---

## `.venv/lib/python3.13/site-packages/PIL/DcxImagePlugin.py`

### `class DcxImageFile(PcxImageFile)`
Handles DCX (multi-page PCX) image files.

#### `seek(self, frame: int) -> None`
Selects the specified sub-image (frame) in the DCX file.

**Example:**
```python
dcx = DcxImageFile("pages.dcx")
dcx.seek(2)  # Load third page
```

#### `tell(self) -> int`
Returns the index of the current frame.

**Example:**
```python
index = dcx.tell()
```

---

## `.venv/lib/python3.13/site-packages/PIL/DdsImagePlugin.py`

### `class DDSD(IntFlag)`
Flags indicating which fields are present in a DDS header (e.g., width, height, pitch).

### `class DDSCAPS(IntFlag)`
Legacy DirectDraw surface capabilities flags.

### `class DDSCAPS2(IntFlag)`
Extended surface capabilities (e.g., texture, cube map).

### `class DDPF(IntFlag)`
Pixel format flags (e.g., RGB, YUV, alpha).

### `class DXGI_FORMAT(IntEnum)`
Enumeration of DXGI texture formats used in DDS files.

---

## `.venv/lib/python3.13/site-packages/PIL/EpsImagePlugin.py`

### `has_ghostscript() -> bool`
Checks whether Ghostscript is available on the system for EPS rendering.

**Example:**
```python
if has_ghostscript():
    print("Ghostscript is available")
```

### `Ghostscript(
    command: list[str],
    stdin: bytes | None = None,
) -> tuple[int, bytes, bytes]`
Executes a Ghostscript command and returns the exit code, stdout, and stderr.

**Example:**
```python
return_code, out, err = Ghostscript(["gs", "-version"])
```

---

## `.venv/lib/python3.13/site-packages/PIL/ExifTags.py`

### `class Base(IntEnum)`
Base class for EXIF tag enumerations. Not used directly.

---

## `.venv/lib/python3.13/site-packages/PIL/FitsImagePlugin.py`

### `class FitsImageFile(ImageFile.ImageFile)`
Implements support for FITS (Flexible Image Transport System) astronomical image files.

**Example:**
```python
fits_image = FitsImageFile("image.fits")
```

### `class FitsGzipDecoder(ImageFile.PyDecoder)`

#### `decode(self, buffer: bytes | Image.SupportsArrayInterface) -> tuple[int, int]`
Decodes a block of gzipped