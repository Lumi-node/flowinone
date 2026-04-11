# FlowInOne API Reference

This document provides a reference for the actual API components available in the specified modules of the Pillow (PIL) library. These components are part of the underlying image processing stack used within FlowInOne, particularly for handling diverse image formats and data structures required for multimodal visual representation learning.

---

## `.venv/lib/python3.13/site-packages/PIL/AvifImagePlugin.py`

### `get_codec_version(codec_name: str) -> str | None`
Returns the version string of the specified AVIF codec if available, otherwise `None`. Used to check codec support.

**Example:**
```python
version = get_codec_version("libavif")
print(version)  # e.g., "0.9.1"
```

### `class AvifImageFile(ImageFile.ImageFile)`
Represents an AVIF image file that can be opened and processed using PIL.

#### `seek(self, frame: int) -> None`
Moves to the given frame number in a multi-frame AVIF file.

**Example:**
```python
image = AvifImageFile("animation.avif")
image.seek(1)  # Go to second frame
```

#### `load(self) -> Image.core.PixelAccess | None`
Loads pixel data from the current frame into memory.

**Example:**
```python
pixels = image.load()
print(pixels[0, 0])  # Access first pixel
```

---

## `.venv/lib/python3.13/site-packages/PIL/BdfFontFile.py`

### `bdf_char(data: bytes) -> tuple[int, int, int, int, bytes]`
Parses a single BDF character definition from raw byte data and returns its metrics and bitmap.

**Example:**
```python
char_data = bdf_char(b"STARTCHAR space...ENDCHAR")
x, y, w, h, bitmap = char_data
```

### `class BdfFontFile(FontFile.FontFile)`
Loads and represents a BDF (Bitmap Distribution Format) font for use in PIL.

**Example:**
```python
with open("font.bdf", "rb") as f:
    font = BdfFontFile(f)
```

---

## `.venv/lib/python3.13/site-packages/PIL/BlpImagePlugin.py`

### `class Format(IntEnum)`
Enumeration indicating the BLP file format version (BLP1 or BLP2).

### `class Encoding(IntEnum)`
Specifies the pixel encoding type used in BLP files (e.g., JPEG, DXT).

### `class AlphaEncoding(IntEnum)`
Indicates how alpha channel data is encoded in the BLP image.

### `unpack_565(i: int) -> tuple[int, int, int]`
Decodes a 16-bit RGB565 value into (R, G, B) components.

**Example:**
```python
r, g, b = unpack_565(0b1111100000011111)
```

### `decode_dxt1(data: bytes, width: int, height: int) -> bytearray`
Decodes DXT1-compressed texture data into raw pixel bytes.

**Example:**
```python
pixels = decode_dxt1(compressed_data, 64, 64)
```

### `decode_dxt3(data: bytes) -> tuple[bytearray, bytearray, bytearray, bytearray]`
Decodes DXT3-compressed data into four component bytearrays (RGBA).

**Example:**
```python
r, g, b, a = decode_dxt3(dxt3_data)
```

---

## `.venv/lib/python3.13/site-packages/PIL/BmpImagePlugin.py`

### `class BmpImageFile(ImageFile.ImageFile)`
Handles BMP (Bitmap) image files for reading within PIL.

**Example:**
```python
image = BmpImageFile("image.bmp")
image.load()
```

---

## `.venv/lib/python3.13/site-packages/PIL/BufrStubImagePlugin.py`

### `register_handler(handler: ImageFile.StubHandler | None) -> None`
Registers or unregisters a handler for BUFR stub images (used for format registration).

**Example:**
```python
register_handler(None)  # Unregister current handler
```

### `class BufrStubImageFile(ImageFile.StubImageFile)`
Placeholder class for BUFR (Binary Universal Form for the Representation of meteorological data) images.

**Example:**
```python
stub = BufrStubImageFile("data.bufr")
```

---

## `.venv/lib/python3.13/site-packages/PIL/ContainerIO.py`

### `class ContainerIO(IO[AnyStr])`
A file-like object that wraps byte or string containers for sequential I/O operations.

#### `isatty(self) -> bool`
Returns `True` if the stream is interactive (TTY), always `False` for ContainerIO.

#### `seekable(self) -> bool`
Returns `True` if the stream supports random access.

#### `seek(self, offset: int, mode: int = io.SEEK_SET) -> int`
Moves the file pointer to the specified position.

**Example:**
```python
container = ContainerIO(b"hello world")
container.seek(6)
```

#### `tell(self) -> int`
Returns the current position of the file pointer.

#### `readable(self) -> bool`
Returns `True` if the stream can be read.

#### `read(self, n: int = -1) -> AnyStr`
Reads up to `n` bytes; if `n` is -1, reads all remaining data.

#### `readline(self, n: int = -1) -> AnyStr`
Reads a single line, optionally limited to `n` bytes.

#### `readlines(self, n: int | None = -1) -> list[AnyStr]`
Reads all lines into a list.

#### `writable(self) -> bool`
Returns `True` if the stream supports writing.

#### `write(self, b: AnyStr) -> NoReturn`
Raises an error — writing is not supported.

#### `writelines(self, lines: Iterable[AnyStr]) -> NoReturn`
Raises an error — writelines is not supported.

#### `truncate(self, size: int | None = None) -> int`
Raises an error — truncation is not supported.

#### `fileno(self) -> int`
Raises `OSError` — no underlying file descriptor.

#### `flush(self) -> None`
No-op; included for file interface compatibility.

---

## `.venv/lib/python3.13/site-packages/PIL/CurImagePlugin.py`

### `class CurImageFile(BmpImagePlugin.BmpImageFile)`
Represents a Windows CUR (cursor) image file, which includes hotspot metadata.

**Example:**
```python
cursor = CurImageFile("cursor.cur")
cursor.seek(0)
```

---

## `.venv/lib/python3.13/site-packages/PIL/DcxImagePlugin.py`

### `class DcxImageFile(PcxImageFile)`
Handles DCX (multi-page PCX) image files.

#### `seek(self, frame: int) -> None`
Selects the given frame (page) in the DCX file.

**Example:**
```python
dcx = DcxImageFile("document.dcx")
dcx.seek(2)  # Load third page
```

#### `tell(self) -> int`
Returns the index of the current frame.

**Example:**
```python
current_page = dcx.tell()
```

---

## `.venv/lib/python3.13/site-packages/PIL/DdsImagePlugin.py`

### `class DDSD(IntFlag)`
Flags describing surface capabilities in DDS files (e.g., texture, cube map).

### `class DDSCAPS(IntFlag)`
Legacy DirectDraw surface capabilities.

### `class DDSCAPS2(IntFlag)`
Extended DirectDraw surface capabilities (e.g., volume, cube maps).

### `class DDPF(IntFlag)`
Describes pixel format flags in DDS headers.

### `class DXGI_FORMAT(IntEnum)`
Enumerates DXGI (DirectX Graphics Infrastructure) texture formats used in DDS.

---

## `.venv/lib/python3.13/site-packages/PIL/EpsImagePlugin.py`

### `has_ghostscript() -> bool`
Checks whether Ghostscript is available on the system for EPS rendering.

**Example:**
```python
if has_ghostscript():
    print("EPS support is enabled")
```

### `Ghostscript(
    command: str,
    args: list[str],
    env: dict[str, str] | None = None
) -> int`
Executes a Ghostscript command with the given arguments and environment.

**Example:**
```python
return_code = Ghostscript("gs", ["-sDEVICE=png16m", "-o", "out.png", "input.eps"])
```

---

## `.venv/lib/python3.13/site-packages/PIL/ExifTags.py`

### `class Base(IntEnum)`
Base class for EXIF tag enumerations; not instantiated directly.

---

## `.venv/lib/python3.13/site-packages/PIL/FitsImagePlugin.py`

### `class FitsImageFile(ImageFile.ImageFile)`
Handles FITS (Flexible Image Transport System) astronomical image files.

**Example:**
```python
fits_image = FitsImageFile("sky.fits")
```

### `class FitsGzipDecoder(ImageFile.PyDecoder)`

#### `decode(self, buffer: bytes | Image.SupportsArrayInterface) -> tuple[int, int]`
