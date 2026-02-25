"""
shared_mem_writer.py — seqlock writer for SharedHandData via ctypes + mmap.

Byte layout (must match shared_defs.h exactly):
  offset  0 : uint32  seq              (seqlock)
  offset  4 : uint8   hand_detected
  offset  5 : uint8[3] _pad
  offset  8 : float[21] lm_x           (MediaPipe native X coords)
  offset 92 : float[21] lm_y           (MediaPipe native Y coords)
  offset176 : double   timestamp
  total    : 184 bytes
"""

import ctypes
import mmap
import os
import struct
import time

# Offsets from shared_defs.h
OFFSET_SEQ           = 0
OFFSET_HAND_DETECTED = 4
OFFSET_LM_X          = 8
OFFSET_LM_Y          = 92
OFFSET_TIMESTAMP     = 176
SHM_SIZE             = 184
SHM_NAME             = "/hologram_hand"

# Load libc for shm_open
_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.shm_open.argtypes  = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]
_libc.shm_open.restype   = ctypes.c_int
_libc.shm_unlink.argtypes = [ctypes.c_char_p]
_libc.shm_unlink.restype  = ctypes.c_int

O_RDWR  = os.O_RDWR


class SharedMemWriter:
    def __init__(self):
        self._fd  = -1
        self._mm  = None
        self._seq = 0

    def open(self) -> bool:
        """Attach to existing shared memory (C++ creates it first)."""
        fd = _libc.shm_open(SHM_NAME.encode(), O_RDWR, 0o666)
        if fd < 0:
            return False

        self._fd  = fd
        self._mm  = mmap.mmap(fd, SHM_SIZE, mmap.MAP_SHARED,
                               mmap.PROT_READ | mmap.PROT_WRITE)
        # Sync current seq value from shared memory
        self._seq = struct.unpack_from('<I', self._mm, OFFSET_SEQ)[0]
        # Round up to next even number (ready state)
        if self._seq & 1:
            self._seq += 1
        return True

    def write(self, lm_x, lm_y, hand_detected: bool, timestamp: float):
        """
        Seqlock write:
          1. Increment seq to odd  → signal write in progress
          2. Write payload
          3. Increment seq to even → signal ready
        Each mmap.flush() = msync(MS_SYNC) = full memory barrier on aarch64.
        """
        mm = self._mm

        # --- Begin write: set seq to odd ---
        self._seq += 1
        struct.pack_into('<I', mm, OFFSET_SEQ, self._seq)
        mm.flush()   # memory barrier before payload

        # --- Write payload ---
        struct.pack_into('<B',   mm, OFFSET_HAND_DETECTED, 1 if hand_detected else 0)
        struct.pack_into('<21f', mm, OFFSET_LM_X, *lm_x)
        struct.pack_into('<21f', mm, OFFSET_LM_Y, *lm_y)
        struct.pack_into('<d',   mm, OFFSET_TIMESTAMP, timestamp)

        mm.flush()   # barrier before seq even

        # --- End write: set seq to even ---
        self._seq += 1
        struct.pack_into('<I', mm, OFFSET_SEQ, self._seq)
        mm.flush()   # final barrier

    def close(self):
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
