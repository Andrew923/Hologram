import cv2
import threading
import time


class FastVideoCapture:
    """
    A fast video capture class that uses a separate thread to continuously
    read frames, minimizing latency by always providing the most recent frame.
    """

    RECONNECT_DELAY = 1.5   # seconds to wait before attempting reconnect
    RECONNECT_TRIES = 10    # max reconnect attempts before giving up

    def __init__(self, src=0, flip_horizontal=False, fps=30):
        """
        Initialize the video capture.

        Args:
            src: Camera source (device index or video file path)
            flip_horizontal: If True, flip frames horizontally (mirror effect)
            fps: Target frame rate cap (reduces firmware stress on cheap cameras)
        """
        self.src = src
        self.flip_horizontal = flip_horizontal
        self.fps = fps
        self.cap = self._open_cap()

        self.ret, self.frame = self.cap.read()
        if self.flip_horizontal and self.frame is not None:
            self.frame = cv2.flip(self.frame, 1)

        self.running = True
        self.lock = threading.Lock()
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    def _open_cap(self):
        cap = cv2.VideoCapture(self.src, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize OS buffer
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        return cap

    def _reader(self):
        """Background thread that continuously reads frames, reconnecting on failure."""
        reconnects = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if not self.running or reconnects >= self.RECONNECT_TRIES:
                    break
                # Camera dropped — wait for firmware to recover then reopen
                print(f"FastVideoCapture: read failed, reconnect attempt {reconnects + 1}/{self.RECONNECT_TRIES}", flush=True)
                self.cap.release()
                time.sleep(self.RECONNECT_DELAY)
                self.cap = self._open_cap()
                reconnects += 1
                continue
            reconnects = 0
            if self.flip_horizontal:
                frame = cv2.flip(frame, 1)
            with self.lock:
                self.ret = ret
                self.frame = frame  # Always overwrite with the newest frame

    def read(self):
        """
        Get the most recent frame.
        
        Returns:
            tuple: (success_flag, frame)
        """
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def release(self):
        """Release the video capture resources."""
        self.running = False
        self.t.join(timeout=1.0)
        self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    @property
    def isOpened(self):
        """Check if the capture is opened."""
        return self.cap.isOpened()
