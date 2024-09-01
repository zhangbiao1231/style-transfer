"""utils/initialization."""
import contextlib
import platform
import threading

def emojis(str=""):
    """Returns an emoji-safe version of a string, stripped of emojis on Windows platforms."""
    return str.encode().decode("ascii", "ignore") if platform.system() == "Windows" else str
class TryExcept(contextlib.ContextDecorator):
    # YOLOv5 TryExcept class. Usage: @TryExcept() decorator or 'with TryExcept():' context manager
    def __init__(self, msg=""):
        """Initializes TryExcept with an optional message, used as a decorator or context manager for error handling."""
        self.msg = msg

    def __enter__(self):
        """Enter the runtime context related to this object for error handling with an optional message."""
        pass

    def __exit__(self, exc_type, value, traceback):
        """Context manager exit method that prints an error message with emojis if an exception occurred, always returns
        True.
        """
        if value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True