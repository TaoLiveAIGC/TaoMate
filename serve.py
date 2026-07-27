#!/usr/bin/env python3
"""Local preview server with HTTP Range support (needed for video seeking).

Usage: python3 serve.py [port]   (default port: 8137)
"""
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class RangeRequestHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler + single-range GET/HEAD support."""

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        # Code/config files must never be served from a stale browser cache;
        # large media files (.mp4/.jpg/...) stay cacheable to avoid re-downloads.
        if not re.search(r"\.(mp4|mov|webm|jpg|jpeg|png|webp|mp3|wav)(\?|$)", self.path, re.I):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path) or not os.path.exists(path):
            return super().send_head()

        range_header = self.headers.get("Range")
        if not range_header:
            return super().send_head()

        match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
        if not match:
            self.send_error(416, "Requested Range Not Satisfiable")
            return None

        size = os.path.getsize(path)
        start_s, end_s = match.groups()
        if not start_s and not end_s:
            self.send_error(416, "Requested Range Not Satisfiable")
            return None

        if start_s:
            start = int(start_s)
            end = int(end_s) if end_s else size - 1
        else:  # suffix range: last N bytes
            start = max(0, size - int(end_s))
            end = size - 1
        end = min(end, size - 1)

        if start > end or start >= size:
            self.send_error(416, "Requested Range Not Satisfiable")
            return None

        ctype = self.guess_type(path)
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        self.send_response(206)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(os.path.getmtime(path)))
        self.end_headers()

        if self.command == "GET":
            f.seek(start)
            self._range_file = f
            self._range_remaining = end - start + 1
            return f  # let do_GET invoke copyfile
        f.close()
        return None

    def copyfile(self, source, outputfile):
        range_file = getattr(self, "_range_file", None)
        if range_file is None:
            return super().copyfile(source, outputfile)

        remaining = self._range_remaining
        try:
            while remaining > 0:
                chunk = range_file.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)
        finally:
            range_file.close()
            self._range_file = None
        return None


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8137
    server = ThreadingHTTPServer(("", port), RangeRequestHandler)
    print(f"Serving HTTP on port {port} with Range support (http://localhost:{port}/) ...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
