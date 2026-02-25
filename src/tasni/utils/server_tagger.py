import http.server
import json
import os
import socketserver
import sqlite3

PORT = 8000
DB_FILE = "votes.db"
# We assume we are running FROM the output directory
OUTPUT_DIR = "."


def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute(
            """CREATE TABLE IF NOT EXISTS votes
                     (filename TEXT PRIMARY KEY,
                      vote TEXT,
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)"""
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Init Failed: {e}", flush=True)


class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True


class TaggerHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/vote":
            try:
                content_length = int(self.headers["Content-Length"])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode("utf-8"))

                filename = data.get("filename")
                vote = data.get("vote")

                print(f"Vote received: {filename} -> {vote}", flush=True)

                if filename and vote:
                    conn = sqlite3.connect(DB_FILE)
                    c = conn.cursor()
                    c.execute(
                        "INSERT OR REPLACE INTO votes (filename, vote) VALUES (?, ?)",
                        (filename, vote),
                    )
                    conn.commit()
                    conn.close()

                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "success"}).encode())
                else:
                    self.send_error(400, "Missing filename or vote")
            except Exception as e:
                print(f"Error handling POST: {e}", flush=True)
                self.send_error(500, str(e))
        else:
            self.send_error(404)


if __name__ == "__main__":
    init_db()
    # Ensure usage of 0.0.0.0
    try:
        with ThreadingHTTPServer(("0.0.0.0", PORT), TaggerHandler) as httpd:
            print(f"Serving Tagger on 0.0.0.0:{PORT} from {os.getcwd()}", flush=True)
            httpd.serve_forever()
    except Exception as e:
        print(f"Server Crashed: {e}", flush=True)
