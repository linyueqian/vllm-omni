"""Serve the blog markdown as HTML with embedded images.

Usage: python serve_blog.py [--port 8888] [--file qwen3_omni_tts_blog.md]
"""

import argparse
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import markdown


def build_html(md_path: str) -> str:
    with open(md_path) as f:
        md_content = f.read()

    html_body = markdown.markdown(
        md_content,
        extensions=["tables", "fenced_code", "codehilite", "toc"],
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{Path(md_path).stem}</title>
<style>
    body {{
        max-width: 1100px;
        margin: 40px auto;
        padding: 0 20px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        line-height: 1.6;
        color: #24292e;
        background: #fff;
    }}
    h1 {{ border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }}
    h2 {{ border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; margin-top: 2em; }}
    h3 {{ margin-top: 1.5em; }}
    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 1em 0;
    }}
    th, td {{
        border: 1px solid #dfe2e5;
        padding: 6px 13px;
        text-align: left;
    }}
    th {{ background: #f6f8fa; font-weight: 600; }}
    tr:nth-child(even) {{ background: #f6f8fa; }}
    code {{
        background: #f6f8fa;
        padding: 0.2em 0.4em;
        border-radius: 3px;
        font-size: 85%;
    }}
    pre {{
        background: #f6f8fa;
        padding: 16px;
        border-radius: 6px;
        overflow-x: auto;
    }}
    pre code {{ background: none; padding: 0; }}
    img {{ max-width: 100%; height: auto; }}
    hr {{ border: none; border-top: 1px solid #eaecef; margin: 2em 0; }}
    blockquote {{
        border-left: 4px solid #dfe2e5;
        padding: 0 1em;
        color: #6a737d;
        margin: 1em 0;
    }}
    strong {{ font-weight: 600; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""


class BlogHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, html_content="", **kwargs):
        self.html_content = html_content
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == "/" or self.path.endswith(".html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(self.html_content.encode())
        else:
            super().do_GET()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--file", type=str, default="qwen3_omni_tts_blog.md")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    md_path = args.file
    if not Path(md_path).exists():
        md_path = str(script_dir / args.file)

    html = build_html(md_path)

    def handler_factory(*handler_args, **handler_kwargs):
        return BlogHandler(*handler_args, html_content=html, **handler_kwargs)

    server = HTTPServer(("0.0.0.0", args.port), handler_factory)
    print(f"Serving {md_path} at http://localhost:{args.port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
