"""Plain TCP forwarder: listen on HOST:PORT, pass every connection to TARGET_HOST:TARGET_PORT.

HexaShell's SAM 3 server listens on 127.0.0.1 only. The service job runs it on a local port and runs
this next to it on the node's network address, so episodes on other Bouchet nodes can reach it
without changing HexaShell. Stdlib only.

    python -m policy.tcp_forward --listen 0.0.0.0:18300 --target 127.0.0.1:18299
"""
from __future__ import annotations

import argparse
import asyncio


async def pipe(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
    try:
        while data := await r.read(1 << 16):
            w.write(data)
            await w.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        try:
            w.close()
        except Exception:  # noqa: BLE001
            pass


async def main(listen: str, target: str) -> None:
    lh, lp = listen.rsplit(":", 1)
    th, tp = target.rsplit(":", 1)

    async def handle(cr, cw):
        try:
            sr, sw = await asyncio.open_connection(th, int(tp))
        except OSError:
            cw.close()
            return
        await asyncio.gather(pipe(cr, sw), pipe(sr, cw))

    server = await asyncio.start_server(handle, lh, int(lp))
    print(f"forwarding {listen} -> {target}", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", required=True)
    ap.add_argument("--target", required=True)
    a = ap.parse_args()
    asyncio.run(main(a.listen, a.target))
