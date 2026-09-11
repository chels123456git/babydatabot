"""
flask_app.py — WSGI entry point for running BabyDataBot in WEBHOOK mode.

bot.py normally runs with `application.run_polling(...)`, which needs a
long-lived background process — fine for Render/Railway, but PythonAnywhere's
free "Beginner" tier does not allow always-on background processes at all.
What it *does* give you for free is a normal web app. So on PythonAnywhere
we flip the bot from "pull updates from Telegram" (polling) to "Telegram
pushes each update to us" (a webhook) — same handlers, same predictions,
same SQLite database, just a different way of receiving messages.

How a message flows in this mode:

    Telegram  --HTTPS POST-->  this Flask app (PythonAnywhere's web server)
                                       |
                                       v
                     the exact same `Application` object bot.py builds
                          (handlers / predictions / database)

Nothing about bot.py's actual bot logic changes — this file only adapts
*how updates arrive*.

One-time setup (see the separate setup instructions):
  1. Set these in the PythonAnywhere WSGI configuration file (top of the
     file, before anything imports this module):
       TELEGRAM_BOT_TOKEN, MY_USER_ID, PARTNER_USER_ID, LOCAL_TZ, DB_PATH,
       WEBHOOK_SECRET, BOT_HTTP_PROXY (= "http://proxy.server:3128" — the
       PythonAnywhere free-tier outbound proxy).
  2. Point PythonAnywhere's WSGI file at this Flask `app` object.
  3. Tell Telegram where to send updates by visiting (once):
       https://api.telegram.org/bot<TOKEN>/setWebhook?url=https://<your-username>.pythonanywhere.com/webhook/<WEBHOOK_SECRET>
"""

from __future__ import annotations

import asyncio
import os

from flask import Flask, request
from telegram import Update

from bot import build_application, init_db, logger

WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET")
if not WEBHOOK_SECRET:
    raise RuntimeError("WEBHOOK_SECRET environment variable is not set.")

app = Flask(__name__)

# PythonAnywhere runs this module once per web worker process (not once per
# request), so it's safe to do one-time setup here: create the DB tables,
# build the same Application bot.py uses for polling, and give it its own
# asyncio event loop that stays alive for the lifetime of this process.
init_db()
telegram_app = build_application()

_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)
_loop.run_until_complete(telegram_app.initialize())
logger.info("BabyDataBot ready (webhook mode).")


@app.route(f"/webhook/{WEBHOOK_SECRET}", methods=["POST"])
def telegram_webhook():
    """Telegram POSTs one JSON update here per message/button tap."""
    update = Update.de_json(request.get_json(force=True), telegram_app.bot)
    _loop.run_until_complete(telegram_app.process_update(update))
    return "OK"


@app.route("/")
def index():
    # Just a friendly response if the bare domain is opened in a browser.
    return "BabyDataBot is running."
