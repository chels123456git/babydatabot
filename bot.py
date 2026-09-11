"""
BabyDataBot — a private Telegram bot for two parents to log feeds and sleep
for a newborn, and get simple predictions for what's coming next.

Design summary (read this before diving into the code):

1. SECURITY
   Every handler is wrapped with the `@restricted` decorator, which checks
   the sender's Telegram user ID against ALLOWED_USERS (built from the
   MY_USER_ID / PARTNER_USER_ID env vars). Anyone else gets a fixed
   "Unauthorized access" reply and nothing else happens. A catch-all
   handler at the bottom of the file makes sure even messages that don't
   match any button/command still get this treatment.

2. DATA MODEL (SQLite, one file, two tables)
   - feeds(id, timestamp_utc, amount_ml, milk_type, logged_by_id, logged_by_name)
   - sleep(id, start_utc, end_utc, logged_by_start_id, logged_by_start_name,
           logged_by_end_id, logged_by_end_name)
   `sleep.end_utc IS NULL` means the baby is currently asleep (an "open"
   session) — that single fact drives the sleep toggle button and the
   wake-up prediction.

3. TIME HANDLING
   All timestamps are stored in UTC (so the math never has to think about
   daylight saving or which parent's phone timezone it was on). They are
   only converted to LOCAL_TZ (an env var, e.g. "Asia/Singapore") when
   displayed to a human.

4. PREDICTIONS (see the `predict_*` functions)
   - Next feed time  = last feed time + average gap between the last 5 feeds
   - Next feed amount = average amount of the last 3 feeds
   - Next sleep event: if the baby is awake, predict when they'll next need
     to sleep (last wake time + average recent "awake window" length); if
     the baby is already asleep, predict when they'll wake up instead
     (sleep start + average recent sleep-session length). Either way you
     get one useful "next sleep-related time" instead of a prediction that
     doesn't apply to the current state.

5. WHY SYNCHRONOUS SQLITE INSIDE ASYNC HANDLERS?
   python-telegram-bot v20+ is asyncio-based, but sqlite3's standard library
   API is synchronous. For two users doing a handful of taps a day, a
   blocking sqlite3 call (sub-millisecond on a local file) inside an async
   handler is not a real performance problem, and it keeps the code much
   simpler to read for someone learning the codebase. If this ever needed
   to scale to many concurrent users, swapping in `aiosqlite` would be the
   next step — the DB functions are already isolated in one place to make
   that swap easy later.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from functools import wraps
from statistics import mean
from typing import Iterator, Optional
from zoneinfo import ZoneInfo

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

# --------------------------------------------------------------------------
# Configuration (all from environment variables — nothing hardcoded)
# --------------------------------------------------------------------------

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
DB_PATH = os.environ.get("DB_PATH", "babydata.db")
LOCAL_TZ_NAME = os.environ.get("LOCAL_TZ", "UTC")

_raw_my_id = os.environ.get("MY_USER_ID")
_raw_partner_id = os.environ.get("PARTNER_USER_ID")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable is not set.")
if not _raw_my_id or not _raw_partner_id:
    raise RuntimeError(
        "MY_USER_ID and PARTNER_USER_ID environment variables must both be set."
    )

try:
    MY_USER_ID = int(_raw_my_id)
    PARTNER_USER_ID = int(_raw_partner_id)
except ValueError as exc:
    raise RuntimeError(
        "MY_USER_ID and PARTNER_USER_ID must be numeric Telegram user IDs."
    ) from exc

# The set the @restricted decorator checks against.
ALLOWED_USER_IDS = {MY_USER_ID, PARTNER_USER_ID}

try:
    LOCAL_TZ = ZoneInfo(LOCAL_TZ_NAME)
except Exception as exc:  # invalid IANA name
    raise RuntimeError(
        f"LOCAL_TZ={LOCAL_TZ_NAME!r} is not a valid timezone name "
        "(e.g. 'Asia/Singapore', 'America/New_York', 'UTC')."
    ) from exc

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
# Quiet down the noisy HTTP client library a bit.
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("babydatabot")

# Conversation states for the "log a feed" flow.
FEED_AWAITING_AMOUNT, FEED_AWAITING_TYPE = range(2)

MILK_BREASTMILK = "breastmilk"
MILK_FORMULA = "formula"

MAIN_MENU = ReplyKeyboardMarkup(
    [
        [KeyboardButton("🍼 Log Feed"), KeyboardButton("💤 Sleep / Wake")],
        [KeyboardButton("📊 Predictions"), KeyboardButton("📅 Today's Summary")],
    ],
    resize_keyboard=True,
)


# --------------------------------------------------------------------------
# Time helpers
# --------------------------------------------------------------------------


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def to_local(dt: datetime) -> datetime:
    return dt.astimezone(LOCAL_TZ)


def fmt(dt: datetime) -> str:
    """Human-friendly local time, e.g. 'Wed 11 Sep, 3:45 PM'."""
    return to_local(dt).strftime("%a %d %b, %I:%M %p")


def fmt_duration(delta: timedelta) -> str:
    total_minutes = max(0, int(delta.total_seconds() // 60))
    hours, minutes = divmod(total_minutes, 60)
    if hours and minutes:
        return f"{hours}h {minutes}m"
    if hours:
        return f"{hours}h"
    return f"{minutes}m"


# --------------------------------------------------------------------------
# Database layer
# --------------------------------------------------------------------------


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Yields a sqlite3 connection with dict-like row access, always closed."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feeds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                amount_ml REAL NOT NULL,
                milk_type TEXT NOT NULL CHECK (milk_type IN ('breastmilk', 'formula')),
                logged_by_id INTEGER NOT NULL,
                logged_by_name TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sleep (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_utc TEXT NOT NULL,
                end_utc TEXT,
                logged_by_start_id INTEGER NOT NULL,
                logged_by_start_name TEXT NOT NULL,
                logged_by_end_id INTEGER,
                logged_by_end_name TEXT
            );
            """
        )
    logger.info("Database ready at %s", DB_PATH)


def insert_feed(amount_ml: float, milk_type: str, user_id: int, user_name: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO feeds (timestamp_utc, amount_ml, milk_type, logged_by_id, logged_by_name) "
            "VALUES (?, ?, ?, ?, ?)",
            (now_utc().isoformat(), amount_ml, milk_type, user_id, user_name),
        )


def get_recent_feeds(limit: int) -> list[sqlite3.Row]:
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT * FROM feeds ORDER BY timestamp_utc DESC LIMIT ?", (limit,)
        )
        return cur.fetchall()


def get_open_sleep_session() -> Optional[sqlite3.Row]:
    with get_connection() as conn:
        cur = conn.execute("SELECT * FROM sleep WHERE end_utc IS NULL ORDER BY start_utc DESC LIMIT 1")
        return cur.fetchone()


def start_sleep(user_id: int, user_name: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO sleep (start_utc, logged_by_start_id, logged_by_start_name) VALUES (?, ?, ?)",
            (now_utc().isoformat(), user_id, user_name),
        )


def end_sleep(session_id: int, user_id: int, user_name: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE sleep SET end_utc = ?, logged_by_end_id = ?, logged_by_end_name = ? WHERE id = ?",
            (now_utc().isoformat(), user_id, user_name, session_id),
        )


def get_completed_sleep_sessions(limit: int) -> list[sqlite3.Row]:
    """Most recent completed (start + end) sleep sessions, most recent first."""
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT * FROM sleep WHERE end_utc IS NOT NULL ORDER BY start_utc DESC LIMIT ?",
            (limit,),
        )
        return cur.fetchall()


def get_todays_feeds() -> list[sqlite3.Row]:
    start_of_day_local = to_local(now_utc()).replace(hour=0, minute=0, second=0, microsecond=0)
    start_of_day_utc = start_of_day_local.astimezone(timezone.utc).isoformat()
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT * FROM feeds WHERE timestamp_utc >= ? ORDER BY timestamp_utc ASC",
            (start_of_day_utc,),
        )
        return cur.fetchall()


def get_todays_sleep_sessions() -> list[sqlite3.Row]:
    start_of_day_local = to_local(now_utc()).replace(hour=0, minute=0, second=0, microsecond=0)
    start_of_day_utc = start_of_day_local.astimezone(timezone.utc).isoformat()
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT * FROM sleep WHERE start_utc >= ? ORDER BY start_utc ASC",
            (start_of_day_utc,),
        )
        return cur.fetchall()


# --------------------------------------------------------------------------
# Prediction engine
# --------------------------------------------------------------------------


def predict_next_feed_time() -> Optional[tuple[datetime, timedelta]]:
    """Returns (predicted_next_feed_time, average_gap) or None if not enough data."""
    rows = get_recent_feeds(5)
    if len(rows) < 2:
        return None
    timestamps = sorted(datetime.fromisoformat(r["timestamp_utc"]) for r in rows)
    gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
    avg_gap = sum(gaps, timedelta()) / len(gaps)
    return timestamps[-1] + avg_gap, avg_gap


def predict_next_feed_amount() -> Optional[float]:
    rows = get_recent_feeds(3)
    if not rows:
        return None
    return mean(r["amount_ml"] for r in rows)


def predict_next_sleep_event() -> Optional[tuple[str, datetime, timedelta]]:
    """
    Returns (kind, predicted_time, based_on_average) where kind is
    "wake" (baby is asleep now, predicting wake-up) or
    "sleep" (baby is awake now, predicting next sleep time).
    Returns None if there isn't enough history yet.
    """
    open_session = get_open_sleep_session()

    if open_session is not None:
        # Baby is currently asleep -> predict wake-up time from average
        # duration of recent completed sleep sessions.
        completed = get_completed_sleep_sessions(5)
        if not completed:
            return None
        durations = [
            datetime.fromisoformat(r["end_utc"]) - datetime.fromisoformat(r["start_utc"])
            for r in completed
        ]
        avg_duration = sum(durations, timedelta()) / len(durations)
        start = datetime.fromisoformat(open_session["start_utc"])
        return "wake", start + avg_duration, avg_duration

    # Baby is currently awake -> predict next sleep time from average
    # "awake window" length (time between one sleep ending and the next
    # one starting), using the most recent sessions.
    completed = get_completed_sleep_sessions(6)
    if len(completed) < 2:
        return None
    # completed is newest-first; sort oldest-first to pair consecutive sessions.
    completed_sorted = sorted(completed, key=lambda r: r["start_utc"])
    awake_windows = []
    for prev_session, next_session in zip(completed_sorted, completed_sorted[1:]):
        prev_end = datetime.fromisoformat(prev_session["end_utc"])
        next_start = datetime.fromisoformat(next_session["start_utc"])
        awake_windows.append(next_start - prev_end)
    if not awake_windows:
        return None
    avg_awake = sum(awake_windows, timedelta()) / len(awake_windows)
    last_wake_time = datetime.fromisoformat(completed_sorted[-1]["end_utc"])
    return "sleep", last_wake_time + avg_awake, avg_awake


# --------------------------------------------------------------------------
# Authorization
# --------------------------------------------------------------------------


def restricted(handler):
    """Decorator: blocks any user not in ALLOWED_USER_IDS with a fixed reply."""

    @wraps(handler)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user = update.effective_user
        if user is None or user.id not in ALLOWED_USER_IDS:
            logger.warning("Blocked unauthorized user: %s", user.id if user else "unknown")
            message = update.effective_message
            if message is not None:
                await message.reply_text("🚫 Unauthorized access.")
            elif update.callback_query is not None:
                await update.callback_query.answer("Unauthorized access.", show_alert=True)
            return ConversationHandler.END
        return await handler(update, context, *args, **kwargs)

    return wrapper


def display_name_for(user) -> str:
    """Label stored alongside each log entry. Falls back to the Telegram
    first name if it's neither of the two configured users (shouldn't
    happen since @restricted already filters, but keeps this safe)."""
    if user.id == MY_USER_ID:
        return user.first_name or "Me"
    if user.id == PARTNER_USER_ID:
        return user.first_name or "Partner"
    return user.first_name or "Unknown"


# --------------------------------------------------------------------------
# Command handlers
# --------------------------------------------------------------------------


@restricted
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "👶 *BabyDataBot* is online.\n\n"
        "Use the buttons below to log a feed, start/end sleep, or see predictions.\n"
        "Type /cancel any time to back out of logging a feed.",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=MAIN_MENU,
    )


@restricted
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "🍼 *Log Feed* — record an amount (ml) and whether it was breastmilk or formula.\n"
        "💤 *Sleep / Wake* — tap once to record sleep start, tap again to record wake-up.\n"
        "📊 *Predictions* — next feed time & amount, next sleep-related event.\n"
        "📅 *Today's Summary* — totals for today.\n"
        "/cancel — cancel a feed entry in progress.",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=MAIN_MENU,
    )


# ---- Feed logging conversation -------------------------------------------


@restricted
async def start_feed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.effective_message.reply_text(
        "How many ml did they just have? (numbers only, e.g. 90)"
    )
    return FEED_AWAITING_AMOUNT


@restricted
async def receive_feed_amount(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = (update.effective_message.text or "").strip()
    try:
        amount_ml = float(text)
        if amount_ml <= 0:
            raise ValueError
    except ValueError:
        await update.effective_message.reply_text(
            "That doesn't look like a valid amount. Please send a number of ml, e.g. 90."
        )
        return FEED_AWAITING_AMOUNT

    context.user_data["pending_feed_amount_ml"] = amount_ml
    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("🤱 Breastmilk", callback_data=f"feedtype:{MILK_BREASTMILK}"),
                InlineKeyboardButton("🍼 Formula", callback_data=f"feedtype:{MILK_FORMULA}"),
            ]
        ]
    )
    await update.effective_message.reply_text("Was that breastmilk or formula?", reply_markup=keyboard)
    return FEED_AWAITING_TYPE


@restricted
async def receive_feed_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    milk_type = query.data.split(":", 1)[1]

    amount_ml = context.user_data.pop("pending_feed_amount_ml", None)
    if amount_ml is None:
        # Shouldn't normally happen, but keep the conversation from getting stuck.
        await query.edit_message_text("Something went wrong — please start again with 🍼 Log Feed.")
        return ConversationHandler.END

    user = update.effective_user
    insert_feed(amount_ml, milk_type, user.id, display_name_for(user))

    label = "🤱 Breastmilk" if milk_type == MILK_BREASTMILK else "🍼 Formula"
    await query.edit_message_text(
        f"Logged: {amount_ml:g} ml of {label} at {fmt(now_utc())}."
    )
    return ConversationHandler.END


@restricted
async def cancel_feed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.pop("pending_feed_amount_ml", None)
    await update.effective_message.reply_text("Feed entry cancelled.", reply_markup=MAIN_MENU)
    return ConversationHandler.END


# ---- Sleep toggle ----------------------------------------------------------


@restricted
async def toggle_sleep(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    open_session = get_open_sleep_session()

    if open_session is None:
        start_sleep(user.id, display_name_for(user))
        await update.effective_message.reply_text(f"😴 Sleep started at {fmt(now_utc())}.")
        return

    start_time = datetime.fromisoformat(open_session["start_utc"])
    end_sleep(open_session["id"], user.id, display_name_for(user))
    duration = now_utc() - start_time
    await update.effective_message.reply_text(
        f"☀️ Wake-up logged at {fmt(now_utc())}. Slept for {fmt_duration(duration)}."
    )


# ---- Predictions & summary --------------------------------------------------


@restricted
async def show_predictions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lines = ["📊 *Predictions*\n"]

    feed_time_prediction = predict_next_feed_time()
    if feed_time_prediction:
        next_time, avg_gap = feed_time_prediction
        lines.append(
            f"🍼 Next feed: around *{fmt(next_time)}* "
            f"(avg gap of last 5 feeds: {fmt_duration(avg_gap)})"
        )
    else:
        lines.append("🍼 Next feed: not enough data yet (need at least 2 logged feeds).")

    amount_prediction = predict_next_feed_amount()
    if amount_prediction is not None:
        lines.append(f"🥛 Next feed amount: around *{amount_prediction:.0f} ml* (avg of last 3 feeds)")
    else:
        lines.append("🥛 Next feed amount: not enough data yet.")

    sleep_prediction = predict_next_sleep_event()
    if sleep_prediction:
        kind, predicted_time, avg = sleep_prediction
        if kind == "wake":
            lines.append(
                f"😴 Baby is asleep — likely wake-up: *{fmt(predicted_time)}* "
                f"(avg recent sleep length: {fmt_duration(avg)})"
            )
        else:
            lines.append(
                f"💤 Next likely sleep time: *{fmt(predicted_time)}* "
                f"(avg recent awake window: {fmt_duration(avg)})"
            )
    else:
        lines.append("💤 Next sleep event: not enough data yet (need at least 2 completed sleep sessions).")

    await update.effective_message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


@restricted
async def today_summary(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    feeds = get_todays_feeds()
    sleep_sessions = get_todays_sleep_sessions()

    total_ml = sum(f["amount_ml"] for f in feeds)
    breastmilk_count = sum(1 for f in feeds if f["milk_type"] == MILK_BREASTMILK)
    formula_count = sum(1 for f in feeds if f["milk_type"] == MILK_FORMULA)

    total_sleep = timedelta()
    for s in sleep_sessions:
        if s["end_utc"]:
            total_sleep += datetime.fromisoformat(s["end_utc"]) - datetime.fromisoformat(s["start_utc"])

    lines = [
        "📅 *Today's Summary*\n",
        f"🍼 Feeds: {len(feeds)} ({breastmilk_count} breastmilk, {formula_count} formula)",
        f"🥛 Total volume: {total_ml:.0f} ml",
        f"💤 Sleep sessions started today: {len(sleep_sessions)}",
        f"⏱️ Total sleep time today: {fmt_duration(total_sleep)}",
    ]
    await update.effective_message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


# ---- Catch-all (unauthorized users / unrecognized input) -------------------


async def catch_all(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    message = update.effective_message
    if message is None:
        return
    if user is None or user.id not in ALLOWED_USER_IDS:
        await message.reply_text("🚫 Unauthorized access.")
    else:
        await message.reply_text("Not sure what you mean — use the menu below.", reply_markup=MAIN_MENU)


# --------------------------------------------------------------------------
# Application setup
# --------------------------------------------------------------------------


def build_application() -> Application:
    builder = Application.builder().token(TELEGRAM_BOT_TOKEN)

    # Some free hosts (e.g. PythonAnywhere's "Beginner" tier) only allow
    # outbound internet access through their own proxy. If BOT_HTTP_PROXY is
    # set, route all Bot API calls through it; on every other host this
    # variable is simply left unset and nothing changes.
    proxy_url = os.environ.get("BOT_HTTP_PROXY")
    if proxy_url:
        builder = builder.proxy(proxy_url).get_updates_proxy(proxy_url)

    application = builder.build()

    feed_conversation = ConversationHandler(
        entry_points=[MessageHandler(filters.Regex("^🍼 Log Feed$"), start_feed)],
        states={
            FEED_AWAITING_AMOUNT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_feed_amount)
            ],
            FEED_AWAITING_TYPE: [CallbackQueryHandler(receive_feed_type, pattern=r"^feedtype:")],
        },
        fallbacks=[CommandHandler("cancel", cancel_feed)],
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(feed_conversation)
    application.add_handler(MessageHandler(filters.Regex("^💤 Sleep / Wake$"), toggle_sleep))
    application.add_handler(MessageHandler(filters.Regex("^📊 Predictions$"), show_predictions))
    application.add_handler(MessageHandler(filters.Regex("^📅 Today's Summary$"), today_summary))

    # Lower-priority group: anything that didn't match a handler above,
    # from anyone (authorized users get a gentle nudge, others get blocked).
    application.add_handler(MessageHandler(filters.ALL, catch_all), group=1)

    return application


def main() -> None:
    init_db()
    application = build_application()
    logger.info("BabyDataBot starting (polling)...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
