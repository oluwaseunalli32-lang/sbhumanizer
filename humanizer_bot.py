import os
import asyncio
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from google import genai
from google.genai import types
from google.genai.errors import APIError

# ---------- Configuration ----------
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

# ---------- Logging ----------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Quiet down noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)
logging.getLogger("telegram.ext.Application").setLevel(logging.INFO)

# ---------- Constants ----------
SYSTEM_INSTRUCTION = (
    "You are a helpful assistant that rewrites text to sound more natural, "
    "authentic, and human. Use contractions, vary sentence structure, avoid "
    "AI-sounding phrases, and make it conversational. Keep the original meaning intact."
)

# Try these models in order if one is overloaded
MODEL_FALLBACKS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

# Retryable HTTP statuses from Gemini
RETRYABLE_CODES = {429, 500, 502, 503, 504}

# ---------- Gemini Call ----------
def _generate_sync(text: str, model: str) -> str:
    """Blocking call to Gemini (runs in a thread)."""
    response = client.models.generate_content(
        model=model,
        contents=f"Rewrite the following text to sound more natural, authentic, and human:\n\n{text}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.8,
            max_output_tokens=1500,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    return response.text.strip()


async def humanize_text(text: str) -> str:
    """Call Gemini with retries across models."""
    last_error = None

    for model in MODEL_FALLBACKS:
        for attempt in range(3):  # 3 attempts per model
            try:
                return await asyncio.to_thread(_generate_sync, text, model)
            except APIError as e:
                code = getattr(e, "code", None) or getattr(e, "status_code", None)
                if code in RETRYABLE_CODES:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    logging.warning(
                        f"{model} returned {code}, retrying in {wait}s "
                        f"(attempt {attempt + 1}/3)"
                    )
                    await asyncio.sleep(wait)
                    last_error = e
                    continue
                logging.error(f"Gemini non-retryable error on {model}: {e}")
                last_error = e
                break  # try next model
            except Exception as e:
                logging.error(f"Unexpected error on {model}: {e}")
                last_error = e
                break

    logging.error(f"All models failed. Last error: {last_error}")
    return "⚠️ Gemini is a bit overloaded right now. Please try again in a few seconds."


# ---------- Command Handlers ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hi! Send me any AI-generated text and I'll rewrite it to sound more "
        "natural, authentic, and human.\n\n"
        "Just paste your text and I'll do the rest!\n\n"
        "Use /help to see all commands."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📝 *How to use this bot:*\n\n"
        "1. Send me any AI-generated text.\n"
        "2. I'll rewrite it to sound more natural, authentic, and human.\n\n"
        "*Commands:*\n"
        "/start - Start the bot and see instructions\n"
        "/help - Show this message\n"
        "/humanize <text> - Explicitly rewrite the given text\n\n"
        "*Example:*\n"
        "`/humanize This is a very generic AI paragraph.`",
        parse_mode="Markdown",
    )


async def humanize_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /humanize with inline arguments."""
    if not context.args:
        await update.message.reply_text(
            "Please send the text you want me to humanize. Example:\n"
            "/humanize This is some AI-generated text."
        )
        return

    text = " ".join(context.args)
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )
    rewritten = await humanize_text(text)
    await update.message.reply_text(rewritten)


# ---------- Message Handler ----------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    if not user_text:
        return

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )
    rewritten = await humanize_text(user_text)
    await update.message.reply_text(rewritten)


# ---------- Main ----------
def main():
    if not TELEGRAM_TOKEN or not GEMINI_API_KEY:
        raise ValueError("Please set TELEGRAM_BOT_TOKEN and GEMINI_API_KEY")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("humanize", humanize_command))

    # Free-text messages (excluding commands)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
