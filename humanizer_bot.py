import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from openai import OpenAI

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hi! Send me any AI-generated text and I'll rewrite it to sound more natural, authentic, and human.\n\n"
        "Just paste your text and I'll do the rest!"
    )

async def humanize_text(text: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that rewrites text to sound more natural, authentic, and human. "
                        "Use contractions, vary sentence structure, avoid AI-sounding phrases, and make it conversational. "
                        "Keep the original meaning intact."
                    )
                },
                {
                    "role": "user",
                    "content": f"Rewrite the following text to sound more natural, authentic, and human:\n\n{text}"
                }
            ],
            temperature=0.8,
            max_tokens=1500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI error: {e}")
        return "⚠️ Sorry, I couldn't process that right now. Please try again later."

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    if not user_text:
        return
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    rewritten = await humanize_text(user_text)
    await update.message.reply_text(rewritten)

def main():
    if not TELEGRAM_TOKEN or not OPENAI_API_KEY:
        raise ValueError("Please set TELEGRAM_BOT_TOKEN and OPENAI_API_KEY")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
