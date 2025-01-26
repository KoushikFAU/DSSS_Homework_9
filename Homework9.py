import torch
from transformers import pipeline
import telebot
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Telegram bot
BOT_TOKEN = '7250658195:AAFXvCHyMLTYMpE2VB4DGCxok0g-Vdh7TTU'
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN not found!")
bot = telebot.TeleBot(BOT_TOKEN)

# Initialize TinyLlama pipeline
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

# Command to start biot
@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    bot.reply_to(message, "Hey,I am your AI assistant. How are you doing?")

# Define message handler
@bot.message_handler(func=lambda msg: True)
def respond_to_message(message):
    user_message = message.text
    print(f"User Message: {user_message}")
    
    messages = [
        {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate."},
        {"role": "user", "content": user_message},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Generates a response using tiny llama
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    response = outputs[0]["generated_text"]

    # Clean up the response to remove the unwanted parts :(
    response = response.split("</s>")[-1].strip()
    bot.reply_to(message, response)

# bot's polling to listen for messages
bot.infinity_polling()