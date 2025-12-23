# main.py
from bot import bot

print("Бот запускается...")
bot.polling(none_stop=True, interval=0)