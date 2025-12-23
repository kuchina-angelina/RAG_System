# bot.py
import telebot
import tempfile
import requests
from config import token
from pdf_loader import read_pdf, simple_approach
from database import create_embeddings,  add_doc_to_db, embedding_model, collection
from retrieval import  semantic_search
from reranker import rerank, reranker_model
from augmentation import prepare_context
from generate_qwen import generate_qwen, model, tokenizer

import os
import hashlib
import traceback


bot = telebot.TeleBot(token)
# Хранилище данных пользователя
user_data = {}

# === Вспомогательные функции ===
def get_file_hash(file_path):
    """Вычисляет SHA256-хэш для проверки дубликатов"""
    hash_func = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def list_user_docs(user_id):
    """Формирует список загруженных документов"""
    data = user_data.get(user_id, {})
    docs = data.get("documents", [])
    if not docs:
        return "У вас пока нет загруженных документов."
    text = "Ваши документы:\n\n"
    for index, document in enumerate(docs, start=1):
        text += f"{index}. {document['name']}\n"
    return text


# === Обработчики ===
@bot.message_handler(commands=["start"])
def start(message):
    bot.send_message(
        message.chat.id,
        "Привет! Я бот для анализа PDF-документов.\n"
        "Отправьте PDF-файл, чтобы я мог с ним работать."
    )


@bot.message_handler(content_types=["document"])
def handle_document(message):
    """Обработка загруженного PDF"""
    if message.document.mime_type != 'application/pdf':
        bot.send_message(message.chat.id, "Отправьте PDF-файл.")
        return

    try:
        file_info = bot.get_file(message.document.file_id)
        file_url = f'https://api.telegram.org/file/bot{bot.token}/{file_info.file_path}'
        response = requests.get(file_url)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name


        file_hash = get_file_hash(temp_file_path)
        user_id = message.chat.id
        user_entry = user_data.setdefault(user_id, {"documents": []})
        docs = user_entry["documents"]

        # Проверка дубликатов
        for doc in docs:
            if doc["hash"] == file_hash:
                bot.send_message(user_id, f"Файл *{message.document.file_name}* уже был загружен.", parse_mode="Markdown")
                return


        # === Обработка документа ===
        docs_obj = read_pdf(temp_file_path)

        if not docs_obj or all(not page.page_content.strip() for page in docs_obj):
            os.unlink(temp_file_path)
            bot.send_message(
                message.chat.id,
                "Файл не содержит текста. Отправьте другой PDF!"
            )
            return

        all_text_chunks = simple_approach(docs_obj)
        print(len(tokenizer.encode(all_text_chunks[0]["page_content"])))
        all_embeddings = create_embeddings(embedding_model, all_text_chunks)
        add_doc_to_db(all_text_chunks, all_embeddings, collection)

        # Сохраняем в user_data
        if "all_text_chunks" not in user_entry:
            user_entry["all_text_chunks"] = []

        # Добавляем новые чанки, не стирая старые
        user_entry["all_text_chunks"].extend(all_text_chunks)
        user_entry["processed"] = True


        # Добавляем документ в список
        docs.append({
            "name": message.document.file_name,
            "path": temp_file_path,
            "hash": file_hash,
            "chunks": all_text_chunks
        })

        markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add("Задать вопрос", "Загрузить ещё документ", "Мои документы")

        bot.send_message(
            user_id,
             f"Файл *{message.document.file_name}* успешно обработан! Выберете в меню неохдимую опцию для дальнейших действий",
            reply_markup=markup,
            parse_mode="Markdown"
        )

    except Exception as e:
        traceback.print_exc()
        bot.send_message(message.chat.id, f"Ошибка при обработке файла: {e}")


@bot.message_handler(func=lambda msg: msg.text == "Мои документы")
def show_documents(message):
    """Показ списка загруженных документов"""
    user_id = message.chat.id
    docs_text = list_user_docs(user_id)
    bot.send_message(user_id, docs_text)


@bot.message_handler(func=lambda msg: msg.text == "Загрузить ещё документ")
def upload_new_doc(message):
    bot.send_message(message.chat.id, "📎 Отправьте новый PDF-файл.")


@bot.message_handler(func=lambda msg: msg.text == "Задать вопрос")
def ask_question(message):
    """Начало взаимодействия с вопросом"""
    user_id = message.chat.id
    data = user_data.get(user_id, {})

    if not data.get("processed"):
        bot.send_message(user_id, "Сначала загрузите PDF-документ.")
        return

    bot.send_message(user_id, "Введите ваш вопрос:")
    data["state"] = "await_question"


@bot.message_handler(func=lambda msg: user_data.get(msg.chat.id, {}).get("state") == "await_question")
def handle_question(message):
    """Обработка запроса и генерация ответа"""
    user_id = message.chat.id
    query = message.text
    print('query', query)
    data = user_data[user_id]

    try:
        print("Выполняем семантический поиск...")
        top_embeddings_semantic_search = semantic_search(embedding_model, collection, query)
                                        
        print("Выполняем реранкинг...")
        top_chunks_rerank = rerank(reranker_model, query, top_embeddings_semantic_search, top_n=3)
        print(top_chunks_rerank)
        
        print("Формируем контекст...")
        context, page_numbers = prepare_context(top_chunks_rerank)
        print(context)

        # Получаем имя документа (если загружено)
        file_name = None
        if user_data[user_id]["documents"]:
            file_name = user_data[user_id]["documents"][-1]["name"]  # последний загруженный документ

        print("Выполняем генерацию...")
        answer = generate_qwen(model, tokenizer, query, context, page_numbers, file_name)

        print('Закончили генерацию...')

        bot.send_message(user_id, answer)

    except Exception as e:
        traceback.print_exc()
        bot.send_message(user_id, f"Ошибка при обработке запроса: {e}")

