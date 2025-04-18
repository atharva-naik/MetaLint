from aiogram import types
from aiogram.dispatcher.filters.builtin import CommandStart
from aiogram.types import CallbackQuery

from data.config import *
from keyboards.default import main_keyboard
from keyboards.inline import *
from loader import dp, bot
from utils.bot_utils import delete_last_message_call, delete_last_message


@dp.message_handler(CommandStart(), user_id=users)
async def bot_start(message: types.Message):
    current_user_paths[message.from_user.id] = data_base_path
    await message.answer(text="Здравствуйте!",
                         reply_markup=main_keyboard)
    await message.answer(text="Бот готов к использованию")
    await message.answer(text="Выберите раздел:",
                         reply_markup=show_files_and_folders(data_base_path, message.from_user.id))


@dp.message_handler(text="Главный раздел", user_id=users)
async def show_folders(message: types.Message):
    await delete_last_message(message)
    current_user_paths[message.from_user.id] = data_base_path
    await message.answer(
        text='Выберите раздел или файл:',
        reply_markup=show_files_and_folders(data_base_path, message.from_user.id))


@dp.callback_query_handler(f_and_f_callback.filter(path='more'))
async def show_current_folder(call: types.CallbackQuery, callback_data: dict):
    await delete_last_message_call(call)
    await call.message.answer(
        text=f"Продолжение:",
        reply_markup=show_files_and_folders(current_user_paths[call.from_user.id], call.from_user.id,
                                            n=files_in_folders[f'{call.from_user.id}nn']))


@dp.callback_query_handler(f_and_f_callback.filter(path='back'), user_id=users)
async def show_current_folder(call: CallbackQuery, callback_data: dict):
    await delete_last_message_call(call)
    arr = current_user_paths[call.from_user.id].split(r'\\')
    del arr[-1]
    current_user_paths[call.from_user.id] = r'\\'.join(arr)
    await call.message.answer(
        text='Выберите раздел или файл',
        reply_markup=show_files_and_folders(current_user_paths[call.from_user.id], call.from_user.id))


@dp.callback_query_handler(f_and_f_callback.filter(), user_id=users)
async def show_current_folder(call: CallbackQuery, callback_data: dict):
    await delete_last_message_call(call)
    choosed_folder = files_in_folders[call.from_user.id][int(callback_data.get('path'))]
    current_path = rf'{current_user_paths[call.from_user.id]}\\{choosed_folder}'
    if os.path.isdir(current_path):
        current_user_paths[call.from_user.id] = current_path
        await call.message.answer(
            text='Выберите раздел',
            reply_markup=show_files_and_folders(current_user_paths[call.from_user.id], call.from_user.id))

    else:
        name = current_path.split('.')

        if name[-1] == 'txt':
            with open(current_path, encoding='UTF-8') as f:
                text = f.read()
                f.close()
            await call.message.answer(text=text)

        else:
            await bot.send_document(chat_id=call.from_user.id,
                                    document=open(current_path, 'rb'))