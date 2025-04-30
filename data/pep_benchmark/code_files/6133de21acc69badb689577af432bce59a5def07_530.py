import logging

from databases import Database
from uuid import UUID

from core.security.password import get_password_hash

from core.models.user import BottifyUserInModel, BottifyUserModel

from core.database.tables.bottify_user import get_bottify_user_table

from core.database.helpers import build_model_from_row

user_table = get_bottify_user_table()


async def read_user_by_id(database: Database, user_id: int):
    query = user_table.select().where(user_table.c.id == user_id).limit(1)
    row = await database.fetch_one(query)
    return build_model_from_row(row, BottifyUserModel)


async def read_user_by_guid(database: Database, guid_in: UUID):
    if isinstance(guid_in, UUID):
        user_guid = guid_in
    elif isinstance(guid_in, str):
        try:
            user_guid = UUID(guid_in)
        except ValueError as e:
            logging.error(f"Read User by Guid:Failed to Parse UUID from String")
            return None
    else:
        logging.error(
            f"Read User by Guid:User GUID must be either UUID or String:Got: {type(guid_in)}"
        )
        return None
    query = user_table.select().where(user_table.c.guid == user_guid).limit(1)
    row = await database.fetch_one(query)
    return build_model_from_row(row, BottifyUserModel)


async def read_user_by_username(database: Database, username: str):
    if not isinstance(username, str):
        logging.error(
            f"Read User by Username:Username Must be type String:Got: {type(username)}"
        )
    query = user_table.select().where(user_table.c.username == username).limit(1)
    row = await database.fetch_one(query)
    return build_model_from_row(row, BottifyUserModel)


async def create_user(database: Database, user_in: BottifyUserInModel):
    query = user_table.insert()
    hashed_password = get_password_hash(user_in.password)
    success = False
    if not hashed_password:
        logging.error(
            f"Create User Error:Failed to Hash Password:User Data: {user_in.json()}"
        )
        return success
    user_data = user_in.dict(exclude={"password"})
    user_data.update({"hashed_password": hashed_password})
    await database.execute(query, values=user_data)
    success = True
    return success


async def read_users(database: Database, limit: int):
    if not isinstance(limit, int):
        logging.error(
            f"Read Users Error:Limit Param Must be an Integer:Got: {type(limit)}"
        )
    query = user_table.select().limit(limit)
    users = []
    async for row in database.iterate(query):
        users.append(build_model_from_row(row, BottifyUserModel))
    if not users:
        logging.error(f"Read Users Error:Failed to Read Any Users")
    return users
