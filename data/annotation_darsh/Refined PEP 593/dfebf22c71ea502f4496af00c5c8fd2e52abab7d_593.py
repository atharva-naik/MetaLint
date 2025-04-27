import logging
import re

import bson
from common import event_log_stream

from common.models.event_log_object import EventLogObject
from common.services.queue_service import QueueService

logger = logging.getLogger("cipher-codec")


class EventLogService(object):
    """ Event log service - user to extract a log message,
    build this into an event object and send this to the
    event logging service via a rabbit mq queue.

    """

    def __init__(self):
        """ Initialize connection to queue service. """

        self.queue_service = QueueService()

    def log_event(self,):
        """ Entry point into the event log service.
        Receives a message string from the event log `io.StringIO` stream,
        converts this to am EventLogObject and sends this to the queue service.
        """

        logged_message = event_log_stream.getvalue()
        event_log_stream.truncate(0)
        event_log_stream.seek(0)

        bson_data = self.process_message_string_parser(logged_message)

        self.queue_service.send_message(self.queue_service.MESSAGE_LOGGING_QUEUE, bson_data)

    def process_message_string_parser(self, event_message: str) -> bson:
        """ Parses the event message string using regex to find values using a set pattern.
        Inserts these values into the relative fields and builds the EventLogObject.

        :param event_message: The logged message string.
        :return: A BSON string object
        """

        log_object = EventLogObject()

        log_object.log_time = re.search(r'TIME: (?:\S)(?<=\[)(.*?)(?=,)', event_message).group(1)
        log_object.log_level = re.search(r'LEVEL: (?:\S)(?<=\[)(.*?)(?=\])]', event_message).group(1)
        log_object.log_module = re.search(r'MODULE: (?:\S)(?<=\[)(.*?)(?=\])]', event_message).group(1)
        log_object.log_function = re.search(r'FUNCTION: (?:\S)(?<=\[)(.*?)(?=\])]', event_message).group(1)
        log_object.log_message = re.search(r'MESSAGE: (?:\S)(?<=\[)(.*)(?=\])', event_message).group(1)

        return bson.dumps(log_object.to_dict())