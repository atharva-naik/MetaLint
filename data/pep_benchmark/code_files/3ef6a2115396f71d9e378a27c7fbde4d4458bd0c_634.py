import sys
from datetime import datetime
from BaseDetector import BaseDetector


# SilentTestDetector never detect
class SilentTestDetector(BaseDetector):
    def __init__(self):
        BaseDetector.__init__(self)
        self.detectorName = 'SilentDetector'

    def execute(self):
        BaseDetector.execute(self)

# AlertingTestDetector always detect
class AlertingTestDetector(BaseDetector):
    def __init__(self, priority = 'DEBUG'):
        BaseDetector.__init__(self, priority)
        self.detectorName = 'AlertingDetector'
    def execute(self):
        BaseDetector.execute(self)
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        detectionTime = timestamp
        description = ''
        name = self.detectorName
        priority = self.priority
        body = ''

        BaseDetector.sendAlarm(self,timestamp, name, priority, description, detectionTime, body)

# HaltingTestDetector never returns
class HaltingTestDetector(BaseDetector):
    def __init__(self):
        BaseDetector.__init__(self)
        self.detectorName = 'HaltingDetector'
    def execute(self):
        BaseDetector.execute(self)
        while True:
            pass

# RaiseErrorDetector always fails with code error, returns -1
class RaiseErrorTestDetector(BaseDetector):
    def __init__(self):
        BaseDetector.__init__(self)
        self.detectorName = 'RaiseErrorDetector'

    def execute(priorityself):
        BaseDetector.execute(self)
        raise Exception('Error Raiser for ErrorRaiserDetector')

def main(args):
    detectorName = args[0]
    # print detectorName

    priority = None
    if len(args) > 1:
        priority = args[1]

    if detectorName == 'SilentDetector':
        myDetector = SilentTestDetector()
    elif detectorName == 'AlertingDetector':
        myDetector = AlertingTestDetector(priority)
    elif detectorName == 'HaltingDetector':
        myDetector = HaltingTestDetector()
    elif detectorName == 'RaiseErrorDetector':
        myDetector = RaiseErrorTestDetector()

    myDetector.execute()


if __name__ == '__main__':
    # print sys.argv[1:]
    main(sys.argv[1:])