import random
from datetime import datetime, timezone

from django.core.management.base import BaseCommand, CommandError
from django.core.management import call_command

from trainings.models import Training
from tasks.models import Task, TaskType


def select_random_keyword(keywords):
    return random.choice(keywords)

class Command(BaseCommand):
    help = 'Create training tasks if they are due'

    def handle(self, *args, **options):
        # get all
        trainings = Training.objects.all()

        for training in trainings:
            # get last training task for agent of the given training
            training_type = TaskType.objects.get(pk=2)

            create_task = False

            try:
                # check the date of the last task
                tasks = Task.objects.filter(agent=training.agent,
                                            type=training_type,
                                            status=1)
                task = tasks.latest('created_at')

                # calculate the time diff
                time_diff = datetime.now(timezone.utc) - task.created_at
                diff_seconds = time_diff.total_seconds()

                if diff_seconds > training.interval:
                    # time to create a new task
                    create_task = True

            except Task.DoesNotExist:
                create_task = True

            if create_task:
                keyword = select_random_keyword(training.keywords.all())
                call_command('create_search_task', keyword, no_out=False,
                             type=2, agent=training.agent.id)
                self.stdout.write(
                    self.style.SUCCESS(
                        'Created a training task for %s' % (training.agent)
                    )
                )
            else:
                self.stdout.write(
                    self.style.SUCCESS(
                        'No training task needed for %s' % (training.agent)
                    )
                )
