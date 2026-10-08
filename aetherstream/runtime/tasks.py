"""Request-owned work that may outlive its downstream connection.

Unlike singleton scheduled jobs, these tasks stay on the originating instance.
The release drain barrier keeps that instance alive until they settle.
"""
import asyncio

_TASKS = set()


def spawn_detached(coroutine, *, name=None):
    task = asyncio.create_task(coroutine, name=name)
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return task


def pending_count():
    return sum(not task.done() for task in _TASKS)
