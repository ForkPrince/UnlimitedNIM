from __future__ import annotations

import asyncio
from unittest import mock

import pytest
import pytest_asyncio

from app import QueueFull, RateLimiter
from tests.conftest import cancel_all, settle


def make_limiter(fake_clock, max_rpm=40, max_queue=500, retry_ttl=300.0) -> RateLimiter:
    return RateLimiter(
        max_rpm=max_rpm,
        max_queue=max_queue,
        retry_ttl=retry_ttl,
        clock=fake_clock,
    )


async def acquire_many(limiter, n, priority=3):
    return await asyncio.gather(*[limiter.acquire(priority) for _ in range(n)])


async def release_all_and_reset(limiter, fake_clock, slots) -> None:
    for s in slots:
        await limiter.release(s)
    fake_clock.advance(60)
    await limiter.drain()
    await settle(10)


@pytest.mark.asyncio
async def test_tc001_single_request_fires_immediately(fake_clock):
    limiter = make_limiter(fake_clock)
    slot = await limiter.acquire(3)
    assert slot.request_id
    assert limiter.window_count == 1
    assert limiter.queue_depth == 0
    assert slot.request_id in limiter.in_flight


@pytest.mark.asyncio
async def test_tc002_40_simultaneous_all_fire(fake_clock):
    limiter = make_limiter(fake_clock)
    slots = await acquire_many(limiter, 40)
    assert len(slots) == 40
    assert limiter.window_count == 40
    assert limiter.queue_depth == 0
    assert len(limiter.in_flight) == 40


@pytest.mark.asyncio
async def test_tc003_41st_request_queued_when_window_full(fake_clock):
    limiter = make_limiter(fake_clock)
    await acquire_many(limiter, 40)
    task = asyncio.create_task(limiter.acquire(3))
    await settle()
    assert limiter.window_count == 40
    assert limiter.queue_depth == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert limiter.queue_depth == 0
    assert limiter.total_cancelled == 1


@pytest.mark.asyncio
async def test_tc004_window_reset_fires_10_from_queue(fake_clock):
    limiter = make_limiter(fake_clock)
    await acquire_many(limiter, 40)
    tasks = [asyncio.create_task(limiter.acquire(3)) for _ in range(10)]
    await settle()
    assert limiter.queue_depth == 10
    fake_clock.advance(60)
    await limiter.drain()
    results = await asyncio.gather(*tasks)
    assert len(results) == 10
    assert limiter.window_count == 10
    assert limiter.queue_depth == 0


@pytest.mark.asyncio
async def test_tc005_window_reset_fires_exactly_40_of_75(fake_clock):
    limiter = make_limiter(fake_clock)
    await acquire_many(limiter, 40)
    tasks = [asyncio.create_task(limiter.acquire(3)) for _ in range(75)]
    await settle()
    assert limiter.queue_depth == 75
    fake_clock.advance(60)
    await limiter.drain()
    await settle()
    done = sum(1 for t in tasks if t.done())
    assert done == 40
    assert limiter.window_count == 40
    assert limiter.queue_depth == 35
    await cancel_all(*(t for t in tasks if not t.done()))


@pytest.mark.asyncio
async def test_tc006_window_reset_queue_empty_idle(fake_clock):
    limiter = make_limiter(fake_clock)
    await acquire_many(limiter, 40)
    fired = limiter.total_fired
    fake_clock.advance(60)
    await limiter.drain()
    assert limiter.window_count == 0
    assert limiter.total_fired == fired
    assert limiter.queue_depth == 0


@pytest.mark.asyncio
async def test_tc007_all_inflight_from_previous_minute_new_request_fires(fake_clock):
    limiter = make_limiter(fake_clock)
    await acquire_many(limiter, 40)
    fake_clock.advance(60)
    slot = await limiter.acquire(3)
    assert limiter.window_count == 1
    assert slot.request_id


@pytest.mark.asyncio
async def test_tc008_previous_minute_inflight_60_queued_fire_40(fake_clock):
    limiter = make_limiter(fake_clock)
    await acquire_many(limiter, 40)
    tasks = [asyncio.create_task(limiter.acquire(3)) for _ in range(60)]
    await settle()
    assert limiter.queue_depth == 60
    fake_clock.advance(60)
    new_task = asyncio.create_task(limiter.acquire(3))
    await settle(6)
    assert limiter.window_count == 40
    assert limiter.queue_depth == 21
    done = sum(1 for t in [*tasks, new_task] if t.done())
    assert done == 40
    await cancel_all(*(t for t in [*tasks, new_task] if not t.done()))


@pytest.mark.asyncio
async def test_tc009_mixed_inflight_uses_available_slots(fake_clock):
    limiter = make_limiter(fake_clock)
    await acquire_many(limiter, 20)
    fake_clock.advance(60)
    await acquire_many(limiter, 20)
    assert limiter.window_count == 20
    results = await acquire_many(limiter, 5)
    assert limiter.window_count == 25
    assert limiter.queue_depth == 0
    assert len(results) == 5
    assert limiter.total_fired == 45


@pytest.mark.asyncio
async def test_tc010_previous_minute_completion_fires_40(fake_clock):
    limiter = make_limiter(fake_clock)
    slots = await acquire_many(limiter, 40)
    queued = [asyncio.create_task(limiter.acquire(3)) for _ in range(50)]
    await settle()
    assert limiter.queue_depth == 50
    fake_clock.advance(60)
    await limiter.release(slots[0])
    await settle()
    assert limiter.window_count == 40
    assert limiter.queue_depth == 10
    done = sum(1 for t in queued if t.done())
    assert done == 40
    await cancel_all(*(t for t in queued if not t.done()))


@pytest.mark.asyncio
async def test_tc011_previous_minute_completion_fires_exactly_one(fake_clock):
    limiter = make_limiter(fake_clock)
    old = await limiter.acquire(3)
    fake_clock.advance(60)
    await acquire_many(limiter, 39)
    assert limiter.window_count == 39
    queued = [asyncio.create_task(limiter.acquire(3)) for _ in range(5)]
    await settle()
    await limiter.release(old)
    await settle()
    assert limiter.window_count == 40
    assert limiter.queue_depth == 4
    done = sum(1 for t in queued if t.done())
    assert done == 1
    await cancel_all(*(t for t in queued if not t.done()))


@pytest.mark.asyncio
async def test_tc012_current_minute_completion_fires_nothing(fake_clock):
    limiter = make_limiter(fake_clock)
    slots = await acquire_many(limiter, 40)
    queued = [asyncio.create_task(limiter.acquire(3)) for _ in range(5)]
    await settle()
    await limiter.release(slots[0])
    await settle()
    assert limiter.window_count == 40
    assert limiter.queue_depth == 5
    assert all(not t.done() for t in queued)
    await cancel_all(*queued)


@pytest.mark.asyncio
async def test_tc013_highest_priority_fires_first(fake_clock):
    limiter = make_limiter(fake_clock)
    slots = await acquire_many(limiter, 40)
    t3 = asyncio.create_task(limiter.acquire(3))
    t1 = asyncio.create_task(limiter.acquire(1))
    t5 = asyncio.create_task(limiter.acquire(5))
    await settle()
    assert limiter.queue_depth == 3
    assert not t3.done() and not t1.done() and not t5.done()
    await release_all_and_reset(limiter, fake_clock, slots)
    r3, r1, r5 = await asyncio.gather(t3, t1, t5)
    rids = list(limiter.in_flight)
    assert rids[0] == r1.request_id
    assert rids[1] == r3.request_id
    assert rids[2] == r5.request_id


@pytest.mark.asyncio
async def test_tc014_fifo_within_same_priority(fake_clock):
    limiter = make_limiter(fake_clock)
    slots = await acquire_many(limiter, 40)
    t_a = asyncio.create_task(limiter.acquire(1))
    await settle()
    fake_clock.advance(5)
    t_b = asyncio.create_task(limiter.acquire(1))
    await settle()
    assert limiter.queue_depth == 2
    await release_all_and_reset(limiter, fake_clock, slots)
    r_a, r_b = await asyncio.gather(t_a, t_b)
    rids = list(limiter.in_flight)
    assert rids[0] == r_a.request_id
    assert rids[1] == r_b.request_id


@pytest.mark.asyncio
async def test_tc015_new_high_priority_jumps_ahead(fake_clock):
    limiter = make_limiter(fake_clock)
    slots = await acquire_many(limiter, 40)
    t5 = asyncio.create_task(limiter.acquire(5))
    await settle()
    t1 = asyncio.create_task(limiter.acquire(1))
    await settle()
    assert limiter.queue_depth == 2
    await release_all_and_reset(limiter, fake_clock, slots)
    r5, r1 = await asyncio.gather(t5, t1)
    rids = list(limiter.in_flight)
    assert rids[0] == r1.request_id
    assert rids[1] == r5.request_id


@pytest.mark.asyncio
async def test_tc016_40_low_priority_eventually_fire(fake_clock):
    limiter = make_limiter(fake_clock)
    await acquire_many(limiter, 40)
    tasks = [asyncio.create_task(limiter.acquire(5)) for _ in range(40)]
    await settle()
    assert limiter.queue_depth == 40
    fake_clock.advance(60)
    await limiter.drain()
    await settle()
    assert limiter.window_count == 40
    assert limiter.queue_depth == 0
    assert all(t.done() for t in tasks)


@pytest.mark.asyncio
async def test_tc017_starvation_promotion_after_ttl(fake_clock):
    limiter = make_limiter(fake_clock, retry_ttl=300.0)
    await acquire_many(limiter, 40)
    t5 = asyncio.create_task(limiter.acquire(5))
    await settle()
    assert limiter.queue_depth == 1
    all_p1s: list = []
    for cycle in range(5):
        p1s = [asyncio.create_task(limiter.acquire(1)) for _ in range(40)]
        all_p1s.extend(p1s)
        await settle()
        fake_clock.advance(60)
        await limiter.drain()
        await settle()
        if cycle < 4:
            assert not t5.done()
    assert t5.done()
    fake_clock.advance(60)
    await limiter.drain()
    await settle()
    await asyncio.gather(*all_p1s)
    assert all(t.done() for t in all_p1s)


@pytest.mark.asyncio
async def test_tc021_restart_resets_state(fake_clock):
    limiter = make_limiter(fake_clock)
    await acquire_many(limiter, 30)
    assert limiter.window_count == 30
    fresh = RateLimiter(clock=fake_clock)
    assert fresh.window_count == 0
    assert fresh.queue_depth == 0
    assert len(fresh.in_flight) == 0


@pytest.mark.asyncio
async def test_tc022_clock_jump_forward_resets_cleanly(fake_clock):
    limiter = make_limiter(fake_clock)
    await acquire_many(limiter, 20)
    fake_clock.advance(90)
    slot = await limiter.acquire(3)
    assert limiter.window_count == 1
    assert limiter.queue_depth == 0


@pytest.mark.asyncio
async def test_tc023_clock_jump_backward_does_not_reset(fake_clock):
    limiter = make_limiter(fake_clock)
    slots = await acquire_many(limiter, 40)
    queued = [asyncio.create_task(limiter.acquire(3)) for _ in range(3)]
    await settle()
    assert limiter.queue_depth == 3
    fake_clock.set(fake_clock.t - 30)
    await limiter.release(slots[0])
    await settle()
    assert limiter.window_count == 40
    assert limiter.queue_depth == 3
    assert all(not t.done() for t in queued)
    await cancel_all(*queued)


@pytest.mark.asyncio
async def test_tc024_idle_then_100_burst(fake_clock):
    limiter = make_limiter(fake_clock)
    tasks = [asyncio.create_task(limiter.acquire(3)) for _ in range(100)]
    await settle()
    assert limiter.window_count == 40
    assert limiter.queue_depth == 60
    fake_clock.advance(60)
    await limiter.drain()
    await settle()
    assert limiter.window_count == 40
    assert limiter.queue_depth == 20
    fake_clock.advance(60)
    await limiter.drain()
    await settle()
    assert limiter.window_count == 20
    assert limiter.queue_depth == 0
    await asyncio.gather(*tasks)
    assert all(t.done() for t in tasks)


@pytest.mark.asyncio
async def test_tc020_stale_inflight_swept(fake_clock):
    limiter = make_limiter(fake_clock)
    slots = await acquire_many(limiter, 5)
    fake_clock.advance(400)
    await limiter.sweep(max_stream_age=300.0)
    assert len(limiter.in_flight) == 0
    assert limiter.total_cancelled == 5
    for s in slots:
        assert s.request_id not in limiter.in_flight


@pytest.mark.asyncio
async def test_tc028_two_concurrent_one_slot_atomic(fake_clock):
    limiter = make_limiter(fake_clock)
    await acquire_many(limiter, 39)
    t1 = asyncio.create_task(limiter.acquire(3))
    t2 = asyncio.create_task(limiter.acquire(3))
    await settle()
    assert limiter.window_count == 40
    assert limiter.queue_depth == 1
    t1.cancel()
    t2.cancel()
    await cancel_all(t1, t2)


@pytest.mark.asyncio
async def test_tc029_queue_full_raises(fake_clock):
    limiter = make_limiter(fake_clock, max_rpm=1, max_queue=2)
    await acquire_many(limiter, 1)
    t2 = asyncio.create_task(limiter.acquire(3))
    await settle()
    t3 = asyncio.create_task(limiter.acquire(3))
    await settle()
    assert limiter.queue_depth == 2
    with pytest.raises(QueueFull) as excinfo:
        await limiter.acquire(3)
    assert excinfo.value.retry_after >= 1
    await cancel_all(t2, t3)


@pytest.mark.asyncio
async def test_tc030_queued_client_disconnect_removes(fake_clock):
    limiter = make_limiter(fake_clock)
    await acquire_many(limiter, 40)
    task = asyncio.create_task(limiter.acquire(3))
    await settle()
    assert limiter.queue_depth == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert limiter.queue_depth == 0
    assert limiter.window_count == 40
    assert limiter.total_cancelled == 1


@pytest.mark.asyncio
async def test_tc031_inflight_disconnect_keeps_window_count(fake_clock):
    limiter = make_limiter(fake_clock)
    slots = await acquire_many(limiter, 40)
    queued = [asyncio.create_task(limiter.acquire(3)) for _ in range(3)]
    await settle()
    await limiter.release(slots[0])
    await settle()
    assert limiter.window_count == 40
    assert limiter.queue_depth == 3
    assert all(not t.done() for t in queued)
    await cancel_all(*queued)